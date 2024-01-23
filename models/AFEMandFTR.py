import numpy as np

from .ffc import *
from .layers import *
from .VAN import VANBlock
import torchvision


class ResnetBlock_remove_IN(nn.Module):
    def __init__(self, dim, dilation=1):
        super(ResnetBlock_remove_IN, self).__init__()

        self.ffc1 = FFC_BN_ACT(dim, dim, 3, 0.75, 0.75, stride=1, padding=1, dilation=dilation, groups=1, bias=False,
                               norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, enable_lfu=False)

        self.ffc2 = FFC_BN_ACT(dim, dim, 3, 0.75, 0.75, stride=1, padding=1, dilation=1, groups=1, bias=False,
                               norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, enable_lfu=False)

    def forward(self, x):
        output = x
        _, c, _, _ = output.shape
        output = torch.split(output, [c - int(c * 0.75), int(c * 0.75)], dim=1)
        x_l, x_g = self.ffc1(output)
        output = self.ffc2((x_l, x_g))
        output = torch.cat(output, dim=1)
        output = x + output

        return output


class GlobalFilter(nn.Module):
    def __init__(self, dim, h=32, w=17):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x):
        _, _, old_H, old_W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        _, H, W, _ = x.shape
        if H != self.h or W != self.w:
            new_weight = self.complex_weight.reshape(1, self.h, self.w, -1).permute(0, 3, 1, 2)
            new_weight = torch.nn.functional.interpolate(
                new_weight, size=(H, W), mode='bicubic', align_corners=True).permute(0, 2, 3,
                                                                                     1).reshape(
                H, W, -1, 2).contiguous()
        else:
            new_weight = self.complex_weight

        weight = torch.view_as_complex(new_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(old_H, old_W), dim=(1, 2), norm='ortho')
        x = x.permute(0, 3, 1, 2)
        return x


class GFBlock(nn.Module):

    def __init__(self, dim, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False,
                 padding_type='reflect',
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, h=32, w=17):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.norm2 = norm_layer(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        self.act = act_layer(inplace=True)

    def forward(self, x):
        x = x + self.act(self.norm2(self.conv(self.norm1(self.filter(x)))))
        return x


class MaskedSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids):
        """`input_ids` is expected to be [bsz x seqlen]."""
        return super().forward(input_ids)


class MultiLabelEmbedding(nn.Module):
    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_positions, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, input_ids):
        # input_ids:[B,HW,4](onehot)
        out = torch.matmul(input_ids, self.weight)  # [B,HW,dim]
        return out


class base_model(nn.Module):
    def __init__(self):
        super().__init__()
        act = nn.SiLU
        # encoder and decoder for re_img feature
        self.enconv1 = nn.Sequential(*[nn.ReflectionPad2d(3),
                                       nn.Conv2d(in_channels=7, out_channels=64, kernel_size=7, padding=0),
                                       nn.BatchNorm2d(64), act(True)])
        self.enconv2 = nn.Sequential(*[nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(128), act(True),
                                       VANBlock(dim=128, kernel_size=21,
                                                dilation=3, act_layer=act)])
        self.enconv3 = nn.Sequential(*[nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(256), act(True),
                                       VANBlock(dim=256, kernel_size=21,
                                                dilation=3, act_layer=act)])
        self.enconv4 = nn.Sequential(*[nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(512), act(True),
                                       VANBlock(dim=512, kernel_size=21,
                                                dilation=3, act_layer=act)])

        self.GateConv1 = nn.Sequential(*[nn.ReflectionPad2d(3),
                                         GateConv(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=0),
                                         nn.BatchNorm2d(64), act(True)])
        self.GateConv2 = nn.Sequential(*[GateConv(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
                                         nn.BatchNorm2d(128), act(True)])
        self.GateConv3 = nn.Sequential(
            *[GateConv(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
              nn.BatchNorm2d(256), act(True)])
        self.GateConv4 = nn.Sequential(
            *[GateConv(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
              nn.BatchNorm2d(512), act(True),
              VANBlock(dim=512, kernel_size=21,
                       dilation=3, act_layer=act)])

        self.GateConvt1 = nn.Sequential(*[GateConv(512, 256, kernel_size=4, stride=2, padding=1, transpose=True),
                                          nn.BatchNorm2d(256), act(True)])
        self.GateConvt2 = nn.Sequential(*[GateConv(256, 128, kernel_size=4, stride=2, padding=1, transpose=True),
                                          nn.BatchNorm2d(128), act(True)])
        self.GateConvt3 = nn.Sequential(*[GateConv(128, 64, kernel_size=4, stride=2, padding=1, transpose=True),
                                          nn.BatchNorm2d(64), act(True)])
        blocks = []
        # resnet blocks
        for i in range(9):
            blocks.append(ResnetBlock_remove_IN(512, 1))

        self.middle = nn.Sequential(*blocks)

        blocks2 = []
        # resnet blocks
        for i in range(3):
            blocks2.append(ResnetBlock(input_dim=512, out_dim=None, dilation=2))

        self.middle2 = nn.Sequential(*blocks2)

        self.deconv1 = nn.Sequential(*[VANBlock(dim=512, kernel_size=21,
                                                dilation=3, act_layer=act),
                                       nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1,
                                                          output_padding=1),
                                       nn.BatchNorm2d(256), act(True)])
        self.deconv2 = nn.Sequential(*[VANBlock(dim=256, kernel_size=21,
                                                dilation=3, act_layer=act),
                                       nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1,
                                                          output_padding=1),
                                       nn.BatchNorm2d(128), act(True)])
        self.deconv3 = nn.Sequential(*[VANBlock(dim=128, kernel_size=21,
                                                dilation=3, act_layer=act),
                                       nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1,
                                                          output_padding=1),
                                       nn.BatchNorm2d(64), act(True)])
        self.deconv4 = nn.Sequential(
            *[nn.ReflectionPad2d(3), nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0)])
        self.act_last = nn.Tanh()

    def forward(self, FIM_inp, AFEM_inp):
        feature_deep = []
        feature_shallow = []
        y = self.GateConv1(AFEM_inp)
        feature_shallow.append(y)
        y = self.GateConv2(y)
        feature_shallow.append(y)

        y = self.GateConv3(y)
        feature_shallow.append(y)
        y = self.GateConv4(y)
        feature_shallow.append(y)
        y = self.middle2(y)

        feature_deep.append(y)
        y = self.GateConvt1(y)
        feature_deep.append(y)
        y = self.GateConvt2(y)
        feature_deep.append(y)
        y = self.GateConvt3(y)
        feature_deep.append(y)
        feature_deep = feature_deep[::-1]
        feature_shallow = feature_shallow[::-1]
        x = self.enconv1[:2](FIM_inp)
        inp = x.to(torch.float32)
        x = self.enconv1[2:](inp)

        x = self.enconv2[:1](x + feature_deep[0])
        x = self.enconv2[1:](x)

        x = self.enconv3[:1](x + feature_deep[1])
        x = self.enconv3[1:](x)

        x = self.enconv4[:1](x + feature_deep[2])
        x = self.enconv4[1:](x)
        x = self.middle(x + feature_deep[3])
        x = self.deconv1[:1](x)
        x = self.deconv1[1:](x + feature_shallow[0])

        x = self.deconv2[:1](x)
        x = self.deconv2[1:](x + feature_shallow[1])

        x = self.deconv3[:1](x)
        x = self.deconv3[1:](x + feature_shallow[2])

        x = self.deconv4(x + feature_shallow[3])

        x = self.act_last(x)

        # x = self.enconv1(FIM_inp)
        # x = self.enconv2(x)
        # x = self.enconv3(x)
        # x = self.enconv4(x)

        # x = self.middle(x)

        # x = self.deconv1(x)
        # x = self.deconv2(x)
        # x = self.deconv3(x)
        # x = self.deconv4(x)

        # x = self.act_last(x)
        return x


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc):
        super().__init__()
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_nc, out_channels=64, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kw, stride=2, padding=padw)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kw, stride=2, padding=padw)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=kw, stride=2, padding=padw)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=kw, stride=1, padding=padw)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 1, kernel_size=kw, stride=1, padding=padw)

    def forward(self, x):
        conv1 = self.conv1(x)

        conv2 = self.conv2(conv1)
        conv2 = self.bn2(conv2.to(torch.float32))
        conv2 = self.act(conv2)

        conv3 = self.conv3(conv2)
        conv3 = self.bn3(conv3.to(torch.float32))
        conv3 = self.act(conv3)

        conv4 = self.conv4(conv3)
        conv4 = self.bn4(conv4.to(torch.float32))
        conv4 = self.act(conv4)

        conv5 = self.conv5(conv4)
        conv5 = self.bn5(conv5.to(torch.float32))
        conv5 = self.act(conv5)

        conv6 = self.conv6(conv5)

        outputs = conv6

        return outputs, [conv1, conv2, conv3, conv4, conv5]