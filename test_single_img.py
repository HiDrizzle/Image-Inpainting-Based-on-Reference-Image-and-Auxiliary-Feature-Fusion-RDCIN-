from skimage.color import rgb2gray
from src.models.AFEMandFTR import *
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F


def load_grad(img):  # 计算图像梯度作为先验
    img = rgb2gray(img)
    gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度的幅值和方向
    gradient_magnitude = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    # gradient_direction = np.arctan2(gradient_y, gradient_x)
    return 0.4 * gradient_magnitude


def load_aligned(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 创建SIFT对象
    sift = cv2.SIFT_create()
    # 在两幅图像上检测关键点和计算描述符
    try:
        keypoints1, descriptors1 = sift.detectAndCompute(img1_gray, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2_gray, None)
        # 使用FLANN匹配器进行特征点匹配
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        # 选择最佳匹配的特征点
        good_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)

        # 获取匹配点的坐标
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 使用RANSAC算法估计单应性矩阵
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 对第一幅图像进行对齐
        aligned_image = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))
    except cv2.error as e:
        if "The input arrays should have at least 4 corresponding point sets to calculate Homography" in str(e):
            aligned_image = img1
        else:
            aligned_image = img1
    return aligned_image


def load_item(img_path, re_path, mask_path):
    # load image
    img = cv2.imread(img_path)
    re = cv2.imread(re_path)
    # load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    print(mask.shape)
    mask = (mask > 127).astype(np.uint8) * 255  # threshold due to interpolation
    # load grad
    grad = load_grad(img)
    re_grad = load_grad(re)
    # load aligned
    mask1 = cv2.bitwise_not(mask)
    masked_img = cv2.bitwise_and(img, img, mask=mask1)
    aligned_img = load_aligned(re, masked_img)
    batch = dict()
    batch['re_img'] = F.to_tensor(re)
    batch['aligned_img'] = F.to_tensor(aligned_img)
    batch['re_grad'] = F.to_tensor(re_grad)
    batch['grad'] = F.to_tensor(grad)
    batch['image'] = F.to_tensor(img)
    batch['mask'] = F.to_tensor(mask)
    return batch


def test(img_path, re_path, mask_path):
    items = load_item(img_path, re_path, mask_path)
    img = items['image'].to('cuda:0')
    mask = items['mask'].to('cuda:0')
    grad = items['grad'].to('cuda:0')
    re_grad = items['re_grad'].to('cuda:0')
    aligned_img = items['aligned_img'].to('cuda:0')
    masked_img = img * (1 - mask)
    masked_grad = grad * (1 - mask)
    masked_grad = masked_grad.unsqueeze(0)
    re_grad = re_grad.unsqueeze(0)
    mask = mask.unsqueeze(0)
    masked_img = masked_img.unsqueeze(0)
    aligned_img = aligned_img.unsqueeze(0)
    AFEM_inp = torch.cat([masked_grad, re_grad, mask], dim=1)  # just grad
    FIM_inp = torch.cat([masked_img, mask, aligned_img], dim=1)
    output = model(FIM_inp.to(torch.float32), AFEM_inp.to(torch.float32))
    output = mask * output + (1 - mask) * img
    output = output.squeeze(0)
    output = output * 255
    output = output.permute(1, 2, 0).int().cpu().numpy()
    masked_img = masked_img.squeeze(0)
    masked_img = masked_img * 255
    masked_img = masked_img.permute(1, 2, 0).int().cpu().numpy()
    print('saving image')
    cv2.imwrite('./output.jpg', output)
    cv2.imwrite('./masked_img.jpg', masked_img)


if __name__ == "__main__":
    img_path = './Reimage3k/test/test_img/38.jpg'
    mask_path = './Reimage3k/test/test_mask/38.png'
    re_path = './Reimage3k/test/test_re/38.jpg'
    model = base_model().cuda(0)
    model.load_state_dict(torch.load('./ckpt/Reimage3K/InpaintingModel_best_gen.pth', map_location='cuda')['generator'])

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    num_parameters = count_parameters(model)
    print(f"Number of parameters in the model: {num_parameters}")
    model.eval()
    print(model)
    test(img_path, re_path, mask_path)