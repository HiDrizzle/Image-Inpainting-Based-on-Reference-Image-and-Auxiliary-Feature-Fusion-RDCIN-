import base64
import io
import os
import uuid

import torchvision.transforms.functional as F
from flask import Flask, request, jsonify, render_template
from PIL import Image
from src.models.AFEMandFTR import *
import cv2
import numpy as np
from skimage.color import rgb2gray

app = Flask(__name__)

# 加载你的PyTorch模型
model = None  # 请替换为你自己的模型

ALLOWED_EXTENSIONS = {'png', 'jpg', 'JPG', 'jpeg', 'PNG', 'bmp'}
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


def load_model():
    global model
    # 在这里加载你的模型，例如：
    model = base_model().cuda(0)
    model.load_state_dict(torch.load('./ckpt/Reimage3K/InpaintingModel_best_gen.pth', map_location='cuda')['generator'])
    model.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def load_item(img, re, mask):
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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取客户端发送的三张图像
        img = request.files["file0"].read()
        img = Image.open(io.BytesIO(img))
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        re = request.files["file1"].read()
        re = Image.open(io.BytesIO(re))
        re = cv2.cvtColor(np.array(re), cv2.COLOR_RGB2BGR)

        mask = request.files["file2"].read()
        mask = Image.open(io.BytesIO(mask))
        mask = mask.convert("L")
        # 确保 mask 被解释为单通道灰度图像，而不是3通道图像
        mask = np.array(mask)
        items = load_item(img, re, mask)
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
        # 在模型上进行推断
        with torch.no_grad():
            output = model(FIM_inp.to(torch.float32), AFEM_inp.to(torch.float32))
            output = mask * output + (1 - mask) * img
            output = output.squeeze(0)
            output = output * 255
            output = output.permute(1, 2, 0).int().cpu().numpy()

        # 返回结果

        output_img = {'success': True, 'output': output}
        output_img = output_img['output']
        output_img = np.array(output_img, dtype=np.uint8)
        cv2.imwrite('./server_img/output.jpg', output_img)
        _, img_encoded = cv2.imencode('.jpg', output)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        img_data_url = f'data:image/png;base64,{img_base64}'
        response_data = {'success': True, 'output': img_data_url}
        return jsonify(response_data)

    except Exception as e:
        print(str(e))
        response_data = {'success': False, 'error': str(e)}
        return jsonify(response_data)


@app.route('/', methods=['GET', 'POST'])
def root():
    return render_template("up.html")


if __name__ == '__main__':
    load_model()  # 加载模型
    app.run(host='127.0.0.1', port=5000)
