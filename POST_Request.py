import requests
import json
import numpy as np
import cv2

# 定义服务器端的地址
predict_url = 'http://127.0.0.1:5000/predict'
upload_url = 'http://127.0.0.1:5000/upload'


def upload():
    image1_path = './Reimage3k/test/test_img/20.jpg'
    image1 = open(image1_path, 'rb')
    files = {'img': ('img.jpg', image1)}
    response = requests.post(upload_url, files=files)

    return response.content


def predict():
    # 分别加载三张图像
    image1_path = './Reimage3k/test/test_img/20.jpg'  # 替换为你的图像路径
    image2_path = './Reimage3k/test/test_re/20.jpg'  # 替换为你的图像路径
    image3_path = './Reimage3k/test/test_mask/20.png'  # 替换为你的图像路径

    image1 = open(image1_path, 'rb')
    image2 = open(image2_path, 'rb')
    image3 = open(image3_path, 'rb')

    # 构建图像字典
    files = {
        'img': ('img.jpg', image1),
        're': ('re.jpg', image2),
        'mask': ('mask.png', image3)
    }

    # 发送POST请求
    response = requests.post(predict_url, files=files)

    # 解析服务器端的响应
    if response.status_code == 200:
        result = json.loads(response.content)
        if result['success']:
            output_data = result['output']
            output_data = np.array(output_data, dtype=np.uint8)
            print(output_data)
            # 在这里处理模型的输出数据
            cv2.imwrite('./client_img/output.jpg', output_data)
            print("Model output shape:", output_data.shape)
        else:
            print("Server error:", result['error'])
    else:
        print("Server error. Status code:", response.status_code)


if __name__ == "__main__":
    upload()
    predict()
