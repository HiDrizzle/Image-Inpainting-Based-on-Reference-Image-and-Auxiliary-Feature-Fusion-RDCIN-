import glob
import os
import pickle
import random
import cv2
import numpy as np
import skimage.draw
import torch
import torchvision.transforms.functional as F
from skimage.color import rgb2gray
from skimage.feature import canny
from torch.utils.data import DataLoader


def to_int(x):
    return tuple(map(int, x))


class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, flist, input_size, re_flist, mask_rates=None, mask_path=None, augment=True, training=True,
                 test_mask_path=None):
        super(ImgDataset, self).__init__()
        self.augment = augment
        self.training = training
        self.re_dataset = re_flist
        self.data = []
        self.re_data = []
        f = open(flist, 'r')
        for i in f.readlines():
            i = i.strip()
            self.data.append(i)
        f.close()

        f = open(re_flist, 'r')
        for i in f.readlines():
            i = i.strip()
            self.re_data.append(i)
        f.close()

        if training:
            self.irregular_mask_list = []
            with open(mask_path[0]) as f:
                for line in f:
                    self.irregular_mask_list.append(line.strip())
            self.irregular_mask_list = sorted(self.irregular_mask_list, key=lambda x: x.split('/')[-1])
            self.segment_mask_list = []
            with open(mask_path[1]) as f:
                for line in f:
                    self.segment_mask_list.append(line.strip())
            self.segment_mask_list = sorted(self.segment_mask_list, key=lambda x: x.split('/')[-1])
        else:
            self.mask_list = glob.glob(test_mask_path + '/*')
            self.mask_list = sorted(self.mask_list, key=lambda x: x.split('/')[-1])
        self.input_size = input_size
        self.mask_rates = mask_rates

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_grad(self, img):  # 计算图像梯度作为先验
        img = rgb2gray(img)
        gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

        # 计算梯度的幅值和方向
        gradient_magnitude = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
        # gradient_direction = np.arctan2(gradient_y, gradient_x)
        return 0.4 * gradient_magnitude
    def load_edge(self, img):
        return canny(img, sigma=2, mask=None).astype(np.float)
    
    def load_aligned(self, img1, img2):
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

    def load_aligned_grad(self, img1, img2):

        # 创建SIFT对象
        sift = cv2.SIFT_create()
        # 在两幅图像上检测关键点和计算描述符
        try:
            keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
            keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
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
    
    def load_item(self, index):
        size = self.input_size
        # load image
        img = cv2.imread(self.data[index])
        re = cv2.imread(self.re_data[index])
        while img is None:
            print('Bad image {}...'.format(self.data[index]))
            idx = random.randint(0, len(self.data) - 1)
            img = cv2.imread(self.data[idx])
        while re is None:
            print('Bad image {}...'.format(self.re_data[index]))
            idx = random.randint(0, len(self.re_data) - 1)
            img = cv2.imread(self.re_data[idx])
        img = img[:, :, ::-1]
        re = re[:, :, ::-1]
        # resize/crop if needed
        img = self.resize(img, size, size)
        re = self.resize(re, size, size)
        # load mask
        mask = self.load_mask(img, index)
        # load edge
        img_gray = rgb2gray(img)
        re_gray = rgb2gray(re)
        edge = self.load_edge(img_gray)
        re_edge = self.load_edge(re_gray)
        # load grad
        grad = self.load_grad(img)
        re_grad = self.load_grad(re)
        # load aligned
        mask1 = cv2.bitwise_not(mask)  
        masked_img = cv2.bitwise_and(img, img, mask=mask1)
        masked_grad = cv2.bitwise_and(grad, grad, mask=mask1)
        aligned_img = self.load_aligned(re, masked_img)
        aligned_grad = self.load_aligned_grad(re_grad, masked_grad)

        # augment data
        if self.augment and random.random() > 0.5 and self.training:
            img = img[:, ::-1, ...].copy()
            re_grad = re_grad[:, ::-1, ...].copy()
            grad = grad[:, ::-1, ...].copy()
            re = re[:, ::-1, ...].copy()
            aligned_img = aligned_img[:, ::-1, ...].copy()
        if self.augment and random.random() > 0.5 and self.training:
            mask = mask[:, ::-1, ...].copy()
        if self.augment and random.random() > 0.5 and self.training:
            mask = mask[::-1, :, ...].copy()
        batch = dict()
        batch['aligned_img'] = self.to_tensor(aligned_img)
        batch['aligned_grad'] = self.to_tensor(aligned_grad)
        batch['re_img'] = self.to_tensor(re)
        batch['re_grad'] = self.to_tensor(re_grad)
        batch['grad'] = self.to_tensor(grad)
        batch['re_edge'] = self.to_tensor(re_edge)
        batch['edge'] = self.to_tensor(edge)
        # batch['aligned'] = self.to_tensor(aligned_img)
        batch['image'] = self.to_tensor(img)
        batch['mask'] = self.to_tensor(mask)
        batch['name'] = self.load_name(index)
        return batch

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        if self.training is False:
            mask = cv2.imread(self.mask_list[index], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255
            return mask
        else:  # train mode: 40% mask with random brush, 40% mask with coco mask, 20% with additions
            rdv = random.random()
            if rdv < self.mask_rates[0]:
                mask_index = random.randint(0, len(self.irregular_mask_list) - 1)
                mask = cv2.imread(self.irregular_mask_list[mask_index],
                                  cv2.IMREAD_GRAYSCALE)
            elif rdv < self.mask_rates[1]:
                mask_index = random.randint(0, len(self.segment_mask_list) - 1)
                mask = cv2.imread(self.segment_mask_list[mask_index],
                                  cv2.IMREAD_GRAYSCALE)
            else:
                mask_index1 = random.randint(0, len(self.segment_mask_list) - 1)
                mask_index2 = random.randint(0, len(self.irregular_mask_list) - 1)
                mask1 = cv2.imread(self.segment_mask_list[mask_index1],
                                   cv2.IMREAD_GRAYSCALE).astype(np.float32)
                mask2 = cv2.imread(self.irregular_mask_list[mask_index2],
                                   cv2.IMREAD_GRAYSCALE).astype(np.float32)
                mask = np.clip(mask1 + mask2, 0, 255).astype(np.uint8)

            if mask.shape[0] != imgh or mask.shape[1] != imgw:
                mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255  # threshold due to interpolation
            return mask

    def to_tensor(self, img):
        # img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, center_crop=False):
        imgh, imgw = img.shape[0:2]

        if center_crop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        if imgh > height and imgw > width:
            inter = cv2.INTER_AREA
        else:
            inter = cv2.INTER_LINEAR
        img = cv2.resize(img, (height, width), interpolation=inter)
        return img

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

