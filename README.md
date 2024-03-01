# Image-Inpainting-Based-on-Reference-Image-and-Auxiliary-Feature-Fusion-RDCIN-
Official code for Image Inpainting Based on Reference Image and Auxiliary Feature Fusion.
# pipeline
[![dab040aa8e8076a400f9d46a29357b06.png](https://s1.imagehub.cc/images/2024/01/23/dab040aa8e8076a400f9d46a29357b06.png)](https://www.imagehub.cc/image/1aDYiA)
# Environment
## preparing the environment

  ```
  conda create -n train_env python=3.6
  conda activate train_env
  pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
  pip install -r requirement.txt
  ```
## DistributedDataParallel（DDP）
 If you want to achieve distributed training through multiple cards, you can use Distributed Data Parallel (DDP), If you only have one GPU, you can ignore it

  ```
  git clone https://github.com/NVIDIA/apex
  cd apex
  pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" ./
  ```
If you need to train the model, please download the pretrained models for perceptual loss, provided by LaMa:
  ```
  mkdir -p ade20k/ade20k-resnet50dilated-ppm_deepsup/
  wget -P ade20k/ade20k-resnet50dilated-ppm_deepsup/ http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth
  ```
## Pre-trained model
You can download pre trained models to test the data. [Pretrained-model](https://pan.baidu.com/s/1pxnusro0OAIkbWJDi-mo8Q?pwd=B1X1) (Password: B1X1)
# Test image
  There are two situations:
  ## Test Reimage3k/Dped10k
  you can download Reimage3k and DPED10k.
  
  Download Reimage3k datasets from [Reimag3k](https://pan.baidu.com/s/1mlkVfbyQi3_Fv7GTuilZ0w) (Password: A1B3).
  
  Please download DPED10K dataset from [DPED10K](https://pan.baidu.com/share/init?surl=8mwRhUdKsKaL6J-08mdlLQ) (Password: roqs). Create a folder and unzip the dataset into it, then edit the pathes of the folder in options/base_options.py
  ## Test your own image
  There are two options:
  1. If you want to test single image, you can run test_ single_ img.py, by modifying the input image and pre-trained model.
  ```
  python test_single_img.py
  ```
  2. If you want to test multiple images, you can run the model_ Test.py. Please use create.py to generate the txt file for the image folder path, and then modify the config_ Test. yaml in the list is sufficient.
  ```
  python model_test.py --nodes 1 --gpus 1 --GPU_ids '0' --config_file ./config_list/config_test_Reimage.yml
  ```
# Train your own model
  If you need to train the model, please download the pretrained models for perceptual loss, provided by LaMa:
  ```bash
  mkdir -p ade20k/ade20k-resnet50dilated-ppm_deepsup/
  wget -P ade20k/ade20k-resnet50dilated-ppm_deepsup/ http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth
  ```
  
  Prepare the dataset and prepare the. txt file (generated through create. py). Then modify the train. yaml file in the config_list.

  You can run:
  ```
  python train.py --nodes 1 --gpus 1 --GPU_ids '0' --path ./ckpt/xxx/ --config_file ./config_list/config_train_Reimage.yml --DDP
  ```
  or:
  ```
  python train.py --nodes 1 --gpus 1 --GPU_ids '0' --path ./ckpt/xxx/ --config_file ./config_list/config_train_Reimage.yml
  ```
# Acknowledgments
RDCIN is bulit upon the [ZITS](https://github.com/DQiaole/ZITS_inpainting?tab=readme-ov-file) and inspired by [RGTSI](https://github.com/Cameltr/RGTSI). We appreciate the authors' excellent work!

