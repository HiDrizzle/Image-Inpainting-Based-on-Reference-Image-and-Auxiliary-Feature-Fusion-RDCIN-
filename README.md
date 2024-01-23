# Image-Inpainting-Based-on-Reference-Image-and-Auxiliary-Feature-Fusion-RDCIN-
Official code for Image Inpainting Based on Reference Image and Auxiliary Feature Fusion.
# pipeline
[![dab040aa8e8076a400f9d46a29357b06.png](https://s1.imagehub.cc/images/2024/01/23/dab040aa8e8076a400f9d46a29357b06.png)](https://www.imagehub.cc/image/1aDYiA)
# Test image
  There are two situations:
  ## test Reimage3k/Dped10k
  you can download Reimage3k and DPED10k.
  
  Download Reimage3k datasets in [Reimag3k]https://pan.baidu.com/s/1mlkVfbyQi3_Fv7GTuilZ0w code is A1B3.
  
  Please download DPED10K dataset from [Baidu Netdisk]https://pan.baidu.com/share/init?surl=8mwRhUdKsKaL6J-08mdlLQ (Password: roqs). Create a folder and unzip the dataset into it, then edit the pathes of the folder in options/base_options.py
  ## test your own image
  There are two options:
  1. If you want to test single image, you can run test_ single_ img.py, by modifying the input image and pre-trained model.
  2. If you want to test multiple images, you can run the model_ Test.py. Please use create.py to generate the txt file for the image folder path, and then modify the config_ Test. yaml in the list is sufficient.
# Train your own model
  Prepare the dataset and prepare the. txt file (generated through create. py). Then modify the train. yaml file in the config_list.
  
  <span stylt="color:#333333">'python train.py --nodes 1 --gpus 1 --GPU_ids '0' --path ./ckpt/xxx/ --config_file ./config_list/config_train_DPED.yml' --DDP</span>
