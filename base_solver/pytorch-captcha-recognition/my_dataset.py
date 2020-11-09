# -*- coding: UTF-8 -*-
import os
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
from PIL import Image
import one_hot_encoding as ohe
import captcha_setting
import numpy as np
import cv2
class mydataset(Dataset):

    def __init__(self, folder, folder_2 = None, transform=None):
        self.train_image_file_paths = [os.path.join(folder, image_file) for image_file in os.listdir(folder)]
        if(folder_2 is not None):
            self.train_image_file_paths = self.train_image_file_paths + [os.path.join(folder_2, image_file) for image_file in os.listdir(folder_2)]
        print(len(self.train_image_file_paths))
        self.transform = transform

    def __len__(self):
        return len(self.train_image_file_paths)

    def __getitem__(self, idx):
        image_root = self.train_image_file_paths[idx]
        image_name = image_root.split('/')[-1]
        image = Image.open(image_root)
        #print(image)
        fix_size = (160, 60)
        image = image.resize(fix_size)
        # print(image_name)
        if self.transform is not None:
            image = self.transform(image)
        # print(image_name)
        if('_' in image_name):
            label = ohe.encode(image_name.split('_')[0].upper())
        else:
            label = ohe.encode(image_name.split('.')[0].upper())
        return image, label, image_name
def gaussian_blur(img):
    image = np.array(img)
    image_blur = cv2.GaussianBlur(image,(5,5),3)
    new_image = image_blur
    return new_image

transform = transforms.Compose([
    # transforms.ColorJitter(),
    transforms.Grayscale(),
    # transforms.Lambda(gaussian_blur),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.9], std=[0.4]),
    
])
def get_train_data_loader(s=True,d=200):
    print('data path: ', captcha_setting.TRAIN_DATASET_PATH)
    # dataset = mydataset(captcha_setting.TRAIN_DATASET_PATH, captcha_setting.TRAIN_DATASET_PATH_2, transform=transform)
    dataset = mydataset(captcha_setting.TRAIN_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=512, shuffle=s)

def get_test_train_data_loader(s=True,d=256):
    dataset = mydataset(captcha_setting.TRAIN_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=d, shuffle=s)

def get_test_data_loader(s=False,d=1):
    print(captcha_setting.TEST_DATASET_PATH)
    dataset = mydataset(captcha_setting.TEST_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=d, shuffle=s)

def get_predict_data_loader(s=True,d=1):
    dataset = mydataset(captcha_setting.PREDICT_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=d, shuffle=s)