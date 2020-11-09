# -*- coding: UTF-8 -*-
import os
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
from PIL import Image
import one_hot_encoding as ohe
import captcha_setting

class mydataset(Dataset):

    def __init__(self, folder, transform=None):
        #self.train_image_file_paths = [os.path.join(folder, image_file) for image_file in os.listdir(folder)]
        self.train_image_file_paths = []
        for subfolder in os.listdir(folder):
            self.train_image_file_paths += [os.path.join(os.path.join(folder, subfolder), image_file) for image_file in os.listdir(os.path.join(folder, subfolder))]
        self.transform = transform
        #print(self.train_image_file_paths)

    def __len__(self):
        return len(self.train_image_file_paths)

    def __getitem__(self, idx):
        image_root = self.train_image_file_paths[idx]
        image_name = image_root.split(os.path.sep)[-1]
        image = Image.open(image_root)
        if self.transform is not None:
            image = self.transform(image)
        label = captcha_setting.ALL_CHAR_SET.index(image_root.split(os.path.sep)[-2]).upper() # 为了方便，在生成图片的时候，图片文件的命名格式 "4个数字或者数字_时间戳.PNG", 4个字母或者即是图片的验证码的值，字母大写,同时对该值做 one-hot 处理
        #label = image_root.split(os.path.sep)[-2]
        return image, label
class mydataset_full_char(Dataset):

    def __init__(self, folder, transform=None):
        #print('>>>>',folder)
        #self.train_image_file_paths = [os.path.join(folder, image_file) for image_file in os.listdir(folder)]
        self.train_image_file_paths = []
        for subfolder in os.listdir(folder):
            self.train_image_file_paths += [os.path.join(os.path.join(folder, subfolder), image_file) for image_file in os.listdir(os.path.join(folder, subfolder))]
        self.transform = transform
        #print(self.train_image_file_paths)

    def __len__(self):
        return len(self.train_image_file_paths)

    def __getitem__(self, idx):
        image_root = self.train_image_file_paths[idx]
        #print(image_root)
        image_name = image_root.split(os.path.sep)[-1].split('_')[0]
        image = Image.open(image_root)
        if self.transform is not None:
            image = self.transform(image)
        # label = captcha_setting.ALL_CHAR_SET.index(image_name) # 为了方便，在生成图片的时候，图片文件的命名格式 "4个数字或者数字_时间戳.PNG", 4个字母或者即是图片的验证码的值，字母大写,同时对该值做 one-hot 处理
        label = captcha_setting.ALL_CHAR_SET.index(image_root.split(os.path.sep)[-2].split('/')[-1].lower())
        # print(label)
        return image, label
transform = transforms.Compose([
    # transforms.ColorJitter(),
    transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def get_train_data_loader():
    print('data path: ', captcha_setting.TRAIN_DATASET_PATH)
    dataset = mydataset_full_char(captcha_setting.TRAIN_DATASET_PATH, transform=transform)
    #dataset = mydataset(captcha_setting.TRAIN_DATASET_PATH, transform=transform)

    #mydataset(captcha_setting.TRAIN_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=512, shuffle=True)

def get_test_train_data_loader():
    dataset = mydataset(captcha_setting.TRAIN_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=True)

def get_test_data_loader():
    dataset = mydataset_full_char(captcha_setting.TEST_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=True)

def get_predict_data_loader():
    dataset = mydataset(captcha_setting.PREDICT_DATASET_PATH_TEMP, transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=True)