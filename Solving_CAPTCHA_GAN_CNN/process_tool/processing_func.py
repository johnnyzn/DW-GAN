import sys, os
sys.path.insert(1, '/home/ning_a/Desktop/CAPTCHA/base_solver/base_solver_char')
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import torchvision.transforms as transforms
import captcha_setting
import my_dataset
from captcha_cnn_model import CNN, Generator
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image
import copy
import operator
import processing_func

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cnn = CNN()
cnn.eval()
cnn.load_state_dict(torch.load('model_final_mix_2.pkl'))
cnn.to(device)
#for eachimg in filter_containor:
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Linear((captcha_setting.IMAGE_WIDTH//8)*(captcha_setting.IMAGE_HEIGHT//8)*64, 1024),
            nn.Dropout(0.1),  # drop 50% of the neuron
            nn.ReLU())
        self.rfc = nn.Sequential(
            nn.Linear(1024, 256),#captcha_setting.MAX_CAPTCHA*captcha_setting.ALL_CHAR_SET_LEN),
            nn.ReLU()
        )
        self.rfc2 = nn.Sequential(
            nn.Linear(256, 36),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        #print(out.shape)
        out = self.rfc(out)
        out = self.rfc2(out)
        #out = out.view(out.size(0), -1)
        #print(out.shape)
        return out
def gaussian_blur(img):
    image = np.array(img)
    image_blur = cv.GaussianBlur(image,(5,5),0)
    new_image = image_blur
    return new_image
class testdataset(Dataset):

    def __init__(self, folder, transform=None):
        self.train_image_file_paths = [os.path.join(folder, image_file) for image_file in os.listdir(folder)]
        self.transform = transform

    def __len__(self):
        return len(self.train_image_file_paths)

    def __getitem__(self, idx):
        image_root = self.train_image_file_paths[idx]
        image_name = image_root.split(os.path.sep)[-1]
        image = Image.open(image_root)
        image = image.resize((160,60), Image.ANTIALIAS)
        label = image_name
        #label = ohe.encode(image_name.split('_')[0]) # 为了方便，在生成图片的时候，图片文件的命名格式 "4个数字或者数字_时间戳.PNG", 4个字母或者即是图片的验证码的值，字母大写,同时对该值做 one-hot 处理
        if self.transform is not None:
            image = self.transform(image)
            #label = self.transform(label)
        #label = ohe.encode(image_name.split('_')[0])
        return image, label
transform_1 = transforms.Compose([
    # transforms.ColorJitter(),
    transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def get_loader():
    img_path = "/home/ning_a/Desktop/CAPTCHA/dark_web_captcha/yellow_data/"
    dataset = testdataset(img_path, transform=transform_1)
    return DataLoader(dataset, batch_size=1, shuffle=False)
def img_norm(pil_img):
    img=np.array(pil_img)  
#     print(img.shape)
    img=cv.cvtColor(img, cv.COLOR_RGB2BGR) 
    # img = 255-img
    # plt.imshow(img)
    # plt.show()
    threshold = 5 
    img = np.array(img)
    n_img = np.zeros((img.shape[0],img.shape[1]))
    img_aft = cv.normalize(img, n_img, 0,255,cv.NORM_MINMAX)
    # plt.imshow(img_aft)
    # plt.show()
    gray = cv.cvtColor(img_aft,cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    element_4 = cv.getStructuringElement(cv.MORPH_RECT, (2,2))
    thresh = cv.dilate(thresh, element_4, iterations =3)
    thresh = cv.erode(thresh, element_4, iterations = 2)
    thresh = thresh
    # plt.imshow(thresh)
    # plt.show()
    return thresh
def calculate_corner(thresh,img_t, nrootdir="cut_image/"):
    show_img = cv.imread('temp.jpg')
    im2,contours,hierarchy = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
#     print(contours[0])
    new_contours = []
    cur_contours = []
    filter_containor = []
    for i in contours:
        #print(i)
        x, y, w, h = cv.boundingRect(i)   
        cur_contours.append([x, y, w, h])
    contours = sorted(cur_contours, key=operator.itemgetter(0))
    color = [255, 255, 255]
    for i in range(0,len(contours)):  
        x = contours[i][0]
        y = contours[i][1]
        w = contours[i][2] 
        h = contours[i][3]
        newimage=img_t[y:y+h,x:x+w] # 先用y确定高，再用x确定宽
        nrootdir=("cut_image/")
        if (h<8) or (h*w<100) or (h*w >4000):
            continue
        # print(contours[i])
        top, bottom, left, right = [1]*4

        newimage = cv.copyMakeBorder(newimage, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)
        cv.rectangle(show_img, (x,y), (x+w+3,y+h), (153,153,0), 1)
        new_contours.append(contours[i])
#     t0 = [i[0] for i in new_contours]
#     t1 = [i[1] for i in new_contours]
#     t2 = [i[2] for i in new_contours]
#     t3 = [i[3] for i in new_contours]
    # x_max = max([new_contours[i][0] + new_contours[i][2] for i in range(len(new_contours))] )+5
    # x_min = min(t0)
    # y_max = max([new_contours[i][1] + new_contours[i][3] for i in range(len(new_contours))] )
    # y_min = min(t1)
    x_max = 150
    x_min = 10
    y_max = 50
    y_min = 10
    width = (x_max-x_min)//6
    for i in range(0,6):
#         cv.rectangle(img_t, (x_min+i*width,y_min), (x_min+(i+1)*width,y_max), (153,153,0), 1)
        newimage=thresh[y_min:y_max,x_min+i*width-3:x_min+(i+1)*width+3]
        top, bottom, left, right = [1]*4
        newimage = cv.copyMakeBorder(newimage, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)
        newimage = cv.resize(newimage,(30, 60), interpolation = cv.INTER_CUBIC)
        cv.imwrite( "temp.jpg",newimage)
        filter_containor.append(Image.open("temp.jpg"))
    # plt.imshow(show_img)
    # plt.show()
    return filter_containor
def run_cnn_solver(filter_containor):
    transform = transforms.Compose([
        # transforms.ColorJitter(),
        transforms.Grayscale(),
        transforms.Lambda(gaussian_blur),
        transforms.ToTensor()
    ])
    label_predicted = ""
    for eachimg in filter_containor:
        image = transform(eachimg).unsqueeze(0)
#         plt.imshow(eachimg)
#         plt.show()
        # print(image.shape)

        image = torch.tensor(image, device=device).float()
        image = Variable(image).to(device)
        predict_label = cnn(image)
        predict_label = predict_label.cpu()
        _, predicted = torch.max(predict_label, 1)
        # print(captcha_setting.ALL_CHAR_SET[predicted])
        label_predicted += captcha_setting.ALL_CHAR_SET[predicted]
    # print(label_predicted[0:6])
    return label_predicted[0:6]