# -*- coding: UTF-8 -*-
import os
# 验证码中的字符
# string.digits + string.ascii_uppercase
NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
UPLETTER = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
LOWLETTER = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n', 'o','p','q','r','s','t','u','v','w','x','y','z']
#ALPHABET = []
ALL_CHAR_SET = NUMBER + LOWLETTER
ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
MAX_CAPTCHA = 1

# 图像大小
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 30

#TRAIN_DATASET_PATH = 'dataset' + os.path.sep + 'train'
TRAIN_DATASET_PATH = '/home/ning_a/Desktop/CAPTCHA/dataset_1.2' + os.path.sep + 'train'
TEST_DATASET_PATH = '/home/ning_a/Desktop/CAPTCHA/dataset_1.2' + os.path.sep + 'test'
PREDICT_DATASET_PATH = '/home/ning_a/Desktop/CAPTCHA/dataset_1.1' + os.path.sep + 'predict'
PREDICT_DATASET_PATH_TEMP = '/home/ning_a/Desktop/CAPTCHA/Solving_CAPTCHA_GAN_CNN/process_tool/cut_image'

#print(ALL_CHAR_SET.index("a"))