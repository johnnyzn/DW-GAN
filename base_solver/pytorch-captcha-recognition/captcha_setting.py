# -*- coding: UTF-8 -*-
import os
# 验证码中的字符
# string.digits + string.ascii_uppercase
NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
LOWLETTER = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n', 'o','p','q','r','s','t','u','v','w','x','y','z']
#ALPHABET = []
ALL_CHAR_SET = NUMBER + LOWLETTER#+ ALPHABET#
ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
MAX_CAPTCHA = 8

# 图像大小
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160

# TRAIN_DATASET_PATH = '/home/ning_a/Desktop/CAPTCHA/dark_web_captcha/rescator_data/train'
# TRAIN_DATASET_PATH = '/home/ning_a/Desktop/CAPTCHA/dataset_darkweb_bias/ebay/train'
# TRAIN_DATASET_PATH = '/home/ning_a/Desktop/CAPTCHA/dataset_darkweb_bias/ebay/train_500'
# TRAIN_DATASET_PATH_2 = '/home/ning_a/Desktop/CAPTCHA/dataset_darkweb_bias/yellow_data'
# TRAIN_DATASET_PATH = '/home/ning_a/Desktop/CAPTCHA/dark_web_captcha/yellow_data_train'
TRAIN_DATASET_PATH = 'D:/LAB/CAPTCHA/dataset_2.0/len_8_train/'
# TEST_DATASET_PATH = '/home/ning_a/Desktop/CAPTCHA/dataset_darkweb_bias/ebay/test_500'
# TEST_DATASET_PATH = '/home/ning_a/Desktop/CAPTCHA/dataset_darkweb_bias/ebay/test'
# TEST_DATASET_PATH = '/home/ning_a/Desktop/CAPTCHA/dark_web_captcha/rescator_data/test'
TEST_DATASET_PATH = 'D:/LAB/CAPTCHA/dataset_2.0/len_8_test/'
# TEST_DATASET_PATH = '/home/ning_a/Desktop/CAPTCHA/dark_web_captcha/rescator_me_test'
#TRAIN_DATASET_PATH = 'dataset' + os.path.sep + 'train_3_10w'
#TEST_DATASET_PATH = 'dataset' + os.path.sep + 'test_3'
#TEST_DATASET_PATH = '/home/ning_a/Desktop/CAPTCHA/test_generator/digit_3/test'
PREDICT_DATASET_PATH = 'dataset' + os.path.sep + 'predict'