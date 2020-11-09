# -*- coding: UTF-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
import captcha_setting
import my_dataset
from captcha_cnn_model import CNN, Generator
import one_hot_encoding
from torchvision.utils import save_image
def main():
    # device = torch.device("cuda")
    device = torch.device("cpu")
    cnn = CNN()
    cnn.eval()
    
    cnn.load_state_dict(torch.load('synthesizer_4.pkl',map_location={'cuda:0': 'cpu'}))
    cnn.to(device)
    generator = Generator()
    generator.load_state_dict(torch.load('7800.pkl', map_location={'cuda:0': 'cpu'}))
    generator.to(device)
    generator.eval()
    print("load cnn net.")

    #test_dataloader = my_dataset.get_test_train_data_loader()
    test_dataloader = my_dataset.get_test_data_loader()
    correct = 0
    total = 0
    for i, (images1, labels, _) in enumerate(test_dataloader):
        try:
            image = images1
            image = Variable(image)
            image, labels =  image.to(device), labels.to(device)
            # vimage = generator(image)
            predict_label = cnn(image)
            labels = labels.cpu()
            predict_label = predict_label.cpu()
            c0 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
            c1 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
            c2 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
            c3 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
            # c4 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 4 * captcha_setting.ALL_CHAR_SET_LEN:5 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
            # c5 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 5 * captcha_setting.ALL_CHAR_SET_LEN:6 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
            
            #c3 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
            #predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
            predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
            true_label = one_hot_encoding.decode(labels.numpy()[0])
            total += labels.size(0)
            #save_image(vimage,'temp_result/'+str(i)+'.png')
            # print(predict_label.upper(),'>>>>>',true_label)
            if(predict_label.upper() == true_label.upper()):
                correct += 1
            if(total%2000==0):
                print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
        except:
            pass
    print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))

if __name__ == '__main__':
    main()


