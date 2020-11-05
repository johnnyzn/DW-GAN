# -*- coding: UTF-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
import captcha_setting
import my_dataset
from captcha_cnn_model import CNN, Generator
#import one_hot_encoding
from torchvision.utils import save_image
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn = CNN()
    cnn = CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load('model_final_mix_2.pkl'))
    cnn.to(device)
    # generator = Generator()
    # generator.load_state_dict(torch.load('7800.pkl'))
    # generator.to(device)
    # generator.eval()
    print("load cnn net.")

    test_dataloader = my_dataset.get_test_data_loader()
    # test_dataloader = my_dataset.get_predict_data_loader()
    correct = 0
    total = 0
    for i, (images1, labels) in enumerate(test_dataloader):
        image = images1
        #print(image)
        image = Variable(image)
        image, labels =  image.to(device), labels.to(device)
        # vimage = generator(image)

        predict_label = cnn(image)
        labels = labels.cpu()
        predict_label = predict_label.cpu()
        _, predicted = torch.max(predict_label, 1)
        #print(captcha_setting.ALL_CHAR_SET[predicted], '>>>>>>>>>>>>', one_hot_encoding.decode(labels.numpy()[0]))
        # c0 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        # c1 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        # c2 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        # #c3 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        #predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
        # _, predicted = torch.max(predict_label, 1)
        # predict_label = '%s%s%s' % (c0, c1, c2)
        # true_label = one_hot_encoding.decode(labels.numpy()[0])
        total += labels.size(0)
        print(predicted,'>>>>>>>>',labels)
        #save_image(vimage,'temp_result/'+str(i)+'.png')
        #print("labels: ",labels, '||| predict_label: ', predicted)
        if(predicted == labels):
            correct += 1
        if(total%2000==0):
            print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
    print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))

if __name__ == '__main__':
    main()


