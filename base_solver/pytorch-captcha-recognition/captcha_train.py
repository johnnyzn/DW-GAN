# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import my_dataset
from captcha_cnn_model import CNN, Generator
from datetime import datetime
import pickle
import numpy as np
import captcha_setting
import one_hot_encoding
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# Hyper Parameters
num_epochs = 170
batch_size = 100
learning_rate = 0.0002
model_name = 'ex_2_len8.pkl'
model_2_name = 'bias_rescator_me_tunning.pkl'
def main():
    device = torch.device("cuda" )#if torch.cuda.is_available() else "cpu")
    # Train the Model
    print('loading data')
    train_dataloader = my_dataset.get_train_data_loader()
    test_dataloader = my_dataset.get_test_data_loader()
    print('start training')
    loss_list = []
    accuracy_list = []

    for ttest_num in range(0,5):
        cnn = nn.DataParallel(CNN())
        # cnn.load_state_dict(torch.load('./'+model_name))
        # cnn.to(device)
        cnn.cuda()
        # generator = Generator()
        # generator.load_state_dict(torch.load('7800.pkl'))
        # generator.to(device)
        # generator.eval()
        cnn.train()
        print('init net')
        criterion = nn.MultiLabelSoftMarginLoss()
        optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

        print('>>>>>>>>>>>>Round:',ttest_num)
        accuracy = 0
        for epoch in range(num_epochs):
            loss_total = 0
            start = datetime.now()
            for i, (images1, labels, _) in enumerate(train_dataloader):
                #print(i)
                images1 = Variable(images1)
                labels = Variable(labels.float())
                images1, labels =  images1.to(device), labels.to(device)
                # images1 = generator(images1)

                predict_labels = cnn(images1)
                # print(predict_labels.type)
                # print(labels.type)
                loss = criterion(predict_labels, labels)
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step()
                loss_total += loss.item()
                # if (i+1) % 5000 == 0:
                #     print("epoch:", epoch, "step:", i, "loss:", loss.item())
                # if (i+1) % 25000 == 0:
                #     print("save model")
            correct = 0
            total = 0
            # print(len(test_dataloader))
            for i, (images1, labels, _) in enumerate(test_dataloader):
                # print('///////////////////////////////')
                # try:
                image = images1
                image = Variable(image)
                image, labels =  image.to(device), labels.to(device)
                # image = generator(image)
                predict_label = cnn(image)
                labels = labels.cpu()
                predict_label = predict_label.cpu()
                c0 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
                c1 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
                c2 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
                c3 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
                c4 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 4 * captcha_setting.ALL_CHAR_SET_LEN:5 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
                c5 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 5 * captcha_setting.ALL_CHAR_SET_LEN:6 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
                c6 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 6 * captcha_setting.ALL_CHAR_SET_LEN:7 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
                c7 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 7 * captcha_setting.ALL_CHAR_SET_LEN:8 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
                
                #c3 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
                predict_label = '%s%s%s%s%s%s%s%s' % (c0, c1, c2, c3, c4, c5, c6, c7)
                # predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
                true_label = one_hot_encoding.decode(labels.numpy()[0])
                total += 1#labels.size(0)
                #save_image(vimage,'temp_result/'+str(i)+'.png')
                # print(predict_label.upper(),'>>>>>',true_label)
                if(predict_label.upper() == true_label.upper()):
                    correct += 1
            # try:
            print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
            # except:
            # 	pass
            loss_list.append(loss_total)
            with open("loss_record_6.txt", "wb") as fp:   #Pickling
                pickle.dump(loss_list, fp)
            stop = datetime.now()
            # try:
            if(correct / total>accuracy ):
                accuracy = correct / total
                # torch.save(cnn.state_dict(), "./model_lake/"+model_name.replace('.','_'+str(ttest_num)+'.'))   #current is model.pickle
                torch.save(cnn.state_dict(), "./model_lake/"+model_name)   #current is model.pickle
                print('saved!!!!!!!!!!!!!!!!!!!!!!!')
            # except:
	           #  pass
            print("epoch:", epoch, "step:", i, " time:<",stop - start,"> loss:", loss_total)
        accuracy_list.append(accuracy)
        print(sum(accuracy_list)/len(accuracy_list))
    # torch.save(cnn.state_dict(), "./"+model_name)   #current is model.pkl
    # print("save last model")
    print(accuracy_list)
if __name__ == '__main__':
    main()


