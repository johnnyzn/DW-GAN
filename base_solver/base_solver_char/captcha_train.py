# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import my_dataset
from captcha_cnn_model import CNN, Generator

# Hyper Parameters
num_epochs = 100
batch_size = 100
learning_rate = 0.0005
model_name = 'ex22_8_tracing.pkl'
# model_2_name = 'bias_rescator_me_tunning.pkl'
def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    accuracy_list = []
    # Train the Model
    print("Loading Data")
    train_dataloader = my_dataset.get_train_data_loader()
    test_dataloader = my_dataset.get_test_data_loader()
    print("Data Ready")
    accuracy = 0
    for ttest_num in range(0,5):
        print('>>>>>>>>>>>>>>>>> ROUND: ',ttest_num)
        cnn = CNN()
        cnn.to(device)
        cnn.train()
        print('init net')
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
        accuracy = 0
        for epoch in range(num_epochs):
            print('....')
            for i, (images1, labels) in enumerate(train_dataloader):
                images1 = Variable(images1)
                labels = Variable(labels)#.view(-1,1))
                images, labels =  images1.to(device), labels.to(device)
                # images = generator(images)
                #labels = torch.tensor(labels, dtype=torch.long, device=device)
                predict_labels = cnn(images)#.view(1,-1)[0]
                # print(predict_labels.type)
                # print(labels.type)
                # print(images.shape)
                #print(labels)
                #print(predict_labels)
                loss = criterion(predict_labels, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # if (i+1) % 1500 == 0:
                #     print("epoch:", epoch, "step:", i, "loss:", loss.item())
                # if (i+1) % 2500 == 0:
                #     print("save model")
            correct = 0
            total = 0
            for i, (images1, labels) in enumerate(test_dataloader):
                image = images1
                #print(image)
                image = Variable(image)
                image, labels =  image.to(device), labels.to(device)
                # image = generator(image)

                predict_label = cnn(image)
                labels = labels.cpu()
                predict_label = predict_label.cpu()
                _, predicted = torch.max(predict_label, 1)
                total += labels.size(0)
                # print(predicted,'>>>>>>>>',labels)
                if(predicted == labels):
                    correct += 1
            print('Test Accuracy of the model on the %d test images (%d): %f %%' % (total, correct, 100 * correct / total))
            if(correct / total>accuracy):
                accuracy = correct / total
                torch.save(cnn.state_dict(), "./model_lake/"+model_name.replace('.','_'+str(ttest_num)+'.')) 
                # torch.save(cnn.state_dict(), "./model_lake/"+model_name)   #current is model.pickle
                print('saved!!!!!!!!!!!!!!!!!!!!!!!')

        # torch.save(cnn.state_dict(), "./"+model_name)   #current is model.pkl
            print("epoch:", epoch, "step:", i, "loss:", loss.item())
    # print('final accuracy: ',accuracy)
        accuracy_list.append(accuracy)
        print(accuracy_list)
    # torch.save(cnn.state_dict(), "./"+model_name)   #current is model.pkl
    # print("save last model")

if __name__ == '__main__':
    main()


