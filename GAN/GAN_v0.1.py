import argparse
import os
import numpy as np
import math
import one_hot_encoding as ohe
from PIL import Image

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)
os.makedirs("model_G", exist_ok=True)
os.makedirs("model_D", exist_ok=True)
cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        #self.init_size = opt.img_size // 4
        #self.l1 = nn.Sequential(nn.Linear(128 * 60*160, 128 * 60*160//16))

        self.conv_blocks = nn.Sequential(
        	nn.Conv2d(1, 64, 3, stride=1, padding=1),
        	nn.BatchNorm2d(64, 0.8),
        	nn.LeakyReLU(0.5, inplace=True),
        	nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.5, inplace=True),
            #nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
            #nn.Softmax()
        )
    def forward(self, z):
        #out = z.view(60*160, -1)

        #out = self.l1(z)
        #print(out.shape)
        out = z.view(z.shape[0], 1, 60, 160)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        def discriminator_block2(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block    
        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block2(128, 256),
            *discriminator_block2(256, 256),
        )

        # The height and width of downsampled image
        #ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(256 * 4*10, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        #print(out.shape)
        out = out.view(out.shape[0], -1)
        #print(out.shape)
        validity = self.adv_layer(out)

        return validity
class mydataset(Dataset):

    def __init__(self, folder, transform=None):
        self.train_image_file_paths = [os.path.join(folder, image_file) for image_file in os.listdir(folder)]
        self.transform = transform

    def __len__(self):
        return len(self.train_image_file_paths)

    def __getitem__(self, idx):
        image_root = self.train_image_file_paths[idx]
        image_name = image_root.split(os.path.sep)[-1]
        image = Image.open(image_root)
        label = Image.open('train/'+image_name)
        #label = ohe.encode(image_name.split('_')[0]) # 为了方便，在生成图片的时候，图片文件的命名格式 "4个数字或者数字_时间戳.PNG", 4个字母或者即是图片的验证码的值，字母大写,同时对该值做 one-hot 处理
        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)
        #label = ohe.encode(image_name.split('_')[0])
        return image, label
transform = transforms.Compose([
    # transforms.ColorJitter(),
    transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def get_train_data_loader():
    dataset = mydataset('test', transform=transform)
    return DataLoader(dataset, batch_size=128, shuffle=False)
dataloader = get_train_data_loader()
# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
#os.makedirs("../../data/mnist", exist_ok=True)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST(
#         "../../data/mnist",
#         train=True,
#         download=True,
#         transform=transforms.Compose(
#             [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#         ),
#     ),
#     batch_size=opt.batch_size,
#     shuffle=True,
# )

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.003, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0035, betas=(0.5, 0.999))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
flag = True
for epoch in range(1000):
    for i, (imgs, label) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        label.cuda()
        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.array(label)).cuda())
        #label = 
        #print(label.shape[0])
        # Generate a batch of images
        gen_imgs = generator(z)
        #first_loss = adversarial_loss(label, imgs)
        #first_loss.backward()
        # Loss measures generator's ability to fool the discriminator


        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        # if flag:
        # 	d_loss = (2*real_loss + fake_loss) / 3
        # 	flag = False
        # else:
        # 	d_loss = (real_loss + 2*fake_loss) / 3
        # 	flag = True
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        #print(discriminator(gen_imgs))
        # print(
        #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        #     % (epoch, 100, i, len(dataloader), d_loss.item(), g_loss.item())
        # )

        batches_done = epoch * len(dataloader) + i
        if batches_done % (len(dataloader)) == 1:
            print( "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, 100, i, len(dataloader), d_loss.item(), g_loss.item()))
            #save_image(z.data[:25], "original.png", nrow=5, normalize=True)
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            torch.save(generator.state_dict(), "model_G/%d.pkl"%batches_done)
            torch.save(discriminator.state_dict(), "model_D/%d.pkl" %batches_done)