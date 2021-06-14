
# -*-coding: utf-8 -*-

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import imageee
import os
from create_dataser import TorchDataset
import torch.nn as nn
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from yaspin import yaspin
from torchvision import datasets ,models,transforms
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
from pathlib import Path
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import torch.optim as optim
import shutil


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 7)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

net=VGG("VGG19")

criterion=nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
# optimizer=optim.SGD(net.parameters,lr=0.001,momentum=0.9)
train_filename="output.txt"
image_dir='train' 
epoch_num=2
batch_size=64

test_filename="output.txt"
test_image_dir='train' 
print("loading traindata")
train_data = TorchDataset(filename=train_filename, image_dir=image_dir,repeat=1)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)


test_data = TorchDataset(filename=test_filename, image_dir=test_image_dir,repeat=1)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# for epoch in range(epoch_num):
#     for batch_image, batch_label in train_loader:
#         image=batch_image[0,:]
#         image=image.numpy()
#         image = image.transpose(1, 2, 0)  
#         print("batch_image.shape:{},batch_label:{}".format(batch_image.shape,batch_label))

def save_checkpoint(epoch, model, optimizer):
    '''
        Save model checkpoint
    '''
    state = {'epoch': epoch, "model_weights": model, "optimizer": optimizer}
    filename = f"model_state.pth{epoch}.tar"
    torch.save(state, filename)

def rightness(output, target):
    preds = output.data.max(dim=1, keepdim=True)[1]
    return preds.eq(target.data.view_as(preds)).cpu().sum(), len(target)



for epoch in range(epoch_num):
    train_rights=[]
    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    print('running epoch: {}'.format(epoch))
    ###################
    # train the model #
    ###################
    net.train()
    turn=0

    for data, target in train_loader:
        # data = data.to(device)
        # target = target.to(device)



        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = net(data)
        # calculate the batch loss
        loss = criterion(output, target.squeeze().long())
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
        print(train_loss)
        turn+=1
        # if turn %100 == 0:
        #     net.eval()
        #     val_rights=[]


        #     for data,target in test_loader:
        #         data,target=Variable(data),Variable(target)
        #         output=net(data)
        #         right=rightness(output,target)
        #         val_rights.append(right)

#1


    best_model_wts=net.state_dict()
    save_checkpoint(epoch, best_model_wts, optimizer)
    
























