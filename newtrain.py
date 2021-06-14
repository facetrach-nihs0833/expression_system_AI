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
from PIL import Image
from tensorboardX import SummaryWriter



use_gpu = torch.cuda.is_available()
#-------------------------dataset-------------------------------
def default_loader(path):
    return Image.open(path).convert('RGB')


class TorchDataset(Dataset):
    def __init__(self, filename, image_dir, resize_height=32, resize_width=32, repeat=1,loader=default_loader):

        self.image_label_list = self.read_file(filename)
        self.image_dir = image_dir
        self.len = len(self.image_label_list)
        self.repeat = repeat
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.loader=loader


        self.toTensor = transforms.ToTensor()


 
    def __getitem__(self, i):
        index = i % self.len

        image_name, label = self.image_label_list[index]
        image_path = os.path.join(self.image_dir, image_name)
        img=self.loader(image_path)
        #img = self.load_data(image_path, self.resize_height, self.resize_width, normalization=False)
        img = self.data_preproccess(img)
        label=np.array(label)
        return img, label
 
    def __len__(self):
        return len(self.image_label_list)
 
    def read_file(self, filename):
        image_label_list = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                content = line.rstrip().split(' ')
                name = content[0]
                labels = []
                for value in content[1:]:
                    labels.append(int(value))
                image_label_list.append((name, labels))
        return image_label_list
 
    def load_data(self, path, resize_height, resize_width, normalization):

        image = imageee.read_image(path, resize_height, resize_width, normalization)
        return image
 
    def data_preproccess(self, data):

        data = self.toTensor(data)
        return data



train_data = TorchDataset(filename="output.txt", image_dir="train",repeat=1)
test_data = TorchDataset(filename="outputtest.txt", image_dir="test",repeat=1)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64)


#-------------------------NET-------------------------------
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
print(net)



optimizer = torch.optim.Adam(net.parameters())
loss_func = torch.nn.CrossEntropyLoss()

if(use_gpu):
    net = net.cuda()
    loss_func = loss_func.cuda()

def save_checkpoint(epoch, model, optimizer):
    '''
        Save model checkpoint
    '''
    state = {'epoch': epoch, "model_weights": model, "optimizer": optimizer}
    filename = f"model_state.pth{epoch}.tar"
    torch.save(state, filename)


#-------------------------start-------------------------------

for epoch in range(2):
    print(f"now epoch{epoch}")
    train_loss=0.0
    train_acc=0.0

    for x,y in train_loader:
        x, y = Variable(x), Variable(y) 
        if (use_gpu):
            x,y = x.cuda(),y.cuda()
        


        out=net(x)
        loss=loss_func(out,y.squeeze().long())
        train_loss+=loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == y).sum()
        train_acc += train_correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(train_loss/len(train_data))
    print(f"loss:{train_loss/len(train_data)},acc:{train_acc/len(train_data)}")

    net.eval()
    eval_loss = 0.
    eval_acc = 0.
    for x, y in test_loader:
        x, y = Variable(x, volatile=True), Variable(y, volatile=True)
        if (use_gpu):
            x,y = x.cuda(),y.cuda()
        out = net(x)
        loss = loss_func(out, y.squeeze().long())
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == y).sum()
        eval_acc += num_correct.item()
        
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_data)), eval_acc / (len(test_data))))
    best_model_wts=net.state_dict()
    save_checkpoint(epoch, best_model_wts, optimizer)
