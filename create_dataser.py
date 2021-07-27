# -*-coding: utf-8 -*-

import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import imageee
import os
 
class TorchDataset(Dataset):
    def __init__(self, filename, image_dir, resize_height=32, resize_width=32, repeat=1):

        self.image_label_list = self.read_file(filename)
        self.image_dir = image_dir
        self.len = len(self.image_label_list)
        self.repeat = repeat
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.testtotensor =  transforms.Compose([transforms.ToTensor(),transforms.Resize((32,32))])


        self.toTensor = transforms.ToTensor()


 
    def __getitem__(self, i):
        index = i % self.len

        image_name, label = self.image_label_list[index]
        image_path = os.path.join(self.image_dir, image_name)
        img = self.load_data(image_path, self.resize_height, self.resize_width, normalization=False)
        img = self.data_preproccess(img)
        label=np.array(label)
        return img, label
 
    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.image_label_list) * self.repeat
        return data_len
 
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
 















 
if __name__=='__main__':
    train_filename="output.txt"

    image_dir='train'
 
 
    epoch_num=2   
    batch_size=7  
    train_data_nums=10
    max_iterate=int((train_data_nums+batch_size-1)/batch_size*epoch_num) 
 
    train_data = TorchDataset(filename=train_filename, image_dir=image_dir,repeat=1)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

    for epoch in range(epoch_num):
        for batch_image, batch_label in train_loader:
            image=batch_image[0,:]
            image=image.numpy()
            image = image.transpose(1, 2, 0)  
            print("batch_image.shape:{},batch_label:{}".format(batch_image.shape,batch_label))

 
    train_data = TorchDataset(filename=train_filename, image_dir=image_dir,repeat=None)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

    for step, (batch_image, batch_label) in enumerate(train_loader):
        image=batch_image[0,:]
        image=image.numpy()
        image = image.transpose(1, 2, 0)  
        print("step:{},batch_image.shape:{},batch_label:{}".format(step,batch_image.shape,batch_label))
        if step>=max_iterate:
            break



