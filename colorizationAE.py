import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
import tqdm

from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torchvision.io import read_image
from skimage.color import rgb2lab, lab2rgb
from colorizationTools import EncoderDataset, ColorizationAutoencoder
from torch.utils.data import DataLoader

import os
import numpy as np
import random

class AEDatasetGenerator(Dataset):

    def __init__(self, base_dir : str, split_size : float, directory_names):
        super(AEDatasetGenerator, self).__init__()

        self.base_dir = base_dir
        self.directory_names = directory_names # assumes gray comes before color image paths

        self.train_transforms = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        self.test_transforms = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

        self.split_size = split_size

        self.total_images = len(os.listdir(self.base_dir + directory_names[0]))
        self.random_indices = random.sample(list(range(self.total_images)), self.total_images)

        self.split_idx = round(self.total_images * 0.8)
        self.train_idx = self.random_indices[:self.split_idx]
        self.test_idx = self.random_indices[self.split_idx:]

    def __len__(self):
        return self.total_images
    
    def __getitem__(self, idx):
        img_name = str(idx) + ".jpg"
        img = read_image(self.base_dir + self.directory_names[0] + img_name)
        img = img.unsqueeze(0)
        img = F.interpolate(img, (160, 160))
        img = img.squeeze(0)

        img = img.permute(1, 2, 0)
        img = img.repeat(1, 1, 3)
        img = img.permute(2, 0, 1)

        label = read_image(self.base_dir + self.directory_names[1] + img_name)
        label = label.unsqueeze(0)
        label = F.interpolate(label, (160, 160))
        label = label.squeeze(0)
        label = label.permute(1, 2, 0)
        label = label.permute(2, 0, 1)
        
        img = torch.tensor(rgb2lab(img.permute(1, 2, 0) / 255))
        label = torch.tensor(rgb2lab(label.permute(1, 2, 0) / 255))

        img = (img + torch.tensor([0, 128, 128])) / torch.tensor([100, 255, 255])
        label = (label + torch.tensor([0, 128, 128])) / torch.tensor([100, 255, 255])

        img = img.permute(2, 0, 1)
        label = label.permute(2, 0, 1)

        img = img[:1, :, :]
        label = label[1:, :, :]
        return img, label

class ColorizationAutoencoder(nn.Module):
    def __init__(self):
        super(ColorizationAutoencoder, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(128, 256, 3, stride = 2, padding = 1)

        self.pooling2d = nn.MaxPool2d(2, 2)

        self.t_conv1 = nn.ConvTranspose2d(256, 128, 3, stride = 2, padding = 1, output_padding = 1)
        self.t_conv2 = nn.ConvTranspose2d(256, 64, 3, stride = 2, padding = 1, output_padding = 1)
        self.t_conv3 = nn.ConvTranspose2d(128, 128, 3, stride = 2, padding = 1, output_padding = 1)
        self.t_conv4 = nn.ConvTranspose2d(192, 15, 3, stride = 1, padding = 1)

        self.dropout = nn.Dropout(0.2)

        self.converge = nn.Conv2d(16, 2, 3, stride = 1, padding = 1)
    
    def forward(self, x):
        
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))

        xd = F.relu(self.t_conv1(x4))
        xd = torch.cat((xd, x3), dim = 1)
        xd = self.dropout(xd)
        xd = F.relu(self.t_conv2(xd))
        xd = torch.cat((xd, x2), dim = 1)
        xd = self.dropout(xd)
        xd = F.relu(self.t_conv3(xd))
        xd = torch.cat((xd, x1), dim = 1)
        xd = self.dropout(xd)
        xd = F.relu(self.t_conv4(xd))
        xd = torch.cat((xd, x), dim = 1)

        x_out = F.relu(self.converge(xd))

        return x_out
    

class ColorizationWeightInitAE(nn.Module):
    def __init__(self, constant_weight = None):
        super(ColorizationWeightInitAE, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(128, 256, 3, stride = 2, padding = 1)

        self.pooling2d = nn.MaxPool2d(2, 2)

        # Manually initialize constant weight if NOT NONE
        if (constant_weight is not None):
            # iterate through the entire module before initializing weights for each instance type
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.constant_(m.weight, constant_weight)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.constant_(m.weight, constant_weight)
                    nn.init.constant_(m.bias, 0)
                else:
                    nn.init.constant_(None, None)
                    nn.init.constant_(None, 0)                

        self.t_conv1 = nn.ConvTranspose2d(256, 128, 3, stride = 2, padding = 1, output_padding = 1)
        self.t_conv2 = nn.ConvTranspose2d(256, 64, 3, stride = 2, padding = 1, output_padding = 1)
        self.t_conv3 = nn.ConvTranspose2d(128, 128, 3, stride = 2, padding = 1, output_padding = 1)
        self.t_conv4 = nn.ConvTranspose2d(192, 15, 3, stride = 1, padding = 1)

        self.dropout = nn.Dropout(0.2)

        self.converge = nn.Conv2d(16, 2, 3, stride = 1, padding = 1)
    
    def forward(self, x):
        
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))

        xd = F.relu(self.t_conv1(x4))
        xd = torch.cat((xd, x3), dim = 1)
        xd = self.dropout(xd)
        xd = F.relu(self.t_conv2(xd))
        xd = torch.cat((xd, x2), dim = 1)
        xd = self.dropout(xd)
        xd = F.relu(self.t_conv3(xd))
        xd = torch.cat((xd, x1), dim = 1)
        xd = self.dropout(xd)
        xd = F.relu(self.t_conv4(xd))
        xd = torch.cat((xd, x), dim = 1)

        x_out = F.relu(self.converge(xd))

        return x_out

def main():

    pass
    
if __name__ == "__main__":
    main()




        

        