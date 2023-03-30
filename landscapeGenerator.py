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

base_dir = "/Users/kennethzhang/Desktop/landscape Images/"

ok = read_image(base_dir + "color/" 
                + "0.jpg")

batch_size = 16
epochs = 20
num_workers = 0

total_images = len(os.listdir(base_dir + 'color'))
random_indices = random.sample(list(range(total_images)), total_images)

split_idx = round(total_images * 0.8)
train_idx = random_indices[:split_idx]
test_idx = random_indices[split_idx:]

train_transforms = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
test_transforms = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

train_dataset = EncoderDataset(train_idx, base_dir)
test_dataset = EncoderDataset(test_idx, base_dir)

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size, batch_size)

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

model = ColorizationAutoencoder() 

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

n_epochs = 30
train_losses = []
test_losses = []

for epoch in range(1, n_epochs + 1):

    train_loss = 0.0

    for data in train_loader:
        images, labels = data

        images = images.float()
        labels = labels.float()

        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        train_loss += loss.item() * images.size(0)
    
    print('Epoch: {}\Training Loss: {:.6f}'.format(
        epochs, train_loss
    ))  

    loss = train_loss / len(train_loader)
    train_losses.append(loss)      

    test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            output = model(images)
            loss = criterion(output, labels)
            test_loss += loss.item() * images.size(0)

        print("Test Loss: {:.3f}.. ".format(
        test_loss
        ))

    test_loss = test_loss / len(test_loader)
    test_losses.append(test_loss)