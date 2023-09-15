# PYTHON IMPORTS
import os
import copy
from tqdm.notebook import trange, tqdm

# IMAGE IMPORTS 
from PIL import Image
import cv2

# DATA IMPORTS 
import random
import h5py
import numpy as np

# PLOTTING
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# NEURAL NETWORK
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.transforms import ToPILImage, GaussianBlur
from torchvision.transforms import Compose, RandomCrop, ToTensor, Normalize
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as models

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Calculate channel-wise average pooling
        avg_pool = torch.mean(x, dim=1, keepdim=True)

        # Calculate channel-wise max pooling
        max_pool, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate both pooling results along the channel dimension
        pool = torch.cat([avg_pool, max_pool], dim=1)

        # Apply convolutional layer and sigmoid activation
        attention = torch.sigmoid(self.conv(pool))

        # Multiply the input feature map by the attention map
        attended_feature_map = x * attention

        return attended_feature_map


class TPNN(nn.Module):

    def __init__(self, num_classes=2, finalpadding=0, inputsize=1, verbose_level=1):
        super(TPNN, self).__init__()
        
        self.softmax = nn.Softmax()
        
        self.attention = SpatialAttention().to("cuda:0")
        
        self.verbose_level = int(verbose_level)
        
        # ResNet backbone
        self.resnet = models.resnet34(pretrained=True)
        
        # Adjust the first convolutional layer to accept single-channel input
        self.resnet.conv1 = nn.Conv2d(inputsize, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Encoder
        self.encoder1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4
        
        # Decoder
        self.decoder4 = self._make_decoder_block(512, 256, 256)
        self.decoder3 = self._make_decoder_block(512, 128, 128)
        self.decoder2 = self._make_decoder_block(256, 64, 64)
        self.decoder1 = self._make_decoder_block(128, 64, 64, s=4)
        
        # Final convolutional layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1, padding=finalpadding)
      
    def notify(self, mess, level=4):
        if self.verbose_level >= level:
            print(mess)
      
    def _make_decoder_block(self, in_channels, mid_channels, out_channels, s=2):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=s),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, resize=True):
                
        # Spatial self-attention layer
        x    = self.attention(x)
        
        # Encoder
        enc1 = self.encoder1(x)
        self.notify((enc1.shape))
        enc2 = self.encoder2(enc1)
        self.notify((enc2.shape))
        enc3 = self.encoder3(enc2)
        self.notify((enc3.shape))
        enc4 = self.encoder4(enc3)
        self.notify((enc4.shape))
        enc5 = self.encoder5(enc4)
        self.notify((enc5.shape))
        
        
        # Decoder with residual connections
        dec4 = self.decoder4(enc5)
        self.notify((dec4.shape, enc4.shape))
        dec3 = self.decoder3(torch.cat([dec4, enc4], dim=1))
        self.notify((dec3.shape, enc3.shape))
        dec2 = self.decoder2(torch.cat([dec3, enc3], dim=1))
        self.notify((dec2.shape, enc2.shape))
        dec1 = self.decoder1(torch.cat([dec2, enc2], dim=1))
        self.notify((dec1.shape, enc1.shape))
        # Final convolutional layer
        output = self.final_conv(dec1)
        
        # Reshape output to match input dimensions
        if resize:
            output = nn.functional.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        output = self.softmax(output)
        
        return output