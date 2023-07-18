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

class RectangleClass(nn.Module):
    def __init__(self, num_classes=2, ):
        super(RectangleClass, self).__init__()
        
        self.softmax = nn.Softmax()
        
        # ResNet backbone
        self.resnet = models.resnet34(pretrained=True)
        
        # Adjust the first convolutional layer to accept single-channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
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
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def _make_decoder_block(self, in_channels, mid_channels, out_channels, s=2):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=s),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, resize=True):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        
        
        # Decoder with residual connections
        dec4 = self.decoder4(enc5)
        dec3 = self.decoder3(torch.cat([dec4, enc4], dim=1))
        dec2 = self.decoder2(torch.cat([dec3, enc3], dim=1))
        dec1 = self.decoder1(torch.cat([dec2, enc2], dim=1))
        
        # Final convolutional layer
        output = self.final_conv(dec1)
        
        # Reshape output to match input dimensions
        if resize:
            output = nn.functional.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        output = self.softmax(output)
        
        return output


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img_i, target_i = sample['image'], sample['target']
        h, w = img_i.shape[1], img_i.shape[2]
        new_h, new_w = self.size

        if new_h > h or new_w > w:
            raise ValueError(f"Requested crop size ({new_h}, {new_w}) is larger than input size ({h}, {w})")

        i = random.randint(0, h - new_h)
        j = random.randint(0, w - new_w)

        # Crop image and target
        img = img_i[:, i:i+new_h, j:j+new_w]
        target = target_i[:, i:i+new_h, j:j+new_w]

        # Ensure that at least one true value is in target
        while torch.sum(target) == 0:
            i = random.randint(0, h - new_h)
            j = random.randint(0, w - new_w)
            img = img_i[:, i:i+new_h, j:j+new_w]
            target = target_i[:, i:i+new_h, j:j+new_w]

        return {'image': img, 'target': target}

class SegmentationDataset(Dataset):
    def __init__(self, input_folder, target_folder, transform=None, crop=True):
        self.input_folder = input_folder
        self.target_folder = target_folder         
        self.transform = transform
        self.crop = crop
        self.image_filenames = os.listdir(input_folder)      
        self.images = list()
        self.targets = list()
        
        for fn in self.image_filenames:
            image = Image.open(os.path.join(self.input_folder, fn)).convert('L')
            image = Image.fromarray(np.array(image).astype(np.uint8))
            self.images.append(image)
            
        for fn in self.image_filenames:
            target = Image.open(os.path.join(self.target_folder, fn)).convert('L')
            target = np.asarray(np.array(target).astype(np.uint8))
            # target = np.dstack((target, 1-target))
            target = Image.fromarray(target)
            
            # np.dstack(target)
            self.targets.append(target)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        # input_path = os.path.join(self.input_folder, self.image_filenames[index])
        
        input_image = self.images[index]        
        target_image = self.targets[index]
        
        
        # Apply the same random transformation to both image
        seed = np.random.randint(2147483647)            
        if self.transform is not None:
            
            random.seed(seed)
            torch.manual_seed(seed)
            input_image = self.transform(input_image)            
            random.seed(seed)
            torch.manual_seed(seed)
            target_image = self.transform(target_image)
            
        if self.crop:
            sample = {'image': input_image, 'target': target_image}
            croptrans = transforms.Compose([RandomCrop((1024, 1024))])
            sample_transformed = croptrans(sample)
            input_image, target_image = sample_transformed['image'], sample_transformed['target']
        
        # input_image = (input_image * 255).to(torch.float)
        target_image = torch.where(target_image > 0, 1, 0).to(torch.long).squeeze()
        # target_image = nn.functional.one_hot(target_image).permute(0,3,1,2).squeeze()
        
        return input_image, target_image, self.image_filenames[index]

def loadClasses(folder_path):
    class_folders = os.listdir(folder_path)
    labels = []
    image_names = {}

    # Iterate over the class folders
    for class_folder in class_folders:
        class_folder_path = os.path.join(folder_path, class_folder)
        if os.path.isdir(class_folder_path):
            image_files = os.listdir(class_folder_path)
            # Iterate over the image files within the class folder
            for file_name in image_files:
                if file_name.endswith(".png"):
                    image_names[file_name] = True
                    labels.append(class_folder)

    
    classes = np.unique(labels)
    outputs = list()
    
    for image_name in list(image_names.keys()):
        output = None
        for i, classification in enumerate(classes):
            
            fn = f"{folder_path}/{classification}/{image_name}"
            
            current_image = cv2.imread(fn)
            current_image = np.asarray(current_image)
            
            if current_image.ndim == 3:
                current_image = current_image[:, :, 0]
            
            if output is None:
                output = np.zeros(current_image.shape)
            
            output = np.where(current_image > 0, i+1, output)
        outputs.append(Image.fromarray(output))

    return outputs


class SegmentationDataset_Multiclass(Dataset):
    def __init__(self, input_folder, target_folder, transform=None, crop=True):
        self.input_folder = input_folder
        self.target_folder = target_folder         
        self.transform = transform
        self.crop = crop
        self.image_filenames = os.listdir(input_folder)      
        self.images = list()        
        
        for fn in self.image_filenames:
            image = Image.open(os.path.join(self.input_folder, fn)).convert('L')
            image = Image.fromarray(np.array(image).astype(np.uint8))
            self.images.append(image)
            
        self.targets = loadClasses(self.target_folder)
        
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        # input_path = os.path.join(self.input_folder, self.image_filenames[index])
        
        input_image = self.images[index]        
        target_image = self.targets[index]
        
        
        # Apply the same random transformation to both image
        seed = np.random.randint(2147483647)            
        if self.transform is not None:
            
            random.seed(seed)
            torch.manual_seed(seed)
            input_image = self.transform(input_image)            
            random.seed(seed)
            torch.manual_seed(seed)
            target_image = self.transform(target_image)
            
        if self.crop:
            sample = {'image': input_image, 'target': target_image}
            croptrans = transforms.Compose([RandomCrop((1024, 1024))])
            sample_transformed = croptrans(sample)
            input_image, target_image = sample_transformed['image'], sample_transformed['target']
        
        target_image = target_image.to(torch.long).squeeze()
        
        return input_image, target_image, self.image_filenames[index]

