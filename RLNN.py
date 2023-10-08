# PYTHON IMPORTS
import os, copy
from tqdm.notebook import trange, tqdm

# IMAGE IMPORTS 
from PIL import Image
import cv2

# DATA IMPORTS 
import random, h5py
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


class RLNN(nn.Module):

    def __init__(self, num_classes=2, finalpadding=0, inputsize=1, verbose_level=1):
        super(RLNN, self).__init__()
        
        self.softmax = nn.Softmax()
        
        self.attention = SpatialAttention()
        
        self.verbose_level = int(verbose_level)
        
        # ResNet backbone
        self.resnet = models.resnet34(pretrained=True)
        
        # Adjust the first convolutional layer to accept single-channel input
        self.resnet.conv1 = nn.Conv2d(inputsize, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.resnet.conv1 = nn.Conv2d(inputsize, 3, kernel_size=1, stride=1, padding=0, bias=False)
        # self.resnet.conv2 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Encoder
        self.encoder1 = nn.Sequential(
            self.resnet.conv1,
            # self.resnet.conv2,
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
                
        # Encoder
        x    = self.attention(x)
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

class RandomScaleTransform:
    def __init__(self, min_scale_height, max_scale_height, min_scale_width, max_scale_width):
        self.min_scale_height = min_scale_height
        self.max_scale_height = max_scale_height
        self.min_scale_width = min_scale_width
        self.max_scale_width = max_scale_width

    def __call__(self, img):
        scale_factor_height = random.uniform(self.min_scale_height, self.max_scale_height)
        scale_factor_width = random.uniform(self.min_scale_width, self.max_scale_width)
        width, height = img.size
        new_width = int(width * scale_factor_width)
        new_height = int(height * scale_factor_height)
        new_img = img.resize((new_width, new_height), Image.BILINEAR)
        # img = np.zeros((width, height))
        return img

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

def loadClasses(folder_path, fns=None):
    class_folders = os.listdir(folder_path)
    labels = []
    image_names = {}
    
    if fns is not None:
        # Iterate over the class folders
        for class_folder in class_folders:
            class_folder_path = os.path.join(folder_path, class_folder)
            if os.path.isdir(class_folder_path):
                image_files = os.listdir(class_folder_path)
                # Iterate over the image files within the class folder
                for file_name in image_files:
                    image_names[file_name] = True
                    labels.append(class_folder)
    else: 
        for class_folder in class_folders:
            for file_name in fns:
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

def dynamic_pad_resize(images, target_size):
    max_height = max(np.asarray(img).shape[0] for img in images)
    max_width = max(np.asarray(img).shape[1] for img in images)

    padded_images = []
    masks = []
    shapes = []
    for img in images:
        
        height, width = np.asarray(img).shape[:2]
        
        # Create a binary mask for the valid pixels (1) and padded regions (0)
        mask = np.ones((max_height, max_width), dtype=np.uint8)
        mask[:height, :width] = 1

        # Pad the image to the maximum height and width
        padded_img = np.zeros((max_height, max_width), dtype=np.uint8)
        padded_img[:height, :width] = img

        # Resize the padded image to the target size
        resized_img = cv2.resize(padded_img, target_size)

        padded_images.append(Image.fromarray(resized_img))
        masks.append(Image.fromarray(mask))
        shapes.append((height, width))

    return padded_images, masks, shapes
    
def dynamic_resize(images, target_size):
    
    output = list()
    
    for img in images: 
        
        img = np.asarray(img)
        height, width = img.shape[:2]

        # Resize the padded image to the target size
        resized_img = cv2.resize(img, target_size)
        
        output.append(Image.fromarray(resized_img))

    return output

class RLNN_Multiclass(Dataset):
    def __init__(self, input_folder, target_folder, transform=None, crop=True, resize=False, resize_def=2048):
        self.input_folder = input_folder
        self.target_folder = target_folder         
        self.transform = transform
        self.crop = crop
        self.image_filenames = os.listdir(input_folder)      
        self.images_unscaled = list()        
        
        for fn in self.image_filenames:
            image = Image.open(os.path.join(self.input_folder, fn)).convert('L')
            image = Image.fromarray(np.array(image).astype(np.uint8))
            self.images_unscaled.append(image)
                
        # print([img.shape for img in self.images_unscaled])
                
        self.targets_unscaled = loadClasses(self.target_folder, fns=self.image_filenames)
        
        if resize:
            # self.images, self.masks, self.original_shapes  = dynamic_pad_resize(self.images_unscaled, (resize_def, resize_def))
            # self.targets, _ , _ = dynamic_pad_resize(self.targets_unscaled, (resize_def, resize_def))
            
            self.images  = dynamic_resize(self.images_unscaled , (resize_def, resize_def))
            self.targets = dynamic_resize(self.targets_unscaled, (resize_def, resize_def))
        else:
            self.images  = self.images_unscaled
            self.targets = self.targets_unscaled
        
        
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        # input_path = os.path.join(self.input_folder, self.image_filenames[index])
        
        input_image  = self.images[index]        
        target_image = self.targets[index]
        
        # Apply the same random transformation to both image
        seed = np.random.randint(2147483647)            
        if self.transform is not None:
            
            random.seed(seed)
            torch.manual_seed(seed)
            input_image = self.transform(input_image)            
            random.seed(seed)
            torch.manual_seed(seed)
            if np.max(np.asarray(target_image)) == 255:
                target_image = self.transform(target_image)
            else:
                target_image = self.transform(target_image)
            
        if self.crop:
            sample = {'image': input_image, 'target': target_image}
            croptrans = transforms.Compose([RandomCrop((1024, 1024))])
            sample_transformed = croptrans(sample)
            input_image, target_image = sample_transformed['image'], sample_transformed['target']
        
        target_image = target_image.to(torch.long).squeeze()
        
        return input_image.float(), target_image, self.image_filenames[index]