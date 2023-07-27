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

class SquareLocator(nn.Module):
    def __init__(self, num_classes=2, processing_size=(2048, 2048)):
        super(SquareLocator, self).__init__()
        
        self.prep_transform = transforms.Compose([transforms.Resize(processing_size)])
        
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
        
        x_prep = torch.stack([self.prep_transform(img) for img in x])
        
        # Encoder
        enc1 = self.encoder1(x_prep)
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
            print(os.path.join(self.target_folder, fn))
            target = Image.open(os.path.join(self.target_folder, fn)).convert('L')
            
            target = np.asarray(np.array(target).astype(np.uint8))
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
            croptrans = transforms.Compose([RandomCrop((512, 512))])
            sample_transformed = croptrans(sample)
            input_image, target_image = sample_transformed['image'], sample_transformed['target']
        
        target_image = torch.where(target_image > 0, 1, 0).to(torch.long).squeeze()
        
        return input_image, target_image, self.image_filenames[index]

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

    return padded_images, masks, (height, width)

class SquareDataset_Multiclass(Dataset):
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
            self.images, self.masks, self.original_shapes  = dynamic_pad_resize(self.images_unscaled, (resize_def, resize_def))
            self.targets, _ , _ = dynamic_pad_resize(self.targets_unscaled, (resize_def, resize_def))
        else:
            self.images  = self.images_unscaled
            self.targets = self.targets_unscaled
        
        
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
        
        return input_image.float(), target_image, self.image_filenames[index]
def split_and_run_cnn(image_path, model, tilesize=2048):
        
    tensor = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load the image
    image = Image.open(image_path)
    
    # Calculate the number of tiles needed
    width, height = image.size
    num_tiles_x = (width + tilesize-1) // tilesize
    num_tiles_y = (height + tilesize-1) // tilesize
    
    # Create an empty list to store the output tiles
    output_tiles = []
    
    output_gen = np.zeros((width, height))
    
    # Iterate over each tile
    for tile_x in tqdm(range(num_tiles_x)):
        for tile_y in range(num_tiles_y):
                        
            # Calculate the coordinates for the current tile
            x0 = tile_x * tilesize
            y0 = tile_y * tilesize
            x1 = min(x0 + tilesize, width)
            y1 = min(y0 + tilesize, height)
            
            # Crop the image to the current tile
            tile = image.crop((x0, y0, x1, y1))
            
            # Pad the tile if needed
            pad_width = tilesize - tile.width
            pad_height = tilesize - tile.height
            if pad_width > 0 or pad_height > 0:
                padding = ((0, pad_height), (0, pad_width))
                tile = np.pad(tile, padding, mode='constant')
            
            # Preprocess the tile
            tile = np.array(tile)
            
            if np.max(tile) == 1:
                tile = tile * 255
            
            tile = np.where(tile > 127, 255, 0).astype(np.uint8)
            
            tile_tensor = tensor(tile).unsqueeze(0).to("cuda")
            
            # Run the CNN on the tile
            output = model(tile_tensor)
            
            output = output[0, 1, :, :].cpu().detach().numpy().T
            
            # Store the output tile
            
            x_fin = tilesize - pad_width
            y_fin = tilesize - pad_height
            
            temp = output[0:x_fin, 0:y_fin]
            
            
            output_gen[x0:x1, y0:y1] = temp
        torch.cuda.empty_cache()
    return output_gen.T
