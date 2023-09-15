# NOTEBOOK IMPORTS
import os, glob, zipfile, random
import numpy as np
from tqdm.notebook import tqdm
from shutil import copyfile
from datetime import datetime

# IMAGE IMPORTS
import cv2
from PIL import Image

# GIS IMPORTS
import fiona, pyproj
from affine import Affine
from shapely.geometry import shape, mapping, Point, LineString
from shapely.ops import transform, nearest_points, snap
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask
from scipy.spatial import cKDTree

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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# PLOTTING IMPORTS
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# CUSTOM UTILITIES
from WorldFileUtils import *
from GeometryUtils import *
from icp import *
from DataUtils import *

Image.MAX_IMAGE_PIXELS = 933120000


def extractZipFiles(zip_dir, extract_dir):
    # Loop through all files in the ZIP directory
    for filename in os.listdir(zip_dir):
        if filename.endswith('.zip'):
            # Construct the full path for the ZIP file
            zip_path = os.path.join(zip_dir, filename)

            # Open and extract the contents of the ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            print(f'Extracted: {filename} to {extract_dir}')
            
# Define a function to extract the first consecutive numerical characters from a string
def extract_numerical_chars(text):
    numerical_chars = ''
    for char in text:
        if char.isdigit():
            numerical_chars += char
        else:
            break
    return numerical_chars
    
class NN_Multiclass(Dataset):
    def __init__(self, input_folder, target_folder, transform=None, crop=True, resize=False, resize_def=2048, cropsize=1024):
        self.input_folder = input_folder
        self.target_folder = target_folder         
        self.transform = transform
        self.crop = crop
        self.cropsize = cropsize
        self.image_filenames = os.listdir(input_folder)      
        self.images_unscaled = list()        
        
        for fn in self.image_filenames:
            image = Image.open(os.path.join(self.input_folder, fn)).convert('L')
            image = Image.fromarray(np.array(image).astype(np.uint8))
            self.images_unscaled.append(image)
                
        self.targets_unscaled = loadClasses(self.target_folder, fns=self.image_filenames)
        
        if resize:            
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
            croptrans = transforms.Compose([RandomCrop((self.cropsize, self.cropsize))])
            sample_transformed = croptrans(sample)
            input_image, target_image = sample_transformed['image'], sample_transformed['target']
        
        target_image = target_image.to(torch.long).squeeze()
        
        return input_image.float(), target_image, self.image_filenames[index]
        
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
    
    
    # TODO: IMPLEMENT ONLY PARTIAL RUNNING.
    # if fns is not None:
    #     # Iterate over the class folders
    #     for class_folder in class_folders:
    #         class_folder_path = os.path.join(folder_path, class_folder)
    #         if os.path.isdir(class_folder_path):
    #             image_files = os.listdir(class_folder_path)
    #             # Iterate over the image files within the class folder
    #            for file_name in image_files:
    #                image_names[file_name] = True
    #                labels.append(class_folder)
    #else: 
    for class_folder in class_folders:
        for file_name in fns:
            image_names[file_name] = True
            labels.append(class_folder)
    #print(image_names)
    
    classes = np.unique(labels)
    outputs = list()
    
    for image_name in tqdm(list(image_names.keys())):
        output = None
        for i, classification in enumerate(classes):
            
            fn = f"{folder_path}/{classification}/{image_name}"
            
            try:
                current_image = cv2.imread(fn)
                current_image = np.asarray(current_image)
                
                if current_image.ndim == 3:
                    current_image = current_image[:, :, 0]
                
                if output is None:
                    output = np.zeros(current_image.shape)
                
                
                output = np.where(current_image > 0, i+1, output)
            except:
                raise(Exception(f"Could not find {fn}"))
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
 
'''
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

    return outputs'''



def bomb_edges(image, size=2, dims=[0,1,2]):
    image[:, 0:size, dims] = 255
    image[:, -size:, dims] = 255
    image[0:size, :, dims] = 255
    image[-size:, :, dims] = 255
    
    c = np.setdiff1d([0,1,2], dims)
    image[:, 0:size, c] = 0
    image[:, -size:, c] = 0
    image[0:size, :, c] = 0
    image[-size:, :, c] = 0
    
    return image

def split_and_run_cnn(image, model, tilesize=2048, num_dim=3, edges=3, dims_rep=None):

    if dims_rep is None:
        dims_rep=np.arange(num_dim)

    tensor = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load the image
    # image = Image.open(image_path)
    
    if np.asarray(image).ndim == 3:
        image = Image.fromarray(np.asarray(image)[:,:,0])
    
    
    # Calculate the number of tiles needed
    width, height = image.size
    num_tiles_x = (width + tilesize-1) // tilesize
    num_tiles_y = (height + tilesize-1) // tilesize
    
    # Create an empty list to store the output tiles
    output_tiles = []
    
    output_gen = np.zeros((width, height, num_dim))
    
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
            
            # tile = np.where(tile > 127, 255, 0).astype(np.uint8)
            
            tile = tile.astype(np.uint8)
            
            
            tile_tensor = tensor(tile).unsqueeze(0).to("cuda")
            
            # Run the CNN on the tile
            output = model(tile_tensor)
            
            output = output[0, :, :, :].cpu().detach().numpy().T
            
            # POSTPROCESS 
            if edges != 0:
                output = bomb_edges(output, size=edges, dims=dims_rep)
            
            # Store the output tile
            x_fin = tilesize - pad_width
            y_fin = tilesize - pad_height
            
            temp = output[0:x_fin, 0:y_fin, :]
            
            output_gen[x0:x1, y0:y1, :] = temp
        torch.cuda.empty_cache()
    return output_gen