# NOTEBOOK IMPORTS
import os, glob, zipfile, random
import numpy as np
from tqdm.notebook import tqdm


# IMAGE IMPORTS
import cv2
from PIL import Image, ImageFilter

'''
from shutil import copyfile
from datetime import datetime

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

import torch.nn as nn
import torch.optim as optim
'''

# NEURAL NETWORK
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import RandomCrop

# PLOTTING IMPORTS
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# CUSTOM UTILITIES
# from WorldFileUtils import *
# from GeometryUtils import *
# from icp import *
# from DataUtils import *

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
    
class ColorJitter_Regions(object):
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, apply_every=0.5):
        self.apply_every = apply_every
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.color_jitter = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)

    def __call__(self, image):
        width, height = image.size
        
        # RANDOMLY DO IT OR NOT
        if random.uniform(0, 1) < self.apply_every:
            return image
        
        # Create a random mask with values between 0 and 1
        mask = np.random.rand(height, width)
        mask = Image.fromarray((mask * 255).astype(np.uint8))

        # Apply ColorJitter transformations to the image based on the mask
        jittered_image = self.color_jitter(image)
        jittered_image = Image.blend(jittered_image, mask, 0.5)
        image = Image.blend(image, jittered_image, 0.5)

        return image
    
class Blur_Regions(object):
    def __init__(self, max_kernel_size=64, max_sigma=1.5, mask_size=(64, 64), apply_every=0.5):
        self.apply_every = apply_every
        self.max_kernel_size = max_kernel_size
        self.max_sigma = max_sigma
        self.mask_size = mask_size

    def __call__(self, image):
        
        # RANDOMLY DO IT OR NOT
        if random.uniform(0, 1) < self.apply_every:
            return image
        
        # Randomly select kernel size and sigma
        kernel_size = random.randint(1, self.max_kernel_size)
        sigma = random.uniform(0, self.max_sigma)

        # Create a random mask
        mask = self.generate_random_mask(image.size, self.mask_size)

        # Apply Gaussian blur to the image based on the mask
        image = self.apply_blur_with_mask(image, mask, kernel_size, sigma)
        return image

    def generate_random_mask(self, image_size, mask_size):
        # Generate a random mask using numpy
        mask = np.random.rand(*mask_size)
        mask = (mask - mask.min()) / (mask.max() - mask.min())  # Normalize to [0, 1]
        
        # Resize the mask to match the image size
        mask = Image.fromarray((mask * 255).astype(np.uint8))
        mask = mask.resize(image_size, Image.BILINEAR)
        
        return mask

    def apply_blur_with_mask(self, image, mask, kernel_size, sigma):
        # Apply Gaussian blur to the image based on the mask
        image_blurred = Image.blend(mask, image.filter(ImageFilter.GaussianBlur(radius=sigma)), 0.5)
        image = Image.blend(image_blurred, image, 0.5)
        return image
    
class NN_Multiclass(Dataset):
    def __init__(self, input_folder, target_folder, 
                 transform=None, 
                 input_only_transform=None,
                 crop=True, 
                 n_pyramids=0,
                 resize=False, 
                 only_true=False, 
                 resize_def=2048, 
                 flip_outputs=False,
                 cropsize=512):
        
        self.input_folder = input_folder
        self.target_folder = target_folder         
        self.transform = transform
        self.crop = crop
        self.cropsize = cropsize
        self.image_filenames = os.listdir(input_folder)
        self.n_pyramids = n_pyramids
        self.images_unscaled = list()        
        self.onlytrueoutputs = only_true
        self.input_transform = input_only_transform
        self.tensor = transforms.Compose([transforms.ToTensor()])
        
        for fn in self.image_filenames:
            image = Image.open(os.path.join(self.input_folder, fn)).convert('L')
            image = Image.fromarray(np.array(image).astype(np.uint8))
            self.images_unscaled.append(image)
                
        self.targets_unscaled = loadClasses(self.target_folder, fns=self.image_filenames, flip=flip_outputs)
        
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
            
        if self.input_transform is not None:
            input_image = self.input_transform(input_image)
            
        if not isinstance(input_image, torch.Tensor):
            input_image  = self.tensor(input_image)
            
        if not isinstance(target_image, torch.Tensor):
            target_image = self.tensor(target_image)
            
        if self.crop:
            sample = {'image': input_image, 'target': target_image}
            if self.n_pyramids > 0:
                croptrans = transforms.Compose([RandomPyramidCrop((self.cropsize, self.cropsize), self.n_pyramids)])
                sample_transformed = croptrans(sample)
                input_image, target_image = sample_transformed['image'], sample_transformed['target']
            else:
                croptrans = transforms.Compose([RandomCrop((self.cropsize, self.cropsize))])
                sample_transformed = croptrans(sample)
                input_image, target_image = sample_transformed['image'], sample_transformed['target']
        
        target_image = target_image.to(torch.long).squeeze()
        
        if self.onlytrueoutputs:
            mask = (input_image[0, :, :] < 0.5).squeeze()
            target_image = np.where(mask, target_image, 0)
            # for channel in range(target_image.shape[0]):
            #    target_image[channel, :, :] = np.where(mask, target_image[channel, :, :], 0)
        
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
           
class RandomPyramidCrop(object):
    def __init__(self, size, pyramid_depth):
        self.size = size
        self.pyramid_depth = pyramid_depth
        
    def get_rectangular_region(self, image, x, y, d):
        return get_rectangular_region(image, x, y, d)

    def __call__(self, sample):
        img_i, target_i = sample['image'], sample['target']
        h, w = img_i.shape[1], img_i.shape[2]
        
        new_h, new_w = self.size

        i = random.randint(0, h)
        j = random.randint(0, w)
        
        target = torch.from_numpy(self.get_rectangular_region(target_i, j, i, new_h))
        while torch.sum(target) == 0:
            i = random.randint(0, h)
            j = random.randint(0, w)
            target = torch.from_numpy(self.get_rectangular_region(target_i, j, i, new_h))
        
        
        pyramid = []
        for x in range(self.pyramid_depth):
            image_curr = self.get_rectangular_region(img_i[0, :, :], j, i, new_h * (x+1))
            image_curr = np.moveaxis(image_curr, [0], [2])            
            image_curr_r = cv2.resize(image_curr, (new_h, new_h), interpolation=cv2.INTER_LINEAR)
            # print(f"From {image_curr.shape} to {image_curr_r.shape}")
            pyramid.append(image_curr_r)
            
        img = np.moveaxis(np.dstack(pyramid), [2], [0])  
        img = torch.from_numpy(img)
        # print(img.shape)
        
        
        return {'image': img, 'target': target}

def getClasses(dir):
    classes = os.listdir(dir)
    classes.insert(0, "background")
    return classes

def loadClasses(folder_path, fns=None, flip=False):
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
                
                if flip:
                    current_image = np.where(current_image <= 0, 1, 0)
                
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

def bomb_edges(image, size=2, dims=None):
    if dims is None:
        dims = range(image.shape[-1])

    rep_value = np.max(image)

    image[:, 0:size, dims] = rep_value
    image[:, -size:, dims] = rep_value
    image[0:size, :, dims] = rep_value
    image[-size:, :, dims] = rep_value
    
    c = np.setdiff1d([0,1,2], dims)
    image[:, 0:size, c] = 0
    image[:, -size:, c] = 0
    image[0:size, :, c] = 0
    image[-size:, :, c] = 0
    
    return image
    
def get_rectangular_region(image, x, y, d):
        
    r = d // 2
    
    if image.ndim == 2:
        image = np.expand_dims(image, 0)
        
    width, height = image.shape[-2:]
    
    # Calculate the coordinates of the top-left and bottom-right corners of the region
    x1 = max(0, x - r)
    y1 = max(0, y - r)
    x2 = min(width - 1, x + r)
    y2 = min(height - 1, y + r)
    
    image_region = image[:, x1:x2, y1:y2]

    # Calculate the width and height of the rectangular region
    # region_width = x2 - x1 + 1
    # region_height = y2 - y1 + 1

    # Create an empty array filled with zeros for the result
    result = np.zeros((image.shape[0], r * 2, r * 2))
    
    # Calculate the coordinates in the input image for copying data
    x1_in_image = max(0, r - x)
    y1_in_image = max(0, r - y)
    x2_in_image = x1_in_image + image_region.shape[1]# min(image_region.shape[1], width - x + r)
    y2_in_image = y1_in_image + image_region.shape[2]# min(image_region.shape[2], height - (y + r))
    
    
    # Copy the data from the input image to the result
    result[:, x1_in_image:x2_in_image, y1_in_image:y2_in_image,] = image_region
    
    return result
    
def get_rectangular_region_tl(image, x, y, d):
    
    # WHAT'S THE RADIUS?
    r = d // 2
    
    width, height = image.shape[-2:]
    # Calculate the coordinates of the top-left and bottom-right corners of the region
    x1 = max(0, x - r)
    y1 = max(0, y - r)
    x2 = min(width - 1, x + r)
    y2 = min(height - 1, y + r)

    image_region = image[x1:x2, y1:y2]

    # Create an empty array filled with zeros for the result
    result = np.zeros((d, d))

    # Calculate the coordinates in the input image for copying data
    x1_in_image = max(0, r - x)
    y1_in_image = max(0, r - y)
    x2_in_image = x1_in_image + image_region.shape[0]
    y2_in_image = y1_in_image + image_region.shape[1]
    
    # Copy the data from the input image to the result
    result[x1_in_image:x2_in_image, y1_in_image:y2_in_image,] = image_region

    return result
        
def get_pyramid(image, j, i, pyramid_depth, pyramid_size, plot=False):
    
    j = j + pyramid_size // 2
    i = i + pyramid_size // 2
    
    if plot:
        fig, axs = plt.subplots(1, pyramid_depth, figsize=(5 * pyramid_depth, 5))
    
    pyramid = []
    for x in range(pyramid_depth):
        image_curr = get_rectangular_region_tl(image, j, i, pyramid_size * (x+1))
        image_curr_r = cv2.resize(image_curr, (pyramid_size, pyramid_size), interpolation=cv2.INTER_LINEAR)
        pyramid.append(image_curr_r)
        
        if plot:
            axs[x].imshow(image_curr_r)
        
    pyramid = np.dstack(pyramid)
    return pyramid

def split_and_run_cnn(image, model, tilesize=2048, 
    num_dim=3, edges=3, dims_rep=None, n_pyramids=3, device="cuda"):

    if dims_rep is None:
        dims_rep=np.arange(num_dim)

    tensor = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    if np.asarray(image).ndim == 3:
        image = np.asarray(image)[:, :, 0]

    # CALCULATE HOW MANY TILES WE NEED IN X AND Y DIRECTIONS
    width, height = image.shape
    num_tiles_x = (width + tilesize-1) // tilesize
    num_tiles_y = (height + tilesize-1) // tilesize
    
    # OUTPUT STRUCTURES
    output_tiles = []
    output_gen = np.zeros((width, height, model.num_classes))
    
    # FOR EACH TILE
    for tile_x in tqdm(range(num_tiles_x)):
        for tile_y in range(num_tiles_y):
                        
            # COORDINATES OF CURRENT TILE
            x0 = tile_x * tilesize
            y0 = tile_y * tilesize
            x1 = min(x0 + tilesize, width)
            y1 = min(y0 + tilesize, height)
            x_pad = x1 - x0
            y_pad = y1 - y0
            
            # GET PYRAMIDS TO PROCESS IMAGE
            tile = get_pyramid(image, x0, y0, n_pyramids, tilesize)
            
            # TILE PREPROCESSING
            tile = np.array(tile)                               # AS NUMPY ARRAY
            tile = tile * 255 if np.max(tile) == 1 else tile    # SCALE TO UINT 8
            tile = tile.astype(np.uint8)                        # CHANGE DATA TYPE
            tile_tensor = tensor(tile).unsqueeze(0).to(device)  # TO DEVICE

            # RUN CNN ON TILE
            output = model(tile_tensor)
            
            # PROCESS OUTPUTS OUT OF DEVICE
            if device == "cuda":
                output = output[0, :, :, :].cpu().detach().numpy()
            else:
                output = output[0, :, :, :].detach().numpy()
            
            # POSTPROCESS 
            output = np.moveaxis(output, [0], [2])                      # CHANGE CHANNELS SO BANDS ARE IN LAST DIMENSION
            output_tiles.append(output.copy())                          # APPEND TILE TO LIST
            if edges != 0:
                output = bomb_edges(output, size=edges, dims=dims_rep)  # SET EDGE VALUES FOR TILE TO 0
            
            # PUT IT WITH THE REST
            output_gen[x0:x1, y0:y1, :] = output[:x_pad, :y_pad, :]
        del output, tile, tile_tensor
        torch.cuda.empty_cache()
    return output_gen, output_tiles