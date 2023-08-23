# PYTHON IMPORTS
import os, copy, math
from tqdm.notebook import trange, tqdm

# IMAGE IMPORTS 
from PIL import Image, ImageDraw
import cv2

# DATA IMPORTS 
import random, h5py, glob
import numpy as np

# PLOTTING
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# NEURAL NETWORK
import torch

# SHAPES IMPORTS
from shapely.ops import unary_union
from shapely.geometry import LineString

# MY OWN CLASSES
from TileLocator import *
from SquareLocator import *

data_dir = r"C:\Users\fhacesga\OneDrive - University Of Houston\AAA_RECTDNN\data/"

# Process the image and get the result
def upscale_to_max(image, target_dim, threshold=50):
    resized_image = cv2.resize(image, target_dim, interpolation=cv2.INTER_LINEAR)
    resized_image[resized_image > threshold] = 255  # Set pixel values above threshold to max (255)
    return resized_image

def identifyBiggestContour(image, initial_image=None):
        
    if initial_image is None:
        initial_image = image
          
    if initial_image.ndim == 3:
        initial_image = initial_image[:, :, 0]

    # PREPROCESS IMAGE
    sharpened_image = np.where(image > 150, 255, 0).astype(np.uint8)
    edges = cv2.Canny(sharpened_image, 1, 1)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    # LOOP THROUGH IDENTIFIED CONTOURS, SIMPLIFY, AND FILTER BY NUMBER / AREA
    area_ratios = list()
    if image.ndim == 2:
        image_size = image.size
    else:
        image_size = image[:, :, 0].size

    for contour in contours:
        area_ratio = cv2.contourArea(contour) / image_size
        area_ratios.append(area_ratio)

    idx = np.argmax(np.array(area_ratios))
    rectangles_image = np.dstack((initial_image.copy(), initial_image.copy(), initial_image.copy()))
    cv2.drawContours(rectangles_image, contours, -1, (0, 0, 255), 10)
    cv2.drawContours(rectangles_image, [contours[idx]], -1, (0, 255, 0), 10)
    
    epsilon = 0.1 * cv2.arcLength(contours[idx], True)
    approx = cv2.approxPolyDP(contours[idx], epsilon, True)
    cv2.drawContours(rectangles_image, [approx], -1, (255, 0 , 0), 10)
    
    
    return rectangles_image, contours[idx], approx

def findSquares(image, model=None, model_checkpoint=f"{data_dir}SquareLocator/checkpoint_072923.pth"):
    
    # Initialize model
    if model is None:
        model = SquareLocator(finalpadding=1)
        model.load_state_dict(torch.load(model_checkpoint)['model_state_dict'])
    model = model.to("cuda")

    tensor = transforms.Compose([transforms.ToTensor()])
    
    # INPUT IMAGE AND PREP
    shape = image.shape
    image = cv2.resize(image, (512, 512))   
    image_prep = tensor(image).unsqueeze(0).to("cuda")

    # PROCESS IMAGE
    outputs = model(image_prep)
    outputs = outputs[0, :, :, :].detach().cpu().numpy()
    
    # POSTPROCESS
    outputs = outputs * 255
    outputs = outputs.astype(np.uint8)
    outputs = np.moveaxis(outputs, 0, 2)
    
    outputs = cv2.resize(outputs, (shape[1], shape[0]))
    model = model.to("cpu")
    torch.cuda.empty_cache()
    
    return outputs, model

def findKeypoints(image, model=None, model_checkpoint=f"{data_dir}TileLocator/checkpoint_080323.pth"):
    
    # Input handling
    if isinstance(image, np.ndarray):
        image = [image] # Make iterable if needed
    
    # Initialize model if needed
    if model is None:
        model = RectangleClass(num_classes=3)
        model.load_state_dict(torch.load(model_checkpoint)['model_state_dict'])
    model = model.to("cuda")
    
    # PROCESS IMAGE
    for im in image:
        outputs = split_and_run_cnn(Image.fromarray(im), model, tilesize=1024, edges=0, dims_rep=[0])
    
    # background, grid, roads = outputs[:, :, 0], outputs[:, :, 1], outputs[:, :, 2]
    
    model = model.to("cpu")
    torch.cuda.empty_cache()
    
    # return (background.T, grid.T, roads.T), model
    return outputs, model

def extend_lines(lines, percentage):
    extended_lines = []

    for line in lines:
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1

        extension = ((dx ** 2 + dy ** 2) ** 0.5) * (percentage / 100)

        extended_x1 = x1 - dx * (extension / ((dx ** 2 + dy ** 2) ** 0.5))
        extended_y1 = y1 - dy * (extension / ((dx ** 2 + dy ** 2) ** 0.5))
        extended_x2 = x2 + dx * (extension / ((dx ** 2 + dy ** 2) ** 0.5))
        extended_y2 = y2 + dy * (extension / ((dx ** 2 + dy ** 2) ** 0.5))

        extended_lines.append((extended_x1, extended_y1, extended_x2, extended_y2))

    return np.array(extended_lines).astype(int)

def plotLines(original_image, lines, ax=None, mask=None, color=(0, 0, 255), fig_size=(15, 15)):
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
        
    image_with_lines = original_image.copy()
    for line in lines:
        line = [int(x) for x in line]
        x1, y1, x2, y2 = line
        
        if not color:
            curr_color = np.random.randint(255, size=3).astype(np.int32)
            curr_color = (int(curr_color[0]), int(curr_color[1]), int(curr_color[2]))
        else:
            curr_color = color
        cv2.line(image_with_lines, (x1, y1), (x2, y2), curr_color, 2)
    line_arry = np.array([line for line in lines])
    
    if mask is not None:
        image_with_lines = np.where(temp > 0, image_with_lines, 0)
    
    ax.scatter(line_arry[:, 0], line_arry[:, 1])
    ax.scatter(line_arry[:, 2], line_arry[:, 3])
    ax.imshow(image_with_lines)
    return ax

def calculate_scale_factors_small(original_size, target_size):
    scale_x = target_size[0] / original_size[0]
    scale_y = target_size[1] / original_size[1]
        
    if scale_x > scale_y:
        scale_y = scale_y / scale_x
        scale_x = 1
    else:
        scale_y = scale_x / scale_y
        scale_x = 1
    return scale_x, scale_y

def calculate_scale_factors_large(original_size, target_size):
    scale_x = original_size[0] / target_size[0]
    scale_y = original_size[1] / target_size[1]
    return scale_x, scale_y

def rescale_lines(lines, scale_x, scale_y):
    transformation_matrix = np.array([
        [scale_y,       0, 0],
              [0, scale_x, 0],
              [0,       0, 1]
    ])
    rescaled_lines = list()
    for x0, y0, x1, y1 in lines:
        x0_t, y0_t, _ = np.array([x0, y0, 0]).T @ transformation_matrix
        x1_t, y1_t, _ = np.array([x1, y1, 0]).T @ transformation_matrix
        rescaled_lines.append([x0_t, y0_t, x1_t, y1_t])       
    return rescaled_lines
    
def calcAngles(lines):
    return [np.round(math.degrees(math.atan2(line[3] - line[1], line[2] - line[0])),2) for line in lines]

def filterLines_MostPopularAngles(values, N, n_return=2):
    unique_values, value_counts = np.unique(values, return_counts=True)
    
    def thresh_index(values, value, N, ignore_mask=None, verbose=False):
        threshold = np.logical_and(values >= value - N, values <= value + N)
        if ignore_mask is not None:
            threshold = np.logical_and(threshold, ignore_mask)
        return threshold
    
    def calculate_score(value):
        count = value_counts[np.where(unique_values == value)[0][0]]
        neighbors_count = np.sum(value_counts[thresh_index(unique_values, value, N)])        
        return count + neighbors_count
    
    scores        = np.array([calculate_score(x) for x in values])
    sorted_idx    = np.array(scores.argsort()[::-1])
    sorted_values = values[sorted_idx]     
    
    return_list    = list()
    return_indices = list()
    ignore_mask    = np.full(sorted_values.shape, False)
    for n in range(n_return):
        curr_val = sorted_values[~ignore_mask][0]
        return_list.append(curr_val)
        indices = thresh_index(sorted_values, curr_val, N, ignore_mask=~ignore_mask, verbose=True)
        ignore_mask = np.logical_and(~ignore_mask, indices)
        
        return_indices.append(np.where(indices)[0])
    return return_list, return_indices, sorted_idx
    
    
def extend_lines_to_edges(lines, image_shape):
    extended_lines = []

    for line in lines:
        x1, y1, x2, y2 = line
        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf

        if np.abs(slope) > 1:  # Vertical line
            x1_new = int(x1 - y1 / slope)
            y1_new = 0
            x2_new = int(x2 + (image_shape[0] - y2) / slope)
            y2_new = image_shape[0]
        else:  # Horizontal line or near-horizontal line
            y1_new = int(y1 - x1 * slope)
            x1_new = 0
            y2_new = int(y2 + (image_shape[1] - x2) * slope)
            x2_new = image_shape[1]

        extended_lines.append(np.array([x1_new, y1_new, x2_new, y2_new]))

    return extended_lines

def filter_lines_by_distance(lines, min_distance):
    filtered_lines = []
    indices = []
    
    for i, line1 in enumerate(lines):
        x1, y1, x2, y2 = line1

        keep_line = True
        for line2 in filtered_lines:
            x1_other, y1_other, x2_other, y2_other = line2

            if (
                np.linalg.norm(np.array([x1, y1]) - np.array([x1_other, y1_other])) < min_distance
                or np.linalg.norm(np.array([x2, y2]) - np.array([x1_other, y1_other])) < min_distance
            ):
                keep_line = False
                break

        if keep_line:
            filtered_lines.append(line1)
            indices.append(i)

    return filtered_lines, indices
    
def splitHorizontalVertical(arrays, numbers):
    result_dict = {num: 0 for num in numbers}
    
    for idx, array in enumerate(arrays):
        for num in numbers:
            if num in array:
                result_dict[num] = idx
    
    return np.array(list(result_dict.values())).flatten()
    
def lines_to_linestrings(lines):
    linestrings = []
    for x0, y0, x1, y1 in lines:
        linestring = LineString([(x0, y0), (x1, y1)])
        linestrings.append(linestring)
    return linestrings

def linestrings_to_lines(linestrings):
    lines = []
    for linestring in linestrings:
        x0, y0 = linestring.coords[0]
        x1, y1 = linestring.coords[-1]
        lines.append([x0, y0, x1, y1])
    return lines
