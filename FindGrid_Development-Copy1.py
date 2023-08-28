# PYTHON IMPORTS
import os, copy, math
from tqdm.notebook import trange, tqdm

# IMAGE IMPORTS 
from PIL import Image

# DATA IMPORTS 
import random, h5py, glob
import numpy as np

# PLOTTING
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# SHAPES IMPORTS
import shapely
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import LineString, Polygon

# MY OWN CLASSES
from FindGrid import *

# OCR libraries
import pytesseract
from fuzzywuzzy import fuzz
import re

# INITIALIZE
t_path = r'C:\Users\fhacesga\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = t_path

# PREFERENCES
Image.MAX_IMAGE_PIXELS = 933120000


# In[2]:


data_dir = r"C:\Users\fhacesga\OneDrive - University Of Houston\AAA_RECTDNN\data/"

prep_folder   = f"{data_dir}TileLocator/in_prepped_v2/"    
output_folder = f"{data_dir}TileLocator/out_v2/"

image_path = f'{data_dir}TileLocator/in_prepped_v2/48071CIND0A.tif'
 

def wrapFindGrid(image_path):
# Run images through CNNs

# Load the image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

classifications, classModel = findKeypoints(image)
effectiveArea, effectiveAreaModel = findSquares(image)


# Get largest section of mask image

# In[4]:


image_or = cv2.imread(image_path)
image = cv2.cvtColor(image_or, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (512, 512)) 

a, b, c = identifyBiggestContour(effectiveArea[:, :, 1])
image_mask = cv2.drawContours(a[:, :, 0] * 0, contours=[c],contourIdx=-1, 
                              color=(255), thickness=cv2.FILLED)


# Detect lines

# In[5]:


lines, result_image, scale_x, scale_y = line_detection(classifications, effectiveArea)


# Calculate angles and filter by two most popular angles

# In[6]:


angles = calcAngles(lines)

# FILTER BY MOST POPULAR ANGLES
line_angles, line_indices, sorted_idx = filterLines_MostPopularAngles(np.array(angles), 0.5)

# GET RESCALED lines
rescaled_lines_ordered = np.array(lines)[sorted_idx]
filtered_lines = rescaled_lines_ordered[np.concatenate(line_indices).flatten()]


# Extend lines to edges and filter by distance between lines

# In[7]:


extended_lines = extend_lines_to_edges(filtered_lines, image_or.shape)

# Filter lines by distance between endpoints
min_distance = 50 * np.sqrt(scale_x ** 2 + scale_y ** 2)
filtered_lines, filtered_idx = filter_lines_by_distance(extended_lines, min_distance)


# Clip lines

# In[8]:


lines_shp   = lines_to_linestrings(filtered_lines)
split_lines = linestrings_to_lines(unary_union(lines_shp))


#  Run lines on overlap with image and detect which are actually lines in image

# In[9]:


# INVERT AND MAX TO 1
image_or_b = np.asarray(255 - image_or) // 255

if image_or_b.ndim == 3:
    image_or_b = image_or_b[:, :, 0]

overlapping_lines, overlap_image = get_overlapping_lines(split_lines, 
                                                         image_or_b, 
                                                         0.8,)


# Convert lines to an image in which we identify contours

# In[10]:


bw_bounds = draw_lines_to_image(overlapping_lines, (image_or.shape[1], image_or.shape[0]))
contours, hierarchy = cv2.findContours(bw_bounds, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)


# Test which squares are identified

# In[11]:


if False:
    filled_image = np.zeros(image_or.shape)

    # Fill innermost contours with random colors
    for idx, contour in enumerate(contours):
        if hierarchy[0][idx][3] == 0:  # If contour has no child contours
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.drawContours(filled_image, [contour], -1, color, thickness=cv2.FILLED)
    cv2.imwrite("test.png", filled_image)


# Find which squares have a given text

# In[12]:


outdict = {}

key = '48071'

for idx, contour in tqdm(enumerate(contours), total=len(contours)):
    if hierarchy[0][idx][3] == 0:  # If contour has no child contours
        x, y, w, h = cv2.boundingRect(contour)
    
        # Crop the contour region from the image
        cropped_region = image_or[y:y+h, x:x+w, 0]
        cropped_region = pad_image_with_percentage(cropped_region, 20, 20)
        # cropped_region = cv2.resize(cropped_region, (256,256))
        
        # cropped_region = cv2.ximgproc.thinning(cropped_region, thinningType=cv2.ximgproc.THINNING_GUOHALL)
        
        # Perform OCR using pytesseract
        print(idx)
        ocr_text = pytesseract.image_to_string(cropped_region,
                                              config='--psm 12 --oem 3')
        
        text = find_word_with_key(ocr_text, key)
        if text is None:
            fn = f"tempfiles/a_test{idx}.png"
            cv2.imwrite(fn, cropped_region)
            continue
        else:
            fn = f"tempfiles/b_test{idx}.png"
            cv2.imwrite(fn, cropped_region)
        
        outdict[text] = contour


# Save to shapefile

# In[13]:


contour_dict = outdict.copy()

# Convert OpenCV contours to Shapely objects
contours_shapely = {}
for i, (k, v) in enumerate(contour_dict.items()):
    contour_points = [tuple(point[0]) for point in v]
    contours_shapely[k] = Polygon(contour_points).simplify(tolerance=10)
    
create_shapefile_from_dict(contours_shapely, r"tempfiles/test.shp")

