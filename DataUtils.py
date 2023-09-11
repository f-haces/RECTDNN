# NOTEBOOK IMPORTS
import os, glob, zipfile
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