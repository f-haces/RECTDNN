# NOTEBOOK IMPORTS
import os
import glob
import numpy as np
from tqdm.notebook import tqdm

# IMAGE IMPORTS
import cv2
from PIL import Image

# GIS IMPORTS
import fiona
import pyproj
from affine import Affine
from shapely.geometry import shape, mapping
from shapely.geometry import Point, LineString
from shapely.ops import transform, nearest_points, snap
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask
from scipy.spatial import cKDTree


def get_true_pixel_coordinates(raster_path, polygon=None):
    # Open the raster using rasterio
    with rio.open(raster_path) as src:
        # Mask the raster with the polygon
        if polygon is not None:
            data, _ = mask(src, [polygon.geometry[0]], crop=False)
            data = data.squeeze()
        else:
            data = src.read(1)
        
        # Get the indices of all non-zero elements in the masked array
        nonzero_indices = np.nonzero(data)        
        
        # Retrieve the row and column indices separately
        rows = nonzero_indices[0]
        columns = nonzero_indices[1]

        # Get the transformation matrix
        transform = src.transform

        # Convert pixel coordinates to real-world coordinates
        x_out = list()
        y_out = list()
        
        for row, col in zip(rows, columns):
            x, y = rio.transform.xy(transform, row, col)
            # print((row, col, x, y))
            x_out.append(x)
            y_out.append(y)
            
        x_out = np.array(x_out)
        y_out = np.array(y_out)

        return np.vstack((x_out, y_out)).T
        
def find_points_on_linestring(linestring, points):
    points_on_linestring = []

    for point in tqdm(points):
        test = nearest_points(linestring, point)
        points_on_linestring.append(test[0])

    return points_on_linestring