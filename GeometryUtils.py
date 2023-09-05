# NOTEBOOK IMPORTS
import os, glob
import numpy as np
from tqdm.notebook import tqdm

# IMAGE IMPORTS
import cv2
from PIL import Image

# GIS IMPORTS
import fiona, pyproj
from affine import Affine
from shapely.geometry import shape, mapping, Point, LineString
from shapely.ops import transform, nearest_points, snap
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask
from scipy.spatial import cKDTree

# PERSONALIZED IMPORTS
from WorldFileUtils import *

def gradeTransform(raster_path, grading_polygon):
    with rio.open(raster_path,) as src:
        data, _ = mask(src, [grading_polygon.geometry[0]], crop=False)
        data = data.squeeze()
        src.close()
        
    return np.count_nonzero(data)

def get_true_pixel_rc(raster_path, polygon=None, preprocess=None, prep_args={}):
        
    if preprocess is not None:
        with rio.open(raster_path,) as src:
            data, secondary = preprocess(src.read(1), **prep_args)            
            
            # DEFINE WHERE TO SAVE PREPROCESSED IMAGE
            output_path = raster_path[:-4] + "_prep.tif"
            
            # WRITE IMAGE AND WORLD FILE
            cv2.imwrite(output_path, data)
            # cv2.imwrite(output_path + "2.tif", secondary)
            write_world_file_from_affine(src.transform, output_path[:-3]+"tfw")
            
            raster_path = output_path
            
    # Open the raster using rasterio
    with rio.open(raster_path,) as src:
        if polygon is not None:
            data, _ = mask(src, [polygon.geometry[0]], crop=False)
            data = data.squeeze()
        else:
            data = src.read(1)
            
        # cv2.imwrite("tempfiles/test.png", data)
        
        # Get the indices of all non-zero elements in the masked array
        nonzero_indices = np.nonzero(data)        
        
        # Retrieve the row and column indices separately
        rows = nonzero_indices[0]
        columns = nonzero_indices[1]
        
        return columns, rows
    

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
        
def normalize_geometry(gdf, ot):
    """
    Warp by original transform
    """
    ot = ~ot
    geometry = gdf.geometry.affine_transform([ot.a, ot.b, ot.d, ot.e, ot.c, ot.f])
    gdf['geometry'] = geometry
    return gdf
    
def normalize_geometry_opt(coords, ot):
    """
    Warp by original transform, optimized
    """
    ot = np.linalg.inv(np.array(ot).reshape((3,3)))
    
    coords = np.hstack((coords, np.ones((coords.shape[0], 1))))
    
    # Apply the affine transformation to X and Y
    coords_transformed = ot @ coords.T
    
    coords_transformed = coords_transformed.T
    
    return coords_transformed[:, :2]
    
def getActualTransform(ot, ho, scale=1):
    """
    Combine original transform and homography to get actual transform
    """

    ot_a = Affine.from_gdal(*ot)
    ho = ho.flatten()
    ho_a = Affine(*ho[:6])
    first = np.array(Affine.scale(scale) * (ot_a * ho_a))
    second = np.array(Affine.scale(1 - scale) * (ot_a))
    final = first + second
    return Affine(*final[:6])
        
def filter_points_to_unique(A, B):
    # A=coords_gdf.geometry.tolist()
    # B=boundary_points_matching
    
    # Create a set of elements in B for efficient membership checking
    B_set = set(B)

    # Use list comprehension to filter A and B simultaneously
    A, B = zip(*[(a, b) for a, b in zip(A, B) if b not in B_set])

    # Convert the filtered results back to lists if needed
    A = list(A)
    B = list(B)
    
    return A, B
        
def find_points_on_linestring(linestring, points):
    points_on_linestring = []

    for point in tqdm(points):
        test = nearest_points(linestring, point)
        points_on_linestring.append(test[0])

    return points_on_linestring