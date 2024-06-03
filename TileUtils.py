# NOTEBOOK IMPORTS
import os, glob, warnings, pickle, re
import numpy as np
from shutil import copyfile, rmtree
from datetime import datetime
from fuzzywuzzy import process

# IMAGE IMPORTS
from PIL import Image

# GIS IMPORTS
from affine import Affine
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point, MultiPoint, box

# PLOTTING IMPORTS
import matplotlib.pyplot as plt

# CUSTOM UTILITIES
from IndexUtils import * 
from WorldFileUtils import * 

Image.MAX_IMAGE_PIXELS = 933120000
warnings.filterwarnings("ignore")
initialize = False

fips_rep = {
            "4201" : "ESRI:103158",
            "4202" : "ESRI:103159",
            "4203" : "ESRI:103160",
            "4204" : "ESRI:103161",
            "4205" : "ESRI:103162"
        }

import re
def findIndexKey(mystring):
    # INDEX KEY IS DEFINED AS ANY NUMBERS BEFORE AN "_" AND "IND" 
    return re.split(r"[^0-9]", os.path.basename(mystring).split(".")[0].split("_")[0])[0]

def read_world_file(filepath):
    """Read a world file and return its contents as a list of lines."""
    with open(filepath, 'r') as file:
        lines = file.readlines()
    mylines = [line.strip() for line in lines if line.strip() != '']
    return mylines

def read_world_files_from_directory(directory):
    """Read all world files from the given directory."""
    world_files_data = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.tfw', '.pgw', '.jpw')):
            filepath = os.path.join(directory, filename)
            file_data = read_world_file(filepath)
            world_files_data.append([filename] + file_data)
    return world_files_data

def getGeotransform(fn, input_dir):
    return get_geotransform_from_tfw(os.path.join(input_dir, fn))[0]

def getAffine(geotransform):
    return get_affine_from_geotransform(geotransform)


def findTileKey(p, db):
    curr_name = findIndexKey(p)
    test = db.get(curr_name, None)
    if test is None:
        for i in [6, 5]:
            if test is not None:
                continue
            curr_name = findIndexKey(p)[:i]
            test = db.get(curr_name, None)

    if test is None:
        curr_name = ""
    return curr_name


def longitude_to_utm_epsg(lon):
    """Convert longitude to the corresponding UTM EPSG code for the Northern Hemisphere."""
    utm_zone = int((lon + 180) / 6) + 1
    epsg_code = 32600 + utm_zone
    return f"EPSG:{epsg_code}"

def getEPSG(row, df_plane, default_stateplane=32039, default_utm=32615):
    if row["geometry"] is None:
        # print("COULD NOT MATCH GEOMETRY")
        return None
    if row["STATEPLANE"]:
        # FIND THE INTERSECTION BETWEEN THE CORRESPONDING POLYGON AND THE STATEPLANE AREA
        curr_df = df_plane[row['filename'] == df_plane['filename']] 

        # FIND WHICH OF THE MATCHING POLYGONS IS THE LARGEST 
        bestrow = curr_df.loc[curr_df.area.idxmax()]         

        # RETURN IT'S EPSG IF IT'S DEFINED    
        retval = fips_rep.get(bestrow['FIPSZONE'], None)   
        if retval is None:
            print("COULD NOT FIND MATCHING STATEPLANE, RETURNING DEFAULT")
            return None
        return retval
    else:
        return longitude_to_utm_epsg(row['geometry'].centroid.x)
    
def getIndivDict(a, index):
    # PROCESSES TILE OUTPUTS TO ONLY INCLUDE TILE BBOXES AND NAMES
    out = []
    for k, v in a[index].items():
        if k in ["county", "transform_info", "output_transform",]:
            continue
        out.append({'tile' : k, 'index' : index, 'coords' : v['coords'], })
    return out

def findBounds(image_fn, model=None, 
        model_weights=f"{data_dir}RLNN/weights050124.pt",
        creation_params=None,
        device="cpu",
        verbose=True
        ):
    
    if creation_params is None:
        target_size = 512
        original_shapes = []

        # COCO DATASET PARAMS
        category_labels = {
            0 : "County",
            1 : "Tile",
            2 : "Box",
            3 : "Legend"
        }

        categories=[0, 1]

    input_folder = os.path.dirname(os.path.abspath(image_fn))

    # Initialize model
    if model is None:
        model = ultralytics.YOLO(model_weights).to("cpu")

    model = model.to(device)
    
    results = model(image_fn, imgsz=target_size, verbose=verbose)

    if device == "cuda":
        results = [result.cpu() for result in results]
        model   = model.to("cpu")

    return results, model

def bbox_to_polygon(bbox):
    # Function to convert [left, bottom, right, top] to a shapely Polygon
    if len(bbox) == 0:
        return None
    left, bottom, right, top = bbox
    return box(left, bottom, right, top)