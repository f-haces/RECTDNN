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


def findRoads(image, model=None, num_classes=2, num_pyramids=2,
                cnn_run_params=None, cnn_creation_params=None, device="cuda",
                model_checkpoint=f"{data_dir}/FANN/checkpoint_101023.pth"):
    
    if cnn_run_params is None:
        cnn_run_params = {
            "tilesize"   : 2048,
            "edges"      : 0,
            "dims_rep"   : None,
            "n_pyramids" : num_pyramids,
            "num_dim"    : num_classes,
            "device"     : device,
            "verbose"    : False
        }
    
    if cnn_creation_params is None:
        cnn_creation_params = {
            "num_classes" : num_classes,
            "inputsize"   : num_pyramids,
        }
    
    # Input handling
    if isinstance(image, np.ndarray):
        image = [image] # Make iterable if needed
    
    # Initialize model if needed
    if model is None:
        model = TPNN(**cnn_creation_params)
        model.load_state_dict(torch.load(model_checkpoint)['model_state_dict'])
    model = model.to(device)
    
    # PROCESS IMAGE
    for im in image:
        outputs, _ = split_and_run_cnn(im, model, **cnn_run_params)
    
    # background, grid, roads = outputs[:, :, 0], outputs[:, :, 1], outputs[:, :, 2]
    
    model = model.to("cpu")
    torch.cuda.empty_cache()
    
    outputs = outputs * 255
    outputs = outputs.astype(np.uint8)
    
    # return (background.T, grid.T, roads.T), model
    return outputs, model


def get_all_image_paths(directory, image_extensions=['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.tif']):
    """
    Recursively finds all image files in the given directory and its subdirectories.
    
    :param directory: The root directory to start the search from.
    :param image_extensions: A list of image file extensions to look for.
    :return: A list of full file paths to the images found.
    """
    image_paths = []
    
    # Loop through each extension in the list
    for extension in image_extensions:
        # Use glob to find all files matching the extension, recursively
        image_paths.extend(glob.glob(os.path.join(directory, '**', extension), recursive=True))
    
    return image_paths

def findBounds(image_fn, model=None, 
        model_weights=f"{data_dir}RLNN/weights050124.pt",
        creation_params=None,
        device="cpu", verbose=False
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

def findStreetCorners_colorPrep(image_fn, FANN=None, RLNN=None):
    image = np.asarray(Image.open(image_fn))

    if image.ndim == 3:
        image = image[:, :, 0]

    # RUN CNN
    out, FANN = findRoads(image, model=FANN)

    # RUN YOLO
    outbbox, RLNN = findBounds(image_fn)

    # REMOVE ANY CNN OUTSIDE BOUNDS
    bounds = outbbox[0].boxes.xyxy.numpy().astype(np.int32).flatten()
    mask  = np.zeros(out.shape)
    if False:# len(bounds) > 0:
        mask[bounds[1]:bounds[3], bounds[0]:bounds[2], :] = 1

    if np.sum(mask) == 0:
        mask = mask + 1
    out = out * mask
    out = out.astype(np.uint8)

    curr_bounds = mask * 255
    curr_bounds = curr_bounds.astype(np.uint8)

    if np.max(image) < 255:
        image = 255 * image
        image = image.astype(np.uint8)

    print(image.shape, out.shape, curr_bounds.shape)
    test = np.dstack((image, out[:, :, 1], curr_bounds[:, :, 0]))

    return test, FANN, RLNN, bounds

'''
def findRoads(image, model=None, num_classes=2, num_pyramids=2,
                cnn_run_params=None, cnn_creation_params=None, device="cuda",
                model_checkpoint=f"{data_dir}/FANN/checkpoint_101023.pth"):
    
    if cnn_run_params is None:
        cnn_run_params = {
            "tilesize"   : 2048,
            "edges"      : 0,
            "dims_rep"   : None,
            "n_pyramids" : num_pyramids,
            "num_dim"    : num_classes,
            "device"     : device
        }
    
    if cnn_creation_params is None:
        cnn_creation_params = {
            "num_classes" : num_classes,
            "inputsize"   : num_pyramids,
        }
    
    # Input handling
    if isinstance(image, np.ndarray):
        image = [image] # Make iterable if needed
    
    # Initialize model if needed
    if model is None:
        model = TPNN(**cnn_creation_params)
        model.load_state_dict(torch.load(model_checkpoint)['model_state_dict'])
    model = model.to(device)
    
    # PROCESS IMAGE
    for im in image:
        outputs, _ = split_and_run_cnn(im, model, **cnn_run_params)
    
    # background, grid, roads = outputs[:, :, 0], outputs[:, :, 1], outputs[:, :, 2]
    
    model = model.to("cpu")
    torch.cuda.empty_cache()
    
    outputs = outputs * 255
    outputs = outputs.astype(np.uint8)
    
    # return (background.T, grid.T, roads.T), model
    return outputs, model

def findBounds(image_fn, model=None, 
        model_weights=f"{data_dir}RLNN/weights050124.pt",
        creation_params=None,
        device="cpu",
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
    
    results = model(image_fn, imgsz=target_size)

    return results, model


def getRoadPoints(fn, model=None):
    # LOAD IMAGE AS NUMPY ARRAY
    image = np.asarray(cv2.imread(fn))

    # RUN CNN
    out, FANN = findRoads(image, model=model)

    # RUN YOLO
    outbbox, RLNN = findBounds(fn)

    # REMOVE ANY CNN OUTSIDE BOUNDS
    bounds = outbbox[0].boxes.xyxy.numpy().astype(np.int32).flatten()
    mask  = np.zeros(out.shape)
    mask[bounds[1]:bounds[3], bounds[0]:bounds[2], :] = 1
    out = out * mask
    out = out.astype(np.uint8)

    # THIN CNN OUTPUTS
    thin = cv2.ximgproc.thinning(out[:, :, 1], thinningType=cv2.ximgproc.THINNING_GUOHALL)

    # OUTPUT X AND Y
    y, x = np.where(np.asarray(thin > 0))

    out_struct = {
        "x" : x,
        "y" : y,
        "FANN" : FANN,
        "RLNN" : RLNN,
        "thin" : thin,
        "raw"  : out,
        "bbox" : outbbox
    }
    
    return out_struct

def findIntersections(lines):
    intersections = []
    for i in tqdm(range(len(lines))):
        for j in range(i+1, len(lines)):
            line1 = lines[i][0]
            line2 = lines[j][0]

            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2

            # Calculate intersection point
            denominator = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))
            if denominator != 0:
                intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
                intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator

                # Add intersection point to list
                intersections.append((int(intersect_x), int(intersect_y)))
    return intersections

def findInetersections_vectorized(lines):
    intersections = []

    # Convert lines to numpy array for easier manipulation
    lines = lines[:, 0, :]  # Extracting the lines from the unnecessary dimensions

    # Extracting line coordinates
    x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]

    # Reshaping to make calculations easier
    x1, y1, x2, y2 = x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)

    # Calculate differences and determinants
    dx, dy = x2 - x1, y2 - y1
    det = dx * dy[:, np.newaxis] - dy * dx[:, np.newaxis]

    # Check for non-parallel lines
    non_parallel_mask = det != 0

    # Calculate intersection points
    intersect_x = ((x1 * y2 - y1 * x2) * dx - (x1 - x2) * (x1 * dy - y1 * dx)) / det
    intersect_y = ((x1 * y2 - y1 * x2) * dy - (y1 - y2) * (x1 * dy - y1 * dx)) / det

    # Check for valid intersections
    valid_mask = (intersect_x >= 0) & (intersect_y >= 0) & (intersect_x < image.shape[1]) & (intersect_y < image.shape[0])

    # Filter out invalid intersections and non-parallel lines
    valid_intersections = np.column_stack((intersect_x[valid_mask & non_parallel_mask], intersect_y[valid_mask & non_parallel_mask]))

    # Convert intersection points to integer coordinates
    valid_intersections = valid_intersections.astype(int)

    # Remove duplicate intersections
    valid_intersections = np.unique(valid_intersections, axis=0)

    return valid_intersections

def toTF(a):
    a[a > 0] = 1
    a[a <= 0] = 0
    return a

def filterIntersections(inters, lines, fn):
    # APPROACH WITH IMAGE FILTERING. DOESNT WORK
    image_filter = np.asarray(cv2.imread(fn) * 0)
    inter_filter = np.asarray(cv2.imread(fn) * 0)[:, :, 0]

    # Plot the lines
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_filter, (x1, y1), (x2, y2), (255, 255, 255), 2)

    image_filter = image_filter[:, :, 0]
    # CREATE IMAGE MASK
    # Extract x, y coordinates of intersection points
    x_coords, y_coords = zip(*inters)
    # Set the intersection points to white (255) in the mask
    inter_filter[y_coords, x_coords] = 255
    print(inter_filter.shape)
    print(image_filter.shape)


    return toTF(image_filter) * toTF(inter_filter)

def findInetersections_shapely(lines): 
    line_strings = [LineString([(line[0][0], line[0][1]), (line[0][2], line[0][3])]) for line in lines]

    # Create a GeoDataFrame with LineString objects
    gdf = gpd.GeoDataFrame(geometry=line_strings)

    # Calculate intersections
    temp = gdf.unary_union.intersection(gdf.geometry)
    intersection_points = gpd.GeoSeries([point for point in temp if isinstance(point, MultiPoint) or isinstance(point, Point)])
    intersections = gpd.GeoDataFrame(geometry=intersection_points.explode(index_parts=False)).reset_index(drop=True)
    return intersections

fn = tiles[4]
points = getRoadPoints(fn, )

image = np.asarray(cv2.imread(fn))

# Perform Hough Line Transform
lines  = cv2.HoughLinesP(points['thin'], 1, np.pi/720, threshold=20, minLineLength=20, maxLineGap=50)
inters = findInetersections_shapely(lines)

# Plot the lines
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Convert BGR image to RGB for plotting
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)


fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
ax.imshow(image_rgb)
ax.scatter(inters.geometry.x.tolist(), inters.geometry.y.tolist(), marker='x', s=0.5)
fig.savefig("test.png")

'''