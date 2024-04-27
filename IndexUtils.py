# NOTEBOOK IMPORTS
import os, glob, zipfile, warnings
import numpy as np
from tqdm.notebook import tqdm
from shutil import copyfile, rmtree
from datetime import datetime

# IMAGE IMPORTS
import cv2
from PIL import Image

# GIS IMPORTS
import fiona, pyproj
from affine import Affine
from shapely.geometry import shape, mapping, Point, LineString, MultiPolygon, Polygon
from shapely.ops import transform, nearest_points, snap
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask
from scipy.spatial import cKDTree

# PLOTTING IMPORTS
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines

# CUSTOM UTILITIES
from WorldFileUtils import *
from GeometryUtils import *
from icp import *
from DataUtils import *
from FindGrid import *
from PlottingUtils import *
from affinetransformation import *

Image.MAX_IMAGE_PIXELS = 933120000
warnings.filterwarnings("ignore")

def init_databases(ref_dir):
    """
    Initializes global variables for reference databases.

    Reads data from CSV and shapefile files located in the specified reference directory.

    Args:
        ref_dir (str): The directory path containing reference data files.

    Global Variables:
        CIDs (pandas.DataFrame): DataFrame containing CountyCIDs data.
        counties (geopandas.GeoDataFrame): GeoDataFrame containing Counties shapefile data.
        places (geopandas.GeoDataFrame): GeoDataFrame containing Places shapefile data.
    """
    global CIDs, counties, places
    CIDs     = pd.read_csv(f"{ref_dir}CountyCIDs.csv", index_col=0)
    counties = gpd.read_file(f"{ref_dir}Counties.shp")
    places   = gpd.read_file(f"{ref_dir}Places.shp")

    counties["GEOID"] = counties["GEOID"].astype(np.int32)
    places["GEOID"]   = places["GEOID"].astype(np.int32)

def getGEOID(CID,):
    global CIDs
    # DEALING WITH A COMMUNITY ID (CID)
    if CID >= 9e4:
        output = CIDs[CIDs["CID"] == CID]["GEOID_p"].to_numpy()
    else: # DEALING WITH A COUNTY
        output = np.asarray([CID])
    
    if output.size == 0:
        return None
    return output[0]

def getGeometry(geoid, new_epsg=3857):
    
    project = pyproj.Transformer.from_crs(pyproj.CRS('EPSG:4326'), pyproj.CRS(f'EPSG:{new_epsg}'), 
                                          always_xy=True).transform
    
    # DEALING WITH A COMMUNITY ID (CID)
    if geoid >= 9e4:
        output = places[places["GEOID"] == geoid]["geometry"].to_numpy()
    else: # DEALING WITH A COUNTY
        output = counties[counties["GEOID"] == geoid]["geometry"].to_numpy()
    if output.size == 0:
        return None    
    
    output = transform(project, output[0])
    
    return output

def drawGrid(image_t, out):
    # Create a blank image to draw the lines on
    line_image = np.zeros_like(image_t)

    for k, contours in out.items():
        contours = contours.squeeze()
        for i in range(contours.shape[0] - 1):
            start_point = tuple(contours[i, :])
            end_point = tuple(contours[i+1, :])
            color = (255)  # You can change the color (BGR format) as needed
            thickness = 10  # You can adjust the thickness of the line
            line_image = cv2.line(line_image, start_point, end_point, color, thickness)
            
    return line_image > 0

def applyTransform(transform, arry):
    final_points = transform @ np.vstack((arry[:, 0], arry[:, 1], np.ones(arry[:, 0].shape)))
    final_points = final_points[:2, :].T
    return final_points

def adjustStep_cv2(from_points, coords_ras, kdtree, shear=True, rotation = True, perspective=True):
    
    # CALCULATE NEAREST POINTS AND FIND HOMOGRAPHY
    _, nearest_indices = kdtree.query(from_points)
    to_points = np.array([coords_ras[idx] for idx in nearest_indices])
    new_homography, _ = cv2.findHomography(from_points, to_points, cv2.RANSAC, 1000000)
    if not shear:
        scale  = np.sqrt((new_homography[0, 0] ** 2 + new_homography[1, 1] ** 2) / 2)
        new_homography[0, 0] = scale 
        new_homography[1, 1] = scale
    if not perspective:
        new_homography[2, 0] = 0 
        new_homography[2, 1] = 0 
    if not rotation:
        new_homography[0, 1] = 0 
        new_homography[1, 0] = 0 
    
    return new_homography

def adjustStep_affine(from_points, coords_ras, kdtree, 
                      shear=True, rotation = True, perspective=True):
    
    # CALCULATE NEAREST POINTS AND FIND HOMOGRAPHY
    _, nearest_indices = kdtree.query(from_points)
    to_points = np.array([coords_ras[idx] for idx in nearest_indices])
    affine = affineTransformation(from_points[:, 0], from_points[:, 1], 
                                             to_points[:, 0], to_points[:, 1],
                                             verbose=False
                                 )
    
    new_homography = affine.matrix
    
    if not shear:
        scale  = np.sqrt((new_homography[0, 0] ** 2 + new_homography[1, 1] ** 2) / 2)
        new_homography[0, 0] = scale 
        new_homography[1, 1] = scale
    if not perspective:
        new_homography[2, 0] = 0 
        new_homography[2, 1] = 0 
    if not rotation:
        new_homography[0, 1] = 0 
        new_homography[1, 0] = 0 
    
    return new_homography

def find_bbox(binary_image):
    """
    Finds the bounding box coordinates of the foreground object in a binary image.

    Args:
        binary_image (numpy.ndarray): The binary image where foreground pixels are represented as "True".

    Returns:
        numpy.ndarray or None: An array containing the coordinates of the bounding box in the format [x_min, y_min, x_max, y_max]. Returns None if no foreground pixels are found.
    """

    # Find the coordinates of all "True" elements in the binary image
    nonzero_points = cv2.findNonZero(binary_image)

    if nonzero_points is None:
        return None

    # Calculate the bounding rectangle for the "True" elements
    x, y, w, h = cv2.boundingRect(nonzero_points)

    return np.array([x, y, x+w, y+h])

def get_world_file_path(image_path):
    """
    Generates the path to the corresponding world file for a given image file path.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str or None: The path to the world file corresponding to the input image file. Returns None if the file extension is unsupported.
    """

    # Get the file extension (e.g., "png", "jpg", "tif")
    file_extension = image_path.split('.')[-1].lower()

    # Define a dictionary to map file extensions to world file extensions
    extension_mapping = {
        'png': 'pgw',
        'jpg': 'jpw',
        'jpeg': 'jpw',  # You can add more extensions if needed
        'tif': 'tfw',
        'tiff': 'tfw',
    }

    # Check if the file extension is in the mapping
    if file_extension in extension_mapping:
        # Replace the file extension with the corresponding world file extension
        world_file_extension = extension_mapping[file_extension]

        # Create the world file path by replacing the image file extension with the world file extension
        world_file_path = os.path.splitext(image_path)[0] + '.' + world_file_extension

        return world_file_path
    else:
        return None  # Unsupported file extension

def getBoundaryPoints(row, distance=20):
    """
    Interpolates points along the boundary of the largest polygon within the given geopandas DataFrame row.

    Args:
        row (pandas.Series): A DataFrame row containing a 'geometry' key with polygon(s). If no geometry is found, returns None.
        distance (float, optional): Interpolation distance for generating points along the polygon boundary, in the row's Coordinate Reference System (CRS). Default is 20.

    Returns:
        point_boundary_gdf (geopandas.GeoDataFrame): A GeoDataFrame containing the interpolated points along the polygon boundary.
        shp_bounds (tuple): Bounds of the largest shape within the row's geometry, formatted as (minx, miny, maxx, maxy).
    """

    # GET WHICHEVER POLYGON IS THE LARGEST IN THE ROW'S GEOMETRY AND SIMPLIFY
    if row["geometry"] is None:
        return None
    elif isinstance(row["geometry"], MultiPolygon):
        largest_polygon_area = np.argmax([a.area for a in row["geometry"]])
        largest_polygon = row["geometry"][largest_polygon_area].simplify(tolerance=20)
        largest_polygon = largest_polygon.boundary
    else:
        largest_polygon = row["geometry"].boundary
    
    # CONVERT POLYGON TO POINTS
    length = largest_polygon.length # POLYGON LENGTH
    point_boundary_list = list()    # OUTPUT POINT LIST
    
    # INTERPOLATE THROUGH ALL OF LENGTH
    for distance in tqdm(range(0,int(length), distance), disable=True):         
        point = largest_polygon.interpolate(distance)   
        point_boundary_list.append(point)
    point_boundary_gdf = gpd.GeoDataFrame(geometry=point_boundary_list)
    
    # SHAPEFILE BOUNDS
    shp_bounds = [i for i in largest_polygon.bounds]
    return point_boundary_gdf, shp_bounds

def getCountyBoundaryFromImage(countyArea):
    countyArea = cv2.Canny(countyArea, 50, 100)
    # y, x = np.where(np.asarray(countyArea > 0))
    return countyArea# np.vstack((x, y)).T

def extract_bounded_area(image, bbox):
    """
    Extracts the bounded area from the image defined by the bounding box.

    Parameters:
        image_path (str): Path to the image file.
        bbox (tuple): Bounding box coordinates in the format (x, y, width, height),
                      normalized by the total image width and height.

    Returns:
        cropped_image (PIL.Image): Cropped region of the image.
    """
    # image = Image.fromarray(image)

    # Get image width and height
    image_width, image_height = image.size
    
    # Convert normalized bounding box to absolute coordinates
    x, y, width, height = bbox
    x_abs = int(x * image_width)
    y_abs = int(y * image_height)
    width_abs = int(width * image_width)
    height_abs = int(height * image_height)
    
    # Define bounding box region
    bbox_region = (x_abs, y_abs, width_abs, height_abs)

    # Crop the image
    cropped_image = image.crop(bbox_region)
    
    return cropped_image

def findTiles(image_fn, model=None, 
        model_weights=f"{data_dir}BBNN/curr_weights.pt",
        creation_params=None,
        device="cpu",
        ):
        
    input_folder = os.path.dirname(os.path.abspath(image_fn))

    if creation_params is None:
        target_size = 1024
        original_shapes = []

        # COCO DATASET PARAMS
        category_labels = {
            0 : "County",
            1 : "Tile",
            2 : "Box",
            3 : "Legend"
        }

        categories=[0, 1]
    
    # Initialize model
    if model is None:
        model = ultralytics.YOLO(model_weights).to("cpu")

    model = model.to(device)
    
    results = model(image_fn, imgsz=target_size)

    classes = results[0].boxes.data.numpy()[:, -1]
    conf    = results[0].boxes.data.numpy()[:, -2]

    slice = np.logical_and(classes==0, conf > 0.92)

    # GOTTA FIND CORRECT FILE BC RESIZED WERE SAVED WITH PNG EXTENSION
    basen = os.path.basename(results[0].path)[:-4]
    in_fn = glob.glob(os.path.join(input_folder,  basen + '*'))[0]
    key = findKey(basen)

    image = Image.open(in_fn)

    outputs = {}

    for i in np.where(slice)[0]: 
        
        # GET TILE DATA
        bbox = results[0].boxes.xyxyn.numpy()[i]
        data = extract_bounded_area(image, bbox)

        # GET ID FROM TILE
        text = pytesseract.image_to_string(data, config='--psm 12 --oem 3') # -c tessedit_char_whitelist=0123456789
        word = find_word_with_key(text, key, threshold=80, verbose=False)

        outputs[word] = {"bbox" : bbox, "data" : data} # (bbox, data)

    return outputs, model

def findSquares(image, model=None, 
        model_checkpoint=f"{data_dir}RLNN/checkpoint_091323.pth",
        cnn_creation_params=None,
        device="cuda",
        ):
        
    if cnn_creation_params is None:
        cnn_creation_params = {
            "finalpadding" : 1,
            "num_classes"    : 3
        }
    
    # Initialize model
    if model is None:
        model = RLNN(**cnn_creation_params)
        model.load_state_dict(torch.load(model_checkpoint)['model_state_dict'])
    model = model.to(device)

    tensor = transforms.Compose([transforms.ToTensor()])
    
    # INPUT IMAGE AND PREP
    shape = image.shape
    image = cv2.resize(image, (512, 512))   
    image_prep = tensor(image).unsqueeze(0).to(device)

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
    
def findCounty(image, model=None, 
        model_checkpoint=f"{data_dir}CLNN/checkpoint_101423.pth",
        cnn_creation_params=None,
        device="cuda",
        ):
        
    if cnn_creation_params is None:
        cnn_creation_params = {
            "finalpadding" : 1,
            "num_classes"  : 2
        }
    
    # Initialize model
    if model is None:
        model = RLNN(**cnn_creation_params)
        model.load_state_dict(torch.load(model_checkpoint)['model_state_dict'])
    model = model.to(device)

    tensor = transforms.Compose([transforms.ToTensor()])
    
    # INPUT IMAGE AND PREP
    shape = image.shape
    image = cv2.resize(image, (1024, 1024))   
    image_prep = tensor(image).unsqueeze(0).to(device)

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
    
    # outputs = cv2.Canny(outputs, 50,100)
    
    return outputs, model

def findKeypoints(image, model=None, num_classes=5, num_pyramids=3,
                cnn_run_params=None, cnn_creation_params=None, device="cuda",
                model_checkpoint=f"{data_dir}TPNN/checkpoint_091523_pyramids_2.pth"):
    
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
