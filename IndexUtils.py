# NOTEBOOK IMPORTS
import os, glob, warnings
import numpy as np
from tqdm.notebook import tqdm

# IMAGE IMPORTS
import cv2
from PIL import Image

# GIS IMPORTS
import pyproj
from shapely.geometry import MultiPolygon
from shapely.ops import transform
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

# PLOTTING IMPORTS
import matplotlib.pyplot as plt

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
    return True

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
        largest_polygon_area = np.argmax([a.area for a in row["geometry"].geoms])
        largest_polygon = row["geometry"].geoms[largest_polygon_area].simplify(tolerance=20)
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

    # GET CLASSES AND CONFIDENCES OF EACH RESULT
    classes = results[0].boxes.data.numpy()[:, -1]
    conf    = results[0].boxes.data.numpy()[:, -2]

    # GOTTA FIND CORRECT FILE BC RESIZED WERE SAVED WITH PNG EXTENSION
    basen = os.path.basename(results[0].path)[:-4]
    in_fn = glob.glob(os.path.join(input_folder,  basen + '*[!w]'))[0]
    key = findKey(basen)

    image = Image.open(in_fn)
    width, height = image.size
    im_size_arry  = np.array([width, height, width, height])

    # OUTPUT STRUCTURE
    outputs = {}

    # FOR TILES
    slice = np.logical_and(classes==0, conf > 0.92)
    for i in np.where(slice)[0]: 
        
        # GET TILE DATA
        bbox = results[0].boxes.xyxyn.numpy()[i]
        data = extract_bounded_area(image, bbox)

        bbox = bbox * im_size_arry

        # GET ID FROM TILE
        text = pytesseract.image_to_string(data, config='--psm 12 --oem 3') # -c tessedit_char_whitelist=0123456789
        word = find_word_with_key(text, key, threshold=80, verbose=False)

        if isinstance(word, list):
            word = ",".join(word)

        outputs[word] = {"bbox" : bbox, "data" : data} # (bbox, data)

    # FOR COUNTY - GET MOST LIKELY BOX CLASSIFIED AS COUNTY
    county_conf = conf.copy()
    county_conf[classes != 1] = 0
    slice = np.argmax(county_conf)

    bbox = results[0].boxes.xyxyn.numpy()[slice]
    outputs["county"] = {"bbox" : bbox * im_size_arry, "data" : extract_bounded_area(image, bbox)}

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

def find_bbox_yolo(mydict : dict) -> np.array:

    """
    Extracts the maximum Bounding Box of YOLO outputs put through

    Parameters:
        mydict (dict): Nested dict with format {TileName (STR) : {"bbox" : [x_min, y_min, x_max, y_max] ... }}

    Returns:
        bounds (np.array): Numpy array with bounds of all found tiles.
    """

    mylist = []
    for i, (k, v) in enumerate(mydict.items()):
        mylist.append(v["bbox"])

    mylist = np.array(mylist)

    return np.array([np.min(mylist[:, 0]), np.min(mylist[:, 1]), np.max(mylist[:, 2]), np.max(mylist[:, 3])])

def boundsToCV2Rect(bounds):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        bounds: np.array in format [x_min, y_min, x_max, y_max]
        dst: Nxm array of points
    Output:
        rect: np.array in format [c1, c2, c3, c4], where c is an x, y coordinate
    '''
    out = np.array([[bounds[0], bounds[1]], 
                   [bounds[0], bounds[3]],
                   [bounds[2], bounds[1]],
                   [bounds[2], bounds[3]]])
    return out

def translation_matrix(x, y, z=1):
    '''
    Find the translation matrix from a given x and y offseets
    Input:
        x: (float)
        y: (float)
    Output:
        matrix: np.array representing translation matrix
    '''
    matrix = np.array([[1, 0, x],
                        [0, 1, y],
                        [0, 0, z]])
    return matrix

def gradeFit(pts1, kdtree):
    '''
    Predict fit between ICP
    Input:
        x: (float)
        y: (float)
    Output:
        matrix: np.array representing translation matrix
    '''
    dist, _ = kdtree.query(pts1)
    return np.sqrt(np.sum(dist ** 2))

def performICPonIndex(boundaries, dnn_outputs,
               debug=False, plot=True, icp_iterations=30,
               rotation=True, shear=False, perspective=False,
               ):
    '''
    ICP
    Input:
        image_arry (np.array): Image
        bounds_panels: 
        shp_bounds:
        output_image_fn:
        dnn_outputs:
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    # INITIAL APPROXIMATE TRANSFORM BASED ON BOUNDS - FROM AND TO POINT DEFINITIONS BASED ON INPUTS
    from_points = boundsToCV2Rect(boundaries["bounds_panels"])
    to_points   = boundsToCV2Rect(boundaries["shp_bounds"])

    # CALCULATE INITIAL TRANSFORM FROM BOUNDS
    initial_transform = cv2.findHomography(from_points, to_points, cv2.RANSAC, 1000)
    original_homography = initial_transform[0]
    inverse_transform = np.linalg.inv(original_homography)

    # CONVERT THINNED IMAGE TO POINTS
    thin_image = getCountyBoundaryFromImage(dnn_outputs["countyArea"])
    y, x = np.where(thin_image[::-1, :])                   # GET COORDINATES OF EVERY 
    image_points = np.vstack((x, y, np.ones(x.shape)))     # STACK X, Y, AND Z COORDINATES
    
    # TRANSFORM SHAPEFILE POINTS INTO IMAGE COORDINATE SYSTEM
    point_geometry = [[point.geometry.x, point.geometry.y, 1] for i, point in boundaries["point_boundary_gdf"].iterrows()]
    point_geometry = inverse_transform @ np.array(point_geometry).T
    
    # COORDINATE HANDLING
    coords_shp = point_geometry.T
    coords_ras = np.vstack((image_points[0, :], image_points[1, :], np.ones(image_points[1, :].shape))).T
    
    # IMAGE ORIGIN COORDINATE SYSTEM TO IMAGE CENTER COORDINATE SYSTEM
    offsets = np.min(coords_ras, axis=0)
    x_offset, y_offset = offsets[0], offsets[1]
    coords_shp_proc_bl = np.vstack((coords_shp[:, 0] - x_offset, coords_shp[:, 1] - y_offset)).T
    coords_ras_proc_bl = np.vstack((coords_ras[:, 0] - x_offset, coords_ras[:, 1] - y_offset)).T
    initial_points = {"shp" : coords_shp_proc_bl, "ras" : coords_ras_proc_bl}
    
    if debug:
        plt.scatter(coords_shp_proc_bl[:, 0], coords_shp_proc_bl[:, 1])
        plt.scatter(coords_ras_proc_bl[:, 0], coords_ras_proc_bl[:, 1])
        plt.show()
    
    # FAST SEARCH STRUCTURE
    kdtree     = cKDTree(coords_ras_proc_bl)
    
    # ITERATIVE CLOSEST POINT
    reprojected_points = []
    compounded_homography = np.eye(3)
    proc_points = coords_shp_proc_bl
    
    # TRANSFORMATION PARAMS
    # rotation, shear, perspective = True, False, False

    # OUTPUT STRUCTURES
    transforms, grades = [], []

    # ITERATE
    for i in tqdm(range(icp_iterations), disable=True):
        
        _, nearest_indices = kdtree.query(proc_points)
        to_points = np.array([coords_ras_proc_bl[idx] for idx in nearest_indices])
        
        # TAKE ADJUSTMENT STEP
        new_homography = adjustStep_affine(proc_points, coords_ras_proc_bl, kdtree,
                                        shear=shear, rotation=rotation, perspective=perspective)
        
        if debug:
            fig, ax = plt.subplots()
            ax.scatter(proc_points[:, 0], proc_points[:, 1])
            ax.scatter(coords_ras_proc_bl[:, 0], coords_ras_proc_bl[:, 1])
            ax.scatter(to_points[:, 0], to_points[:, 1])

            for i in range(proc_points.shape[0]):
                plt.plot([proc_points[i, 0], to_points[i, 0]],
                         [proc_points[i, 1], to_points[i, 1]], 'ko', linestyle="--")
            plt.show()
        
        transform = new_homography.copy()
        
        # APPLY TRANSFORM FROM ADJUSTMENT TO PROCESSING POINTS AND APPEND TO LIST
        reprojected_points.append(applyTransform(transform, proc_points))

        proc_points = applyTransform(transform, proc_points)
        if debug:
            plt.scatter(proc_points[:, 0], proc_points[:, 1])
            plt.scatter(coords_ras_proc_bl[:, 0], coords_ras_proc_bl[:, 1])
            plt.scatter(to_points[:, 0], to_points[:, 1])
            plt.show()
            
        # COMPOUND TRANSFORMATION
        compounded_homography = compounded_homography @ transform
        
        transforms.append(compounded_homography)
        
        grades.append(gradeFit(proc_points, kdtree))
        
        
        if i % 1 == 0:
            scale  = np.sqrt((new_homography[0, 0] ** 2 + new_homography[1, 1] ** 2) / 2)
            offset = np.sqrt((new_homography[1, 2] ** 2 + new_homography[0, 2] ** 2) / 2)
            # print(f"Scale: {scale:.2f} Offset: {offset:.2f}")
    
    best_transform = transforms[np.argmin(grades)]
    best_points    = reprojected_points[np.argmin(grades)]
    
    if debug:
        plt.plot(range(len(grades)), grades)
        plt.scatter(np.argmin(grades), grades[np.argmin(grades)])
        plt.show()

    if plot:
        plotICP(reprojected_points, initial_points, plot_skip=5, best=best_points)
        plt.show()
    
    transform_dict = {
        "initial" : original_homography,
        "best"    : best_transform 
    }

    return transform_dict

def ICPtoCRSTransform(image_arry, transform_dict):
    # REVERSE Y AXIS
    rev_y_axis = np.array([[1, 0, 0],
                        [0,-1, 0],
                        [0, 0, 1]])

    # move = original_homography @ np.array([0, image_t.shape[0], 0])
    translation = np.eye(3)
    translation[1, 2] = image_arry.shape[0]
    
    adjustment =  np.linalg.inv(transform_dict['best'].copy())
    rev_adj = adjustment.copy()
    rev_adj[1, 1] = rev_adj[1, 1] * -1
    
    output_transform = transform_dict['initial'] @ translation @ rev_adj
    offsets = output_transform @ np.array([[0, 0, 1], [image_arry.shape[0], 0, 1]]).T
    offsets = offsets[:, 1] - offsets[:, 0]

    return output_transform, offsets