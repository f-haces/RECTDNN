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

# TILED INFERENCE
import sahi
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict
sahi.utils.cv.IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.tif']

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


def buildDetectedDatabase(dict):
    # BUILDS DETECTED TILE DATABASE BY READING YOLO RESULTS 
    verbose = False
    dupped_dict = {}
    non_dupped_dict = {}
    db = {}
    possible_tiles = 0

    # FOR EACH TILEINDEX
    for fn in dict.keys():

        # GET AFFINE TRANSFORM FROM CURRENT INDEX
        affine   = Affine(*dict[fn]["output_transform"].flatten()[:6])

        # LIST HOW MANY KEY DUPLICATES WE HAVE FOR EACH INDEX
        db[findKey(fn)] = []
        
        # HOW MANY POSSIBLE TILES DO WE HAVE? ADD ALL FOUND TILES IN INDEX
        possible_tiles = possible_tiles + len(dict[fn]['tile'].keys())


        if verbose:
            print('-' * 10 + fn + '-' * 10 + str(len(dict[fn]['tile'].keys())))

        # FOR EACH TILE IN INDEX
        for i in dict[fn]['tile'].keys():

            # SPLIT NEW LINES AND FILTER THOSE WITH LESS THAN 3 NUMBERS
            text_ori = dict[fn]['tile'][i]['text'].split("\n")
            text = [a for a in text_ori if len("".join(re.findall("\d+", a))) > 3]

            # FIND KEY BASED ON INDEX NAME
            curr_key = findKey(fn)

            # EXTRACT ONE OF THE LIST WITH KEY
            data = process.extractOne(curr_key, text)
            if data is None:
                if verbose:
                    print("NOT FOUND "+ " /n ".join(text_ori))
                continue
            found_text, score = data

            # IF SCORE IS GOOD ENOUGH
            if score > 60:

                # FIND INDEX OF MATCHED TEXT IN LINES 
                idx = text.index(found_text)
                
                # FIND ALL NUMBERS IN MATCHED TEXT
                foundnumbers = "".join(re.findall("\d+", text[idx].replace(" ", "")))
                
                # REMOVE ANY CHARACTERS PRIOR TO PARTIAL MATCH TO KEY
                for a in range(len(curr_key), 1, -1):
                    currentnumbers = foundnumbers.split(curr_key[-a:])[-1]
                    if currentnumbers != foundnumbers:
                        break
                
                # IF WE HAVE EXACTLY 5 CHARACTERS, AND THE LAST IS 8, IT'S LIKELY IT WAS A MISREAD "B"
                if len(currentnumbers) == 5 and (currentnumbers[-1] == "8" or currentnumbers[-1] == "7"):
                    currentnumbers = currentnumbers[:-1]

                if verbose:
                    print(curr_key + currentnumbers + " | " +foundnumbers + " | " +  " /n ".join(text))

                # CALCULATE COORDS FROM AFFINE 
                bbox = dict[fn]['tile'][i]['bbox']
                left, bottom = affine * (bbox[0], bbox[1])
                right, top   = affine * (bbox[2], bbox[3])

                # PREPARE DICT 
                out_dict = dict[fn]['tile'][i]
                out_dict['coords'] = np.array([left, bottom, right, top])

                # SAVE OUTPUT
                dupped_dict[curr_key + currentnumbers] = out_dict
                
                data = non_dupped_dict.get(curr_key + currentnumbers, None)
                if data is None: 
                    non_dupped_dict[curr_key + currentnumbers] = out_dict

                else:
                    i=0
                    while data is not None:
                        data = non_dupped_dict.get(curr_key + currentnumbers + f"_{i}", None)
                        i = i + 1
                    non_dupped_dict[curr_key + currentnumbers + f"_{i}"] = out_dict

                db[findKey(fn)].append(curr_key + currentnumbers)

    return db, non_dupped_dict, dupped_dict

def notify(mess, level=2):
    if level < 4:
        print(mess)

def WorldFilesToDataframe(input_dir, columns=None):
    if not columns:
        columns = ['filename', 'line1', 'line2', 'line3', 'line4', 'line5', 'line6']
    
    # READ ALL THE WORLD FILES FROM A DIRECTORY AND CREATE A DATAFRAME
    data = read_world_files_from_directory(input_dir)
    df = pd.DataFrame(data, columns=columns)

    # CONVERT ALL READ PARAMETERS TO NUMERIC
    for col in [col for col in df.columns if "line" in col]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def getWebmercator(df, input_dir, RLNN = None, verbose=False):
    webmercator = []

    # FOR EACH DATAFRAME
    for i, row in tqdm(df.iterrows(), total=df.shape[0], disable=verbose):

        # GET BASE FILENAME FOR WORLD FILE
        fn = row['filename'].split(".")[0]
        image_files = glob.glob(f'{os.path.join(input_dir, fn)}*[!w]')

        # IF WE DON'T HAVE AN EPSG, CONTINUE
        if pd.isna(row['epsg']):
            notify(f"NO EPSG {fn}")
            webmercator.append([])
            continue
        
        # IF WE CAN'T FIND CORRESPONDING IMAGE
        if len(image_files) == 0:
            notify(f"NO CORRESPONDING IMAGE {fn}")
            webmercator.append([])
            continue

        # FIND THE BOUNDS FOR THAT IMAGE
        bounds, RLNN = findBounds(image_files[0], model=RLNN, verbose=False, device="cuda")

        # IF WE COULDN'T FIND THE IMAGE, THEN NO PROJECTION
        # TODO: ADD IMAGE SIZE
        if len(bounds[0]) == 0:
            notify(f"NO BOUNDS FOUND {image_files[0]}")
            webmercator.append([])
            continue
        
        # GET BOUNDING BOX
        bbox = bounds[0].boxes.xyxy.numpy()[0]

        # REPROJECT BOUNDING BOX
        in_crs  = rio.crs.CRS.from_string(f"{row['epsg']}")
        out_crs = rio.crs.CRS.from_epsg(f"3857")
        left, bottom = row['affine'] * (bbox[0], bbox[1])
        right, top   = row['affine'] * (bbox[2], bbox[3])
        new_bbox = rio.warp.transform_bounds(in_crs, out_crs, left, bottom, right, top)

        # APPEND TO STRUCTURE FOR DATAFRAME
        webmercator.append(new_bbox)

    df["webmerc"] = webmercator
    return df

def buildWorldFileDatabase(input_dir, db, stateplanes):
    df = WorldFilesToDataframe(input_dir)

    # FIND GEOMETRY FOR EACH ROW
    df['key']   = df['filename'].apply(findTileKey, db=db)  # FIND THE KEY FOR THE FILENAME
    df['key_n'] = pd.to_numeric(df['key'], errors='coerce') # CONVERT IT TO NUMERIC
    df["GEOID"] = df["key_n"].apply(getGEOID)               # GET GEOID FOR EACH INDEX
    df['county_polygon'] = df["GEOID"].apply(getGeometry)   # USE FIND GEOMETRY FUNCTION

    # HEURISTICS - DEFINE WHETHER THE FOUND TRANSFORM IS STATEPLANE 
    # BY IT'S SCALE. IF IT'S BIGGER THAN 1, PROBABLY (MOST FILE'S RESOLUTION IS < 0.5 m)
    df['STATEPLANE'] = df['line1'] > 1

    # INTERSECT DF WITH STATE PLANE SHAPEFILE 
    geo_df   = gpd.GeoDataFrame(df, geometry=df['county_polygon']).set_crs("EPSG:3857").to_crs("EPSG:4326")
    df_plane = geo_df.overlay(stateplanes, how='intersection')

    # CALCULATE EPSG CODE
    df['epsg'] = geo_df.apply(getEPSG, df_plane=df_plane, axis=1)
    del geo_df, df_plane

    df['geotransform'] = df['filename'].apply(getGeotransform, input_dir=input_dir)
    df['affine'] = df['geotransform'].apply(getAffine)

    df = getWebmercator(df, input_dir)

    return df

def enlarged_bounds(rasters, n=1):
    """
    Returns an enlarged shapely box that contains the bounds of all input rasters.
    
    Parameters:
    rasters (rasterio.io.DatasetReader or list): The input raster or a list of rasters.
    n (float): The factor by which to enlarge the combined bounds. n=1 means the same size, n=2 means twice as big, etc.
    
    Returns:
    shapely.geometry.polygon.Polygon: The enlarged bounding box.
    """
    # Ensure the input is a list
    if not isinstance(rasters, list):
        rasters = [rasters]
    
    # Combine all raster bounds into a single bounding box
    minx, miny, maxx, maxy = rasters[0].bounds
    for raster in rasters[1:]:
        r_minx, r_miny, r_maxx, r_maxy = raster.bounds
        minx = min(minx, r_minx)
        miny = min(miny, r_miny)
        maxx = max(maxx, r_maxx)
        maxy = max(maxy, r_maxy)

    # Calculate the center of the combined bounding box
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2

    # Calculate the new dimensions
    width = (maxx - minx) * n
    height = (maxy - miny) * n

    # Calculate the new enlarged bounds
    new_minx = center_x - width / 2
    new_miny = center_y - height / 2
    new_maxx = center_x + width / 2
    new_maxy = center_y + height / 2

    # Create a shapely box with the new bounds
    enlarged_box = box(new_minx, new_miny, new_maxx, new_maxy)
    
    return enlarged_box


def plotICP_streets(reprojected_points, initial=None, plot_skip=2, best=None, dpi=200, figsize=(10, 10)):
    # print(initial)
    colors = ['b', 'g']
    icp_iterations = len(reprojected_points)
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    colormap = plt.get_cmap('cool') 

    for i in np.arange(plot_skip, icp_iterations, plot_skip):
        ax.scatter(reprojected_points[i][:, 0], reprojected_points[i][:, 1], 
            color=colormap(i / icp_iterations), s=1, label=f"Iteration {i}")
        
    if initial is not None:
        for i, (k, v) in enumerate(initial.items()):
            ax.scatter(v[:, 0], v[:, 1], label=k, color=colors[i], s=2, marker='x')
        pass
    if best is not None:
        ax.scatter(best[:, 0], best[:, 1], color='red', s=2, marker='x', label="Best Fit")
        
    ax.legend()
    ax.grid()
    ax.axis("equal")
    return fig, ax

def checkConvergence(grades, conv=0.01):
    if len(grades) < 2:
        return False
    return np.abs(grades[-1] - grades[-2]) < conv

def performICPonTile(TLNN, STCN, 
                debug=False, 
                plot=True,
                icp_iterations=30, 
                rotation=True, 
                shear=False, 
                perspective=False,
                save_fig=None,
                conv=0.01,
                rotation_limit=None,
                weights=None
                ):
    
    
    # COORDINATE HANDLING
    # coords_TLNN = np.vstack((TLNN[0, :], TLNN[1, :], np.ones(TLNN[1, :].shape))).T
    # coords_STCN = np.vstack((STCN[0, :], STCN[1, :], np.ones(STCN[1, :].shape))).T

    # MAKE SURE BOTH HAVE COORDINATES
    STCN['x'] = STCN['geometry'].x
    STCN['y'] = STCN['geometry'].y
    TLNN['x'] = TLNN['geometry'].x
    TLNN['y'] = TLNN['geometry'].y

    # GET POINT STRUCTURES
    coords_TLNN = np.array(TLNN[['x', 'y']]) 
    coords_STCN = np.array(STCN[['x', 'y']])
    
    # FAST SEARCH STRUCTURE
    kdtree = cKDTree(coords_STCN)
    
    # ITERATIVE CLOSEST POINT STRUCTURES
    reprojected_points    = []
    compounded_homography = np.eye(3)
    proc_points = coords_TLNN
    
    # OUTPUT STRUCTURES
    transforms, grades = [], []
    initial = {"shp" : coords_STCN, "detected" : coords_TLNN}

    # ITERATE
    for i in tqdm(range(icp_iterations), disable=True):

        # SEARCH CLOSEST POINT IN KDTREE
        _, nearest_indices = kdtree.query(proc_points)
        to_points = np.array([coords_STCN[idx] for idx in nearest_indices])
        
        # TAKE ADJUSTMENT STEP
        new_homography = adjustStep_affine(proc_points, coords_STCN, kdtree,
                                        shear=shear, rotation=rotation, perspective=perspective, rotation_limit=rotation_limit, weights=weights)
        # print(new_homography)
        
        if debug:
            fig, ax = plt.subplots()
            ax.scatter(proc_points[:, 0], proc_points[:, 1])
            ax.scatter(coords_STCN[:, 0], coords_STCN[:, 1])
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
            plt.scatter(coords_STCN[:, 0], coords_STCN[:, 1])
            plt.scatter(to_points[:, 0], to_points[:, 1])
            plt.show()
            
        # COMPOUND TRANSFORMATION
        compounded_homography = compounded_homography @ transform

        # PUT ON OUTPUT STRUCTURES
        transforms.append(compounded_homography)
        grades.append(gradeFit(proc_points, kdtree))
        
        if checkConvergence(grades, conv=conv):
            break

        if i % 1 == 0:
            scale  = np.sqrt((new_homography[0, 0] ** 2 + new_homography[1, 1] ** 2) / 2)
            offset = np.sqrt((new_homography[1, 2] ** 2 + new_homography[0, 2] ** 2) / 2)
            # print(f"Scale: {scale:.2f} Offset: {offset:.2f}")

    # GET BEST TRANSFORMS
    best_transform = transforms[np.argmin(grades)]
    best_points    = reprojected_points[np.argmin(grades)]
    
    if debug:
        plt.plot(range(len(grades)), grades)
        plt.scatter(np.argmin(grades), grades[np.argmin(grades)])
        plt.show()
    
    if plot:
        fig, ax = plotICP_streets(reprojected_points, initial=initial, plot_skip=5, best=best_points)
        if save_fig:
            fig.savefig(save_fig)
        plt.show()
    
    transform_dict = {
        "initial" : initial,
        "reproj"  : reprojected_points,
        "best"    : best_transform,
        "list"    : transforms,
        "grades"  : grades
    }

    return best_transform, transform_dict

def performICPonTile_roads(TLNN, STCN, 
                debug=False, 
                plot=True,
                icp_iterations=30, 
                rotation=True, 
                shear=False, 
                perspective=False,
                save_fig=None,
                ):
    
    
    # COORDINATE HANDLING
    # coords_TLNN = np.vstack((TLNN[0, :], TLNN[1, :], np.ones(TLNN[1, :].shape))).T
    # coords_STCN = np.vstack((STCN[0, :], STCN[1, :], np.ones(STCN[1, :].shape))).T

    # MAKE SURE BOTH HAVE COORDINATES
    STCN['x'] = STCN['geometry'].x
    STCN['y'] = STCN['geometry'].y
    TLNN['x'] = TLNN['geometry'].x
    TLNN['y'] = TLNN['geometry'].y

    # GET POINT STRUCTURES
    coords_TLNN = np.array(TLNN[['x', 'y']]) 
    coords_STCN = np.array(STCN[['x', 'y']])
    
    # FAST SEARCH STRUCTURE
    kdtree = cKDTree(coords_STCN)
    
    # ITERATIVE CLOSEST POINT STRUCTURES
    reprojected_points    = []
    compounded_homography = np.eye(3)
    proc_points = coords_TLNN
    
    # OUTPUT STRUCTURES
    transforms, grades = [], []
    initial = {"shp" : coords_STCN, "detected" : coords_TLNN}

    # ITERATE
    for i in tqdm(range(icp_iterations), disable=True):

        # SEARCH CLOSEST POINT IN KDTREE
        _, nearest_indices = kdtree.query(proc_points)
        to_points = np.array([coords_STCN[idx] for idx in nearest_indices])
        
        # TAKE ADJUSTMENT STEP
        new_homography = adjustStep_affine(proc_points, coords_STCN, kdtree,
                                        shear=shear, rotation=rotation, perspective=perspective)
        # print(new_homography)
        
        if debug:
            fig, ax = plt.subplots()
            ax.scatter(proc_points[:, 0], proc_points[:, 1])
            ax.scatter(coords_STCN[:, 0], coords_STCN[:, 1])
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
            plt.scatter(coords_STCN[:, 0], coords_STCN[:, 1])
            plt.scatter(to_points[:, 0], to_points[:, 1])
            plt.show()
            
        # COMPOUND TRANSFORMATION
        compounded_homography = compounded_homography @ transform

        # PUT ON OUTPUT STRUCTURES
        transforms.append(compounded_homography)
        grades.append(gradeFit(proc_points, kdtree))
        
        if checkConvergence(grades):
            break

        if i % 1 == 0:
            scale  = np.sqrt((new_homography[0, 0] ** 2 + new_homography[1, 1] ** 2) / 2)
            offset = np.sqrt((new_homography[1, 2] ** 2 + new_homography[0, 2] ** 2) / 2)
            # print(f"Scale: {scale:.2f} Offset: {offset:.2f}")

    # GET BEST TRANSFORMS
    best_transform = transforms[np.argmin(grades)]
    best_points    = reprojected_points[np.argmin(grades)]
    
    if debug:
        plt.plot(range(len(grades)), grades)
        plt.scatter(np.argmin(grades), grades[np.argmin(grades)])
        plt.show()
    
    if plot:
        fig, ax = plotICP_streets(reprojected_points, initial=initial, plot_skip=5, best=best_points)
        if save_fig:
            fig.savefig(save_fig)
        plt.show()
    
    transform_dict = {
        "initial" : initial,
        "reproj"  : reprojected_points,
        "best"    : best_transform,
        "list"    : transforms,
        "grades"  : grades
    }

    return best_transform, transform_dict

def calcCenter(a):
    # CALCULATES CENTER OF BOUNDING BOX IN FORM X1 Y1 X2 Y2
    return (a[0] + a[2]) / 2, (a[1] + a[3]) / 2

def pointsToGeodataFrame(ra, y, x):
    # RETURNS A GEODATAFRAME OF POINT COORDINATES GIVEN A SERIES OF PIXEL PAIRS ON A GIVEN RASTER
    xs, ys = rio.transform.xy(ra.transform, y, x)
    xy = list(zip(xs, ys))
    points = MultiPoint(xy)
    detections = gpd.GeoDataFrame(index=[0], geometry=[points]).explode(ignore_index=False, index_parts=False)
    return detections

def draw_bounding_boxes(bboxes, shape):
    # DRAWS BOUNDING BOXES IN IMAGE
    # Create a zero numpy array with the specified shape
    img = np.zeros(shape, dtype=np.uint8)
    
    # Iterate through each bounding box
    for bbox in bboxes:
        x1, y1, x2, y2 = np.int32(bbox)
        # Fill the region specified by the bounding box with 255
        img[y1:y2, x1:x2] = 255
    
    return img

def cleanImageBBOX(image, bbox, rep_value = 0, add=100):
    # SETS EDGES OF IMAGE TO A GIVEN REPLACEMENT VALUE (SIMILAR TO BOMB EDGES)
    image[:bbox[1]+add, :] = rep_value
    image[bbox[3]-add:, :] = rep_value
    image[:, :bbox[0]+add] = rep_value
    image[:, bbox[2]-add:] = rep_value
    return image

def cleanCenterBBOX(coords, bbox, opt_return=[]):
    # RETURNS ONLY BOUNDING BOXES WITHIN A GIVEN, LARGER BOUNDING BOX
    x1, y1, x2, y2 = bbox
    
    # Check which coordinates are within the bounding box limits
    mask = (coords[:, 0] >= x1) & (coords[:, 0] <= x2) & (coords[:, 1] >= y1) & (coords[:, 1] <= y2)
    
    out_l = []
    for a in opt_return:
        out_l.append(a[mask])

    # Filter the coordinates using the mask
    if len(out_l) == 0:
        return coords[mask]
    return coords[mask], *out_l

def getClosestPoints(kdtrees, proc_points, sear_points, weights=None, proc_limit=1000, idx=None, dist_threshold=None):
    assert len(kdtrees) == len(proc_points)
    assert len(proc_points) == len(sear_points)

    if idx is None:
        idx = [[]] * len(kdtrees)

    out_proc_points = []
    out_to_points   = []
    out_weights = []

    if weights is None:
        weights = [1] * len(sear_points)

    for i, kdtree in enumerate(kdtrees):

        if proc_points[i].shape[0] > proc_limit:
            if len(idx[i]) == 0:
                idx[i] = np.random.permutation(proc_points[i].shape[0])[:proc_limit] # np.random.default_rng().choice(proc_points[i].shape[0], size=proc_limit, replace=False)
            proc_points_curr = proc_points[i][idx[i]]
        else:
            proc_points_curr = proc_points[i]

        dist, nearest_indices = kdtree.query(proc_points_curr)

        if dist_threshold is not None:
            proc_points_curr = proc_points_curr[dist < dist_threshold]
            nearest_indices  = nearest_indices[dist < dist_threshold]
        
        if len(nearest_indices) != 0:
            to_points_temp = np.array([sear_points[i][idx] for idx in nearest_indices])
            out_weights.append([weights[i]] * nearest_indices.shape[0]) 
            out_to_points.append(to_points_temp)
            out_proc_points.append(proc_points_curr)

    proc_points = np.vstack(out_proc_points)
    to_points   = np.vstack(out_to_points)
    weights     = np.hstack(out_weights)

    return proc_points, to_points, weights, idx


def adjustStep_affine_weighted(from_points, to_points, 
                      shear=True, rotation = True, perspective=True, rotation_limit=None, weights=None,):

    if shear and rotation:
        transform = affineTransformation(from_points[:, 0], from_points[:, 1], 
                                             to_points[:, 0], to_points[:, 1],
                                             verbose=False, weights=None
                                 )
        
    if not rotation: 
        transform = scalingTranslationTransformation(from_points[:, 0], from_points[:, 1], 
                                             to_points[:, 0], to_points[:, 1],
                                             verbose=False, weights=None
                                 )
    else:
        transform = similarityTransformation(from_points[:, 0], from_points[:, 1], 
                                             to_points[:, 0], to_points[:, 1],
                                             verbose=False, rotation_limit=rotation_limit, weights=None)
    
    new_homography = transform.matrix
    
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

def performWeightedICPonTile(detections, references, 
                debug=False, 
                plot=True,
                icp_iterations=30, 
                rotation=True, 
                shear=False, 
                perspective=False,
                save_fig=None,
                conv=0.01,
                rotation_limit=None,
                weights=None, 
                proc_limit = 10000,
                plot_datasets=[],
                dist_threshold=200
                ):

    if not isinstance(detections, list):
        detections = [detections]

    if not isinstance(references, list):
        references = [references]

    for detection in detections:
        detection['x'] = detection['geometry'].x
        detection['y'] = detection['geometry'].y

    for reference in references:
        reference['x'] = reference['geometry'].x
        reference['y'] = reference['geometry'].y

    # FAST SEARCH STRUCTURE
    sear_points = [np.array(reference[['x', 'y']]) for reference in references]
    proc_points = [np.array(detection[['x', 'y']])  for detection in detections]

    kds = [cKDTree(search) for search in sear_points]    
    
    # ITERATIVE CLOSEST POINT STRUCTURES
    reprojected_points    = []
    compounded_homography = np.eye(3)
    plotting_points = []
    weight_vecs = []
    
    # OUTPUT STRUCTURES
    transforms, grades = [], []
    idx = None    

    # ITERATE
    for i in tqdm(range(icp_iterations), disable=False):
        
        curr_proc_points, to_points, weight_vec, idx = getClosestPoints(kds, proc_points, sear_points, weights=weights, proc_limit=proc_limit, idx=idx, dist_threshold=200)
        

        # TAKE ADJUSTMENT STEP
        new_homography = adjustStep_affine_weighted(curr_proc_points, to_points,
                                        shear=shear, 
                                        rotation=rotation, 
                                        perspective=perspective, 
                                        rotation_limit=rotation_limit, 
                                        weights=weight_vec)        
        
        transform = new_homography.copy()

        def applyTransformWeighted(transform, points):
            return [applyTransform(transform, p) for p in points]

        # APPLY TRANSFORM TO ALL POINTS
        proc_points = applyTransformWeighted(transform, proc_points)

        # COMPOUND TRANSFORMATION
        compounded_homography = compounded_homography @ transform
        
        # APPLY TRANSFORM FROM ADJUSTMENT TO PROCESSING POINTS AND APPEND TO LIST
        curr_proc_points_rep = applyTransform(transform, curr_proc_points)
        reprojected_points.append(curr_proc_points_rep)

        # PUT ON OUTPUT STRUCTURES
        transforms.append(compounded_homography)
        weight_vecs.append(weight_vec)
        
        
        def gradeWeightedFitFullDatasets(kdtrees, proc_points, transform, weights=weights, dist_threshold=None):
            grade = 0
            for i, kdtree in enumerate(kdtrees):
                curr_points = applyTransform(transform, proc_points[i])
                dist, _ = kdtree.query(curr_points)
                if dist_threshold is not None:
                    dist = dist[dist < dist_threshold]
                grade = grade + np.sqrt(np.sum((dist * weights[i]) ** 2) / dist.shape[0])
            return grade
        
        curr_grade = gradeWeightedFitFullDatasets(kds, proc_points, compounded_homography, weights=weights, dist_threshold=dist_threshold)
        grades.append(curr_grade)

        if debug and i % 10 == 0:
            scale  = np.sqrt((new_homography[0, 0] ** 2 + new_homography[1, 1] ** 2) / 2)
            offset = np.sqrt((new_homography[1, 2] ** 2 + new_homography[0, 2] ** 2) / 2)
            print(f"Scale: {scale:.2f} Offset: {offset:.2f} Grades: {curr_grade}")
            print(compounded_homography)

    # GET BEST TRANSFORMS
    best_transform = transforms[np.argmin(grades)]
    best_points    = reprojected_points[np.argmin(grades)]
    
    if debug:
        plt.plot(range(len(grades)), grades)
        plt.scatter(np.argmin(grades), grades[np.argmin(grades)])
        plt.show()
    
    if plot:
        # fig, ax = plotICP_streets(reprojected_points, initial=initial, plot_skip=5, best=best_points)
        fig, ax = plotWeightedICP(reprojected_points, ref_gdfs=plot_datasets, plot_skip=5, best=best_points, weights=weight_vecs)
        if save_fig:
            fig.savefig(save_fig)
        plt.show()
    
    transform_dict = {
        "reproj"  : reprojected_points,
        "best"    : best_transform,
        "list"    : transforms,
        "grades"  : grades
    }

    return best_transform, transform_dict

def plotWeightedICP(reprojected_points, ref_gdfs=[], plot_skip=2, best=None, weights=None, s_base=0.5):
    icp_iterations = len(reprojected_points)
    fig, ax = plt.subplots()
    colormap = plt.get_cmap('cool') 
    
    if weights is None:
        weights = np.ones(reprojected_points.shape[0])

    for gdf in ref_gdfs:
        gdf.plot(ax=ax, markersize=2, marker='x', linewidth=0.5)

    for i in np.arange(plot_skip, icp_iterations, plot_skip):
        ax.scatter(reprojected_points[i][:, 0], reprojected_points[i][:, 1], 
            color=colormap(i / icp_iterations), s=weights[i]*s_base, label=f"Iteration {i}")

    if best is not None:
        ax.scatter(best[:, 0], best[:, 1], color='red', s=1, marker='x', label="Best Fit")
        
    ax.legend()
    ax.grid()
    ax.axis("equal")
    return fig, ax

def getRoadPoints(gdf, distance):
    """
    GENERATED WITH CHATGPT
    Interpolate points along the LineString geometries at every `distance` interval.
    
    Parameters:
    gdf (GeoDataFrame): A GeoDataFrame containing LineString geometries.
    distance (float): The distance interval at which to interpolate points.
    
    Returns:
    GeoDataFrame: A GeoDataFrame with the interpolated Point geometries.
    """
    points = []

    # Efficiently process LineStrings only
    for line in gdf.geometry:
        if line.geom_type == 'LineString':
            num_points = int(line.length // distance) + 1
            distances = [i * distance for i in range(num_points)]
            points.extend([line.interpolate(d) for d in distances])

    # Create a new GeoDataFrame with Point geometries
    points_gdf = gpd.GeoDataFrame(geometry=points, crs=gdf.crs)
    
    return points_gdf

def bboxTransformToCRS(transform, image):
    rev_y_axis = np.array([[1, 0, 0],
                        [0,-1, 0],
                        [0, 0, 1]])
    
    translation = np.eye(3)
    translation[1, 2] = image.shape[0]

    return transform @ translation @ rev_y_axis


def bbox_to_coords_realworld(bbox):
    # BOUNDING BOX 

    x_min, y_min, x_max, y_max = bbox

    xs = [x_min, x_min, x_max, x_max]
    ys = [y_max, y_min, y_min, y_max]
    return xs, ys

def bbox_to_coords_raster(bbox):
    # BOUNDING BOX 

    x_min, y_min, x_max, y_max = bbox

    xs = [x_min, x_min, x_max, x_max]
    ys = [y_min, y_max, y_max, y_min]
    return xs, ys


def processHalfSize(tiles, half_path):
    processing_images=[]
    for tile in tqdm(tiles):
        half_out_fn = os.path.join(half_path, os.path.basename(tile))
        processing_images.append(half_out_fn)
        if not os.path.exists(half_out_fn):
            a    = cv2.imread(tile)
            half = cv2.resize(a,  (0, 0), fx=0.5, fy=0.5)
            cv2.imwrite(half_out_fn, half)

def get_largest_subdirectory(base_dir):
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    numbered_subdirs = [(d, int(d.replace('exp', ''))) for d in subdirs if d.startswith('exp') and d[3:].isdigit()]
    largest_subdir = max(numbered_subdirs, key=lambda x: x[1])[0] if numbered_subdirs else "exp"
    return os.path.join(base_dir, largest_subdir)

def processSAHIresults(yolo_path, streetcorner_out_fn):


    pkldir    = os.path.join(get_largest_subdirectory(yolo_path), "pickles\\")
    print(f"Reading results from: {pkldir}" )

    pkl_files = glob.glob(pkldir + "*")

    streetcorner_dict = {}

    for pkl in pkl_files:
        with open(pkl, 'rb') as f:
            x = pickle.load(f)

        streetcorner_dict[os.path.basename(pkl).split(".")[0]] = np.array([calcCenter(a.bbox.to_xyxy()) for a in x])

    pickle.dump(streetcorner_dict, open(streetcorner_out_fn, "wb" ) )

    return streetcorner_dict
    

def processTiledYOLOs(tiles, model_paths, out_dict_names, proc_dir, imsizes):

    print("Making images half size for tiled inference")
    half_path = os.path.join(proc_dir, "half_size")
    os.makedirs(half_path, exist_ok=True)
    processHalfSize(tiles, half_path)
    
    out_dicts = []

    for i, model_path in enumerate(model_paths): 

        yolo_path = os.path.join(proc_dir, out_dict_names[i])
        os.makedirs(yolo_path, exist_ok=True)

        out_fn = os.path.join(proc_dir, f"{out_dict_names[i]}.pkl")
        
        if not os.path.exists(out_fn):
            
            detection_model = AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model_path=model_path,
                confidence_threshold=0.001,
                device="cuda",  # or 'cuda:0'
            )
            
            result = predict(source=half_path,
                            detection_model=detection_model, 
                            verbose=0, 
                            project=yolo_path,
                            slice_height=imsizes[i], 
                            slice_width=imsizes[i], 
                            model_device="cuda", 
                            return_dict=True, 
                            export_pickle=True,
                            visual_hide_labels=True)

            dict_dir =  os.path.join(proc_dir, out_dict_names[i]+".pkl")
            print(f"Exporting to {dict_dir}")
            curr_dict = processSAHIresults(yolo_path, dict_dir)
        else:
            print(f"Reading from {out_fn}")
            curr_dict = pickle.load(open(out_fn, "rb"))

        out_dicts.append(curr_dict)
    
    return out_dicts
