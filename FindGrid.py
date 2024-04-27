# PYTHON IMPORTS
import os, math, re
from tqdm.notebook import tqdm
from fuzzywuzzy import fuzz

# IMAGE IMPORTS 
from PIL import Image, ImageDraw
import cv2

# DATA IMPORTS 
import random
import numpy as np

# PLOTTING
import matplotlib.pyplot as plt

# NEURAL NETWORK
import torch
import ultralytics

# SHAPES IMPORTS
import geopandas as gpd
from shapely.ops import unary_union, split
from shapely.geometry import LineString, Polygon, Point

# OCR libraries
import pytesseract
from fuzzywuzzy import fuzz
import re


# MY OWN CLASSES
from TPNN import *
from RLNN import *
from DataUtils import *

def initialize():
    if os.getlogin() == 'fhacesga':
        t_path   = r'C:\Users\fhacesga\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
        data_dir = r"C:\Users\fhacesga\OneDrive - University Of Houston\AAA_RECTDNN\data/"
    else:
        t_path   = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        data_dir = r"C:\Users\franc\OneDrive - University Of Houston\AAA_RECTDNN\data/"
    
    pytesseract.pytesseract.tesseract_cmd = t_path
    
    return data_dir
    
global data_dir
data_dir = initialize() 

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

def plotLines(original_image, lines, fig=None, mask=None, color=(0, 0, 255), fig_size=(15, 15), savedir=None):
    if fig is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        ax = fig.axes
        
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
    
    if savedir is not None:
        fig.savefig(savedir)

    return fig

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

def draw_lines_to_image(lines, image_size, line_color=(255), background_color=(0)):
    """
    Draw lines on an image using ImageDraw.

    Args:
        lines (list): List of lines, where each line is defined as (x0, y0, x1, y1).
        image_size (tuple): Size of the output image (width, height).
        line_color (tuple): Color of the lines (R, G, B).
        background_color (tuple): Background color of the image (R, G, B).

    Returns:
        Image: PIL Image object with the drawn lines.
    """
    # Create a new image with the specified background color
    image = Image.new("L", image_size, background_color)
    
    # Initialize ImageDraw object
    draw = ImageDraw.Draw(image)
    
    # Draw each line on the image
    for x0, y0, x1, y1 in lines:
        draw.line((x0, y0, x1, y1), fill=line_color)
    
    return np.asarray(image)
    
    
def get_overlapping_lines(lines_or, image_or, threshold, 
                          processing_size=2400, 
                          verbose=True,
                          dilate_size=3):
    
    # GET RESCALING FACTORS AND PROCESSING SIZES 
    scale = processing_size / np.max(list(image_or.shape))
    scale_x, scale_y = scale, scale
    processing_size = (int(image_or.shape[1] * scale), int(image_or.shape[0] * scale))

    # DILATE TO IMPROVE DETECTABILITY
    image = cv2.dilate(np.array(image_or).astype(np.uint8), np.ones((3,3), np.uint8), iterations=dilate_size)
    image = cv2.resize(image.astype(np.uint8), processing_size)
    cv2.imwrite("tempfiles/test.png", image) # SAVE TO EXAMINE
    
    # RESCALE LINES TO PROCESSING SIZE
    lines = rescale_lines(lines_or, scale_x, scale_y)
    
    # CREATE EMPTY IMAGE TO MARK OVERLAPPING REGIONS
    overlap_image = np.full(image.shape, 0)
    overlap_image_fr = Image.fromarray(overlap_image.copy().astype(np.uint8))
    
    # FOR EACH LINE
    overlapping_lines = []
    overlapping_values = []
    for i, line in tqdm(enumerate(lines), total=len(lines), disable=~verbose):
                
        # UNPACK LINE AND DRAW IT (A BIT THICK) ON A COPY OF EMPTY IMAGE
        x1, y1, x2, y2 = [int(x) for x in line]
        line_image = overlap_image_fr.copy()
        line_draw = ImageDraw.Draw(line_image)
        line_draw.line((x1, y1, x2, y2), fill=1, width=5)
        
        # STRUCTURE TO SHOW HOW MANY PIXELS THE LINE WOULD ENCOMPASS NORMALLY
        line_image_thresh = overlap_image_fr.copy()
        line_draw = ImageDraw.Draw(line_image_thresh)
        line_draw.line((x1, y1, x2, y2), fill=1, width=1)
        
        # OVERLAP ANALYSIS: COUNT OVERLAPPING PIXELS 
        overlap = np.logical_and(image, line_image)
        overlap_count = np.count_nonzero(overlap)
        
        # CALCULATE THRESHOLD ON GIVEN LINE
        threshold_pixels = np.count_nonzero(line_image_thresh) * threshold
        
        # ONLY EXPORT IF ANY PIXELS ARE FOUND AND THRESHOLD IS MET 
        if overlap_count >= threshold_pixels and threshold_pixels != 0:
            overlapping_lines.append(lines_or[i])
            
        # EXPORT VALUES OF LINES
        overlapping_values.append(overlap_count / threshold_pixels)
        
    return overlapping_lines, overlapping_values


def simplifyShapelyPolygon(polygon, tolerance = .1):
    """ Simplify a polygon with shapely.
    Polygon: ndarray
        ndarray of the polygon positions of N points with the shape (N,2)
    tolerance: float
        the tolerance
    """
    poly = shapely.geometry.Polygon(i)
    poly_s = poly.simplify(tolerance=tolerance)
    # convert it back to numpy
    return np.array(poly_s.boundary.coords[:])
    
    
def create_shapefile_from_dict(data_dict, output_shapefile):
    # Create a list to store the features
    features = []
    
    # Iterate through the dictionary and convert polygons to GeoJSON-like features
    for name, polygon in data_dict.items():
        feature = {
            "type": "Feature",
            "properties": {"name": name},
            "geometry": polygon.__geo_interface__
        }
        features.append(feature)
    
    # Create a GeoDataFrame from the features
    gdf = gpd.GeoDataFrame.from_features(features)
    
    # Save the GeoDataFrame as a shapefile
    gdf.to_file(output_shapefile)
    
    
def find_word_with_key(text, key, threshold=80, verbose=True):
    if verbose:
        print("____________________________________") 
    words = re.findall(r'\b\w+\b', text)  # Extract words from the text
    if verbose:
        print(words)
    
    best_match = None
    similarities = []
    for word in words:
        # FILTER OUT SINGLE NUMBER MATCHES (ie 4 FOR 248201)
        if len(word) < len(key):
            similarities.append(0)
            continue
            
        # GET SIMILARITY RATIO
        similarity = fuzz.partial_ratio(key.lower(), word.lower())
        similarities.append(similarity)
        
    similarities = np.array(similarities)
    if similarities.size == 0:
        return None

    # Find the maximum value in the array
    max_value = np.max(similarities)
    
    if max_value < threshold:
        return None
    
    # Find the indices where the maximum value occurs
    max_indices = np.where(similarities == max_value)[0]
        
    matches = []
    
    for idx in max_indices:
        if len(words[idx]) == len(key) and idx+1 != len(words):
            matches.append(f"{words[idx]+words[idx+1]}")
        else:
            matches.append(f"{words[idx]}")
    
    if len(matches) == 1:
        return matches[0]
        
    else:
        return matches
        

def pad_image_with_percentage(image, width_padding_percent, height_padding_percent):
    original_height, original_width = image.shape[:2]
    
    width_padding = int(original_width * (width_padding_percent / 100))
    height_padding = int(original_height * (height_padding_percent / 100))
    
    padded_image = cv2.copyMakeBorder(image, height_padding, height_padding, width_padding, width_padding, cv2.BORDER_CONSTANT, value=255)
    
    return padded_image
    
def line_detection(classifications, effectiveArea, image_or,
                   target_dim=(2400, 2400),   # PROCESSING RESOLUTION
                   threshold=10/255,          # INITIAL THRESHOLDING BARRIER
                   degree_resolution=25,      # HOUGH LINES TRANSFORM - HOW MANY SUBDIVISIONS TO A DEGREE
                   line_length=50,            # HOUGH LINES TRANSFORM - HOW MANY LINES 
                   extend_percent=15,         # EXTEND PERCENT FOR LINES POST RECOGNITION
                   certainty=.90
                  ):
    # RESIZE INPUTS TO TARGET RESOLUTION011201
    effectiveArea_resized = cv2.resize(effectiveArea[:, :, 1], target_dim)
    resized_image = cv2.resize(classifications, target_dim, interpolation=cv2.INTER_LINEAR)
    
    # THRESHOLD IMAGES
    gray = (resized_image[:, :, 4] > threshold) * 255
    gray = gray.astype(np.uint8)
    
    # CONSIDER ADDING SOME LOGIC BY WHICH, INSTEAD OF ONLY USING THE LINE CLASSIFICATIONS
    # WE MIX BOTH AFTER HOUGH LINES P (SO WE USE THE LINES FROM JUST THE ROADS TO FILTER OUT THE COMBINED)
    
    # LINE THINNING AND RETHRESHOLDING
    gray = cv2.ximgproc.thinning(gray, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    # cv2.imwrite('test.png', gray)
    # gray = np.where(effectiveArea_resized > 0, gray, 0)
    
    # HOUGH TRANSFORMS
    lines = cv2.HoughLinesP(gray, rho=1, theta=np.pi/ (180 * degree_resolution), 
                            threshold=100, 
                            minLineLength= line_length * 2, 
                            maxLineGap= line_length // 2)
                            
    # print(lines)
    # UNPACK LINES FROM ADDITIONAL DIMENSION THAT CV2 RETURNS THEM FROM
    lines = [line[0] for line in lines]

    # DRAW LINES ON IMAGE
    image_with_lines = np.zeros((resized_image.shape[0], resized_image.shape[1], 3))

    for line in lines:
        x1, y1, x2, y2 = line

        colors = np.random.randint(255, size=3).astype(np.int32)
        color = (int(colors[0]), int(colors[1]), int(colors[2])) 
        cv2.line(image_with_lines, (x1, y1), (x2, y2), color, 2)

    # RESCALE LINE IMAGE TO ORIGINAL DIMENSION
    result_image = cv2.resize(image_with_lines, image_or.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
    
    # PERCENTUAL LINE EXTENSION
    extended_lines = extend_lines(lines, extend_percent)
    
    # RESCALE LINES TO ORIGINAL DIMENSION
    image_height, image_width, _ = resized_image.shape
    scale_x, scale_y = calculate_scale_factors_large(effectiveArea.shape, target_dim)
    rescaled_lines   = rescale_lines(lines, scale_x, scale_y)
    
    thinimage = cv2.resize(gray, (effectiveArea.shape[1], effectiveArea.shape[0]))
    # cv2.imwrite("test0.png", thinimage)
    
    return rescaled_lines, result_image, scale_x, scale_y, thinimage

def writeImage(image_path, image, verbose):
    if image.ndim == 3:
        if image.shape[2] == 2:
            image = np.dstack((image[:, :, 0], image[:, :, 1], np.zeros(image[:, :, 0].shape)))
    if verbose:
        cv2.imwrite(image_path, image)
        
def findKey(input_string):
    match = re.match(r'\d+', input_string)
    if match:
        return match.group()
    else:
        return None
        
def contours_to_shapely_polygons(contours, simplify_tolerance=10):
    contours = contours[:, 0, :]
    points = [Point(point[0], point[1]) for point in contours]
    return Polygon(points).simplify(tolerance=simplify_tolerance)

def FindGrid_analog(image, classifications, effectiveArea, key, image_path, verbose=True):    
    
    # DETECT LINES
    lines, result_image, scale_x, scale_y, thinimage = line_detection(classifications, 
                                                                      effectiveArea, 
                                                                      image, 
                                                                      line_length=50)
    
    writeImage(f"tempfiles/{filename}_01_linedetection.png", result_image, verbose)
    writeImage(f"tempfiles/{filename}_01_thinimage.png", thinimage, verbose)

    # FILTER BY MOST POPULAR ANGLES
    angles = calcAngles(lines)
    line_angles, line_indices, sorted_idx = filterLines_MostPopularAngles(np.array(angles), 0.5)
    
    # GET RESCALED LINES
    rescaled_lines_ordered = np.array(lines)[sorted_idx]
    filtered_lines = rescaled_lines_ordered[np.concatenate(line_indices).flatten()]
    
    if verbose:
        fig = plotLines(image, filtered_lines, savedir=f"tempfiles/{filename}_02_azimuthfiltering.png")
        plt.close(fig)

    # EXTEND LINES TO EDGES AND FILTER LINES BY DISTANCE OF EDGEPOINTS
    extended_lines = extend_lines_to_edges(filtered_lines, image.shape)
    
    if verbose:
        fig = plotLines(image, extended_lines, savedir=f"tempfiles/{filename}_03_lineextension.png")
        plt.close(fig)
    min_distance = 50 * np.sqrt(scale_x ** 2 + scale_y ** 2)
    filtered_lines_extended, filtered_idx = filter_lines_by_distance(extended_lines, min_distance)

    # EXTEND LINES TO ENSURE CROSSING
    extended_lines = extend_lines_to_edges(filtered_lines_extended, image.shape)
    # extended_lines = extend_lines(filtered_lines[filtered_idx], 200)
    if verbose:
        fig = plotLines(image, extended_lines, savedir=f"tempfiles/{filename}_04_distancefiltering.png")
        plt.close(fig)

    # CLIP LINES BY EACH INTERSECTION
    lines_shp   = lines_to_linestrings(filtered_lines_extended)
    split_lines = linestrings_to_lines(unary_union(lines_shp))
    if verbose:
        fig = plotLines(image, extended_lines, savedir=f"tempfiles/{filename}_05_lineclipping.png")
        plt.close(fig)

    # WHICH LINES HAVE SIGNIFICANT OVERLAP WITH DARK PIXELS IN IMAGE?
    overlapping_lines, overlap_values = get_overlapping_lines(split_lines, 
                                                             thinimage, 
                                                             0.75,
                                                             verbose=verbose,
                                                             dilate_size=3)
    
    print(overlap_values)

    if verbose:
        fig = plotLines(image, overlapping_lines, savedir=f"tempfiles/{filename}_06_overlappinglines.png")
        plt.close(fig)

    # EXTEND OVERLAPPING LINES
    if verbose:
        fig = plotLines(image, overlapping_lines, savedir=f"tempfiles/{filename}_06_overlappinglines_extended.png")
        plt.close(fig)

    # DRAW LINES IN IMAGE TO IDENIFY CONTOURS
    bw_bounds = draw_lines_to_image(overlapping_lines, (image.shape[1], image.shape[0]))
    writeImage(f"tempfiles/{filename}_06_drawnimage.png", bw_bounds, verbose)

    # FIND CONTOURS
    contours, hierarchy = cv2.findContours(bw_bounds, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    # LOOP THROUGH HIGHEST HERARCHY
    highest_level = np.max(hierarchy, axis=1).flatten()[3]
    print(f"Highest Hierarchy: {highest_level} in {hierarchy.shape[1]} contours")
    
    # OUTPUT WHICH SQUARES WERE RECOGNIZED
    if verbose:
        filled_image = np.zeros(image.shape)
        # FILL CONTOURS WITH RANDOM COLORS
        for idx, contour in enumerate(contours):
            if hierarchy[0][idx][3] == highest_level:  # ONLY CONTOURS WITH NO INNER CONTOURS
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.drawContours(filled_image, [contour], -1, color, thickness=cv2.FILLED)
        writeImage(f"tempfiles/{filename}_06_recognizedsquares.png", filled_image, verbose)

    # FIND WHICH CONTOURS CONTAIN WORDS
    outdict = {}
    for idx, contour in tqdm(enumerate(contours), total=len(contours), disable=~verbose):
        if hierarchy[0][idx][3] == highest_level:  # ONLY CONTOURS WITH NO INNER CONTOURS
            x, y, w, h = cv2.boundingRect(contour)
        
            # CROP THE IMAGE TO JUST CONTOUR
            if image.ndim == 3:
                cropped_region = image[y:y+h, x:x+w, 0]
            else: 
                cropped_region = image[y:y+h, x:x+w]
            
            # PAD WITH PERCENTANGE
            cropped_region = pad_image_with_percentage(cropped_region, 20, 20)
            
            # WRITE IMAGE
            writeImage(f"tempfiles/{filename}_07_{idx}.png", cropped_region, verbose)
            
            # Perform OCR using pytesseract
            ocr_text = pytesseract.image_to_string(cropped_region,
                                                  config='--psm 12 --oem 3')
                                                  # -c tessedit_char_whitelist=0123456789
            
            # IF WE CAN'T FIND ANYTHING, CONTINUE
            if len(ocr_text) == 0:
                continue
                
            # CHECK IF THE KEY IS FOUND IN THE TILE, IF NOT CONTINUE
            text = find_word_with_key(ocr_text, key, verbose=verbose)
            if text is None:
                continue
            
            # IF WE FOUND TOO MANY
            if isinstance(text, list):
                print("Found too many names! Splitting along longest sides")
                try:
                    shapely_contour = contours_to_shapely_polygons(contour)
                    outpoly_1, outpoly_2 = splitPolygonByLongerSides(shapely_contour)
                    
                    outdict[text[0]] = convertShapelyToCV2(outpoly_1)
                    outdict[text[1]] = convertShapelyToCV2(outpoly_2)
                except:
                    print("Failure! Results will be inaccurate due to line segment on Tile Boundary not being identified")
                    continue
                continue
            outdict[text] = contour
    
    plt.clf()
    
    return outdict

def FindGrid_OLD(classifications, effectiveArea, key, image_path, verbose=True):

    # Get largest section of mask image
    image_or = cv2.imread(image_path)
    image = cv2.cvtColor(image_or, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (512, 512)) 

    a, b, c = identifyBiggestContour(effectiveArea[:, :, 1])
    image_mask = cv2.drawContours(a[:, :, 0] * 0, contours=[c],contourIdx=-1, 
                                  color=(255), thickness=cv2.FILLED)
    

    # Detect lines
    lines, result_image, scale_x, scale_y, thinimage = line_detection(classifications, effectiveArea, image)
    writeImage(f"tempfiles/{filename}_01_linedetection.png", result_image, verbose)
    writeImage(f"tempfiles/{filename}_01_thinimage.png", thinimage, verbose)

    # FILTER BY MOST POPULAR ANGLES
    angles = calcAngles(lines)
    line_angles, line_indices, sorted_idx = filterLines_MostPopularAngles(np.array(angles), 0.5)
    
    # GET RESCALED LINES
    rescaled_lines_ordered = np.array(lines)[sorted_idx]
    filtered_lines = rescaled_lines_ordered[np.concatenate(line_indices).flatten()]
    
    if verbose:
        plotLines(image_or, filtered_lines, savedir=f"tempfiles/{filename}_02_azimuthfiltering.png")

    # EXTEND LINES TO EDGES AND FILTER LINES BY DISTANCE OF EDGEPOINTS
    extended_lines = extend_lines_to_edges(filtered_lines, image_or.shape)
    if verbose:
        plotLines(image_or, extended_lines, savedir=f"tempfiles/{filename}_03_lineextension.png")
    min_distance = 50 * np.sqrt(scale_x ** 2 + scale_y ** 2)
    filtered_lines, filtered_idx = filter_lines_by_distance(extended_lines, min_distance)

    # EXTEND LINES TO ENSURE CROSSING
    # extended_lines = extend_lines_to_edges(filtered_lines, image_or.shape)
    extended_lines = extend_lines(filtered_lines, 50)
    if verbose:
        plotLines(image_or, extended_lines, savedir=f"tempfiles/{filename}_04_distancefiltering.png")

    # CLIP LINES BY EACH INTERSECTION
    lines_shp   = lines_to_linestrings(filtered_lines)
    split_lines = linestrings_to_lines(unary_union(lines_shp))
    if verbose:
        plotLines(image_or, extended_lines, savedir=f"tempfiles/{filename}_05_lineclipping.png")

    # ONLY KEEP OVERLAPPING LINES
    overlapping_lines, overlap_values = get_overlapping_lines(split_lines, 
                                                             thinimage, 
                                                             0.8,
                                                             verbose=verbose)

    if verbose:
        plotLines(image_or, overlapping_lines, savedir=f"tempfiles/{filename}_06_overlappinglines.png")
    
    # Convert lines to an image in which we identify contours
    bw_bounds = draw_lines_to_image(overlapping_lines, (image_or.shape[1], image_or.shape[0]))
    contours, hierarchy = cv2.findContours(bw_bounds, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    writeImage(f"tempfiles/{filename}_06_drawnimage.png", bw_bounds, verbose)
    highest_level = np.max(hierarchy, axis=1).flatten()[3]
    print(f"Highest Hierarchy: {highest_level} in {hierarchy.shape[1]} contours")
    
    # Test which squares are identified
    if verbose:
        filled_image = np.zeros(image_or.shape)
        
        # Fill innermost contours with random colors
        for idx, contour in enumerate(contours):
            
            if hierarchy[0][idx][3] == highest_level:  # If contour has no child contours
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.drawContours(filled_image, [contour], -1, color, thickness=cv2.FILLED)
        writeImage(f"tempfiles/{filename}_06_recognizedsquares.png", filled_image, verbose)


    # Find which squares have a given text
    outdict = {}

    

    for idx, contour in tqdm(enumerate(contours), total=len(contours), disable=~verbose):
        if hierarchy[0][idx][3] == highest_level:  # If contour has no child contours
            x, y, w, h = cv2.boundingRect(contour)
        
            # Crop the contour region from the image
            cropped_region = image_or[y:y+h, x:x+w, 0]
            cropped_region = pad_image_with_percentage(cropped_region, 20, 20)
            
            writeImage(f"tempfiles/{filename}_07_{idx}.png", cropped_region, verbose)
            
            # Perform OCR using pytesseract
            ocr_text = pytesseract.image_to_string(cropped_region,
                                                  config='--psm 12 --oem 3')
                                                  # -c tessedit_char_whitelist=0123456789
            
            if len(ocr_text) == 0:
                continue
                
            text = find_word_with_key(ocr_text, key, verbose=verbose)
            
            if text is None:
                continue
            
            if isinstance(text, list):
                print("Found too many names! Splitting along longest sides")
                try:
                    shapely_contour = contours_to_shapely_polygons(contour)
                    outpoly_1, outpoly_2 = splitPolygonByLongerSides(shapely_contour)
                    
                    outdict[text[0]] = convertShapelyToCV2(outpoly_1)
                    outdict[text[1]] = convertShapelyToCV2(outpoly_2)
                except:
                    print("Failure! Results will be inaccurate due to line segment on Tile Boundary not being identified")
                    continue
                continue
            outdict[text] = contour
    
    plt.clf()
    
    return outdict
    
def get_polygon_side_lengths(polygon):
    # Get the coordinates of the polygon's vertices
    vertices = polygon.exterior.coords
    
    # Calculate the lengths of the sides
    side_lengths = []
    for i in range(len(vertices)-1):
        start_point = Point(vertices[i])
        end_point = Point(vertices[i+1])
        side_lengths.append(start_point.distance(end_point))
    
    # Add the length of the closing side (from last to first vertex)
    start_point = Point(vertices[-1])
    end_point = Point(vertices[0])
    side_lengths.append(start_point.distance(end_point))
    
    return side_lengths

def get_polygon_longest_side_midpoints(polygon, N):
    # Get the coordinates of the polygon's vertices
    vertices = polygon.exterior.coords
    
    # Calculate the lengths of the sides and their corresponding indices
    side_lengths = [(i, Point(vertices[i]).distance(Point(vertices[i+1])))
                    for i in range(len(vertices)-1)]
    
    # Add the length of the closing side (from last to first vertex)
    side_lengths.append((len(vertices)-1,
                         Point(vertices[-1]).distance(Point(vertices[0]))))
    
    # Sort the sides by length in descending order
    sorted_sides = sorted(side_lengths, key=lambda x: x[1], reverse=True)
    
    # Get the indices of the N longest sides
    longest_side_indices = [index for index, _ in sorted_sides[:N]]
    
    # Calculate the midpoints of the longest sides
    midpoints = []
    for index in longest_side_indices:
        x = (vertices[index][0] + vertices[index+1][0]) / 2
        y = (vertices[index][1] + vertices[index+1][1]) / 2
        midpoints.append((x, y))
    
    midpoints = extend_lines([[midpoints[0][0], midpoints[0][1], midpoints[1][0], midpoints[1][1]]], 500)
    midpoints = midpoints[0]
    return LineString([(midpoints[0], midpoints[1]), (midpoints[2], midpoints[3])])

def splitPolygonByLongerSides(original_polygon):
    
    # GET BOUNDING BOX
    poly = original_polygon.minimum_rotated_rectangle
    
    # GET LINE TO SPLIT WITH
    line = get_polygon_longest_side_midpoints(original_polygon, 2)
    
    
    # SPLIT
    outstruct = split(poly, line)
    print(outstruct)
    
    return outstruct[0], outstruct[1]