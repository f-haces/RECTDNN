import numpy as np

def get_image_sectors(image_width, image_height, coordinates):
    """
    Determine which sector of an image each coordinate belongs to.

    Parameters:
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        coordinates (list or numpy array): Nx2 array of (x, y) coordinates.

    Returns:
        list: A list of sectors for each coordinate.
              Each sector is one of ['top left', 'top right', 'bottom left', 'bottom right'].
    """
    sectors = []
    mid_x, mid_y = image_width / 2, image_height / 2

    for x, y in coordinates:
        if x < mid_x and y < mid_y:
            sectors.append("top left")
        elif x >= mid_x and y < mid_y:
            sectors.append("top right")
        elif x < mid_x and y >= mid_y:
            sectors.append("bottom left")
        else:  # x >= mid_x and y >= mid_y
            sectors.append("bottom right")
    
    return sectors

def all_sectors_present(sectors):
    """
    Check if all four image sectors are present in the given list.

    Parameters:
        sectors (list): A list of sectors, e.g., ['top left', 'top right', 'bottom left', 'bottom right'].

    Returns:
        bool: True if all sectors are present, False otherwise.
    """
    required_sectors = {'top left', 'top right', 'bottom left', 'bottom right'}
    return required_sectors.issubset(set(sectors))


def group_coordinates_by_sector(coordinates, sectors):
    """
    Group coordinates by sector.

    Parameters:
        coordinates (list or numpy array): Nx2 array of (x, y) coordinates.
        sectors (list): List of sector names corresponding to each coordinate.

    Returns:
        dict: A dictionary with sector names as keys and lists of coordinates as values.
    """
    sector_coordinates = {
        "top left": [],
        "top right": [],
        "bottom left": [],
        "bottom right": []
    }
    for coord, sector in zip(coordinates, sectors):
        sector_coordinates[sector].append(coord)
    return sector_coordinates

def compute_sector_center(coords):
    """
    Compute the center of a group of coordinates.

    Parameters:
        coords (numpy array): Array of (x, y) coordinates.

    Returns:
        tuple: (center_x, center_y) representing the center of the coordinates.
    """
    return coords[:, 0].mean(), coords[:, 1].mean()

def compute_bounding_box(coords, center_x, center_y, min_width, min_height, buffer):
    """
    Compute the bounding box for a sector given its coordinates.

    Parameters:
        coords (numpy array): Array of (x, y) coordinates in the sector.
        center_x (float): Center x-coordinate of the sector.
        center_y (float): Center y-coordinate of the sector.
        min_width (float): Minimum width of the bounding box.
        min_height (float): Minimum height of the bounding box.
        buffer (float): Additional buffer around the bounding box.

    Returns:
        tuple: Bounding box in the format (x_min, y_min, x_max, y_max).
    """
    x_min = min(coords[:, 0]) - buffer
    x_max = max(coords[:, 0]) + buffer
    y_min = min(coords[:, 1]) - buffer
    y_max = max(coords[:, 1]) + buffer

    # Ensure the bounding box has at least the minimum dimensions
    if x_max - x_min < min_width:
        delta = (min_width - (x_max - x_min)) / 2
        x_min = center_x - min_width / 2
        x_max = center_x + min_width / 2
    if y_max - y_min < min_height:
        delta = (min_height - (y_max - y_min)) / 2
        y_min = center_y - min_height / 2
        y_max = center_y + min_height / 2

    return x_min, y_min, x_max, y_max

def compute_bounding_boxes(coordinates, sectors, min_width, min_height, buffer=0):
    """
    Main function to compute bounding boxes for all sectors.

    Parameters:
        coordinates (list or numpy array): Nx2 array of (x, y) coordinates.
        sectors (list): List of sector names corresponding to each coordinate.
        min_width (float): Minimum width of the bounding box.
        min_height (float): Minimum height of the bounding box.
        buffer (float): Additional buffer around the bounding box.

    Returns:
        dict: A dictionary where keys are sector names, and values are bounding box tuples
              in the format (x_min, y_min, x_max, y_max).
    """
    sector_coordinates = group_coordinates_by_sector(coordinates, sectors)
    sector_bboxes = {}

    for sector, coords in sector_coordinates.items():
        if not coords:
            # Skip if no coordinates in this sector
            continue

        coords = np.array(coords)
        center_x, center_y = compute_sector_center(coords)
        bbox = compute_bounding_box(coords, center_x, center_y, min_width, min_height, buffer)
        sector_bboxes[sector] = bbox

    return sector_bboxes

def extract_bounding_boxes(image, bounding_boxes):
    """
    Extract regions of an image corresponding to the given bounding boxes.

    Parameters:
        image (numpy array): The input image as a NumPy array (H x W x C for color or H x W for grayscale).
        bounding_boxes (dict): A dictionary where keys are sector names and values are bounding box tuples
                               in the format (x_min, y_min, x_max, y_max).

    Returns:
        dict: A dictionary where keys are sector names and values are cropped image regions (numpy arrays).
    """
    cropped_images = {}
    for sector, (x_min, y_min, x_max, y_max) in bounding_boxes.items():
        # Ensure bounding box values are integers and within image bounds
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(image.shape[1], int(x_max))
        y_max = min(image.shape[0], int(y_max))
        
        # Crop the image region
        cropped_images[sector] = image[y_min:y_max, x_min:x_max]

    return cropped_images

import re

def clean_and_convert_coordinates(raw_coords):
    """
    Clean and convert a list of raw OCR-extracted coordinates to decimal degrees.

    Parameters:
        raw_coords (list of str): Raw OCR strings containing coordinates.

    Returns:
        list of tuple: List of cleaned and converted coordinates as (latitude, longitude).
    """
    def clean_coordinate(coord):
        # Remove unwanted symbols and normalize
        """coord = re.sub(r"[=]", "", coord)
        coord = coord.replace("'", "")
        coord = coord.replace('"', "")
        coord = coord.replace('°', '')
        coord = coord.replace('*', '')
        coord = coord.strip()"""

        coord = re.sub('[^0-9]','', coord)


        match = re.match(r"^(\d{3})(\d{4})", coord)


        if match:
            degrees_part = match.group(1)[:-1]  # Remove trailing '0' from degrees
            coord = degrees_part + match.group(2)
        
        return coord

    def parse_and_convert(coord):
        # Match patterns for degrees, minutes, and seconds
        match = re.match(r"(\d{2})+(\d{2})+(\d{2})", coord)
        if not match:
            return None
        degrees = int(match.group(1))
        minutes = int(match.group(2)) if match.group(2) else 0
        seconds = int(match.group(3)) if match.group(3) else 0
        # Convert to decimal degrees
        return degrees + minutes / 60 + seconds / 3600

    cleaned_coords = []
    for raw_coord in raw_coords:
        coord = clean_coordinate(raw_coord)
        decimal_coord = parse_and_convert(coord)
        if decimal_coord is not None:
            cleaned_coords.append(decimal_coord)
    
    return cleaned_coords