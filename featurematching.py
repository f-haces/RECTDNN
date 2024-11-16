import numpy as np
import geopandas as gpd
from scipy.spatial import KDTree
from tqdm.autonotebook import tqdm
from concurrent.futures import ThreadPoolExecutor

import contextily as cx

from IndexUtils import * 
from TileUtils import *

# TILED INFERENCE
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

def calculate_azimuths(points, target):
    # Calculate the azimuths in a vectorized manner
    dx, dy = points[:, 0] - target[0], points[:, 1] - target[1]
    azimuths = np.degrees(np.arctan2(dy, dx)) % 360
    return azimuths

def azimuthDescriptors(coords, angle_step=30, start_angle=None, search_radius=[50], overlap=True):
    divisor = 1 if overlap else 2

    if not isinstance(search_radius, list):
        search_radius = [search_radius]

    if start_angle is None:
        azimuth_ranges = [(i - angle_step / 2, i + angle_step / 2) for i in np.arange(0, 360, angle_step // divisor)]
    else:
        azimuth_ranges = [(i, i + angle_step) for i in np.arange(start_angle, start_angle + 360, angle_step // divisor)]

    kdtree = KDTree(coords)  # Build KDTree once

    def process_point(point):
        point_results = []
        for radius in search_radius:
            # Get neighbors within current radius
            indices = kdtree.query_ball_point(point, r=radius)
            if len(indices) == 0:
                # If no neighbors within radius, store default value and continue
                point_results.extend([-1] * len(azimuth_ranges))
                continue
            
            neighbor_coords = coords[indices]
            azimuths = calculate_azimuths(neighbor_coords, point)
            distances = np.linalg.norm(neighbor_coords - point, axis=1)

            radius_results = []
            for min_az, max_az in azimuth_ranges:
                mask = (azimuths >= min_az) & (azimuths <= max_az)
                filtered_distances = distances[mask]

                if filtered_distances.size > 0:
                    radius_results.append(np.mean(filtered_distances))
                else:
                    radius_results.append(-1 * radius)

            # Normalize and append radius results
            radius_results = 1 - np.array(radius_results) / radius
            point_results.extend(radius_results)

        return point_results

    # Parallel processing
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_point, coords), total=len(coords), leave=False))

    return np.array(results)  # Ensure output is array for consistent structu

def calcDescriptors(gdf, angle_step, azimuth_radius, overlap=True):
    # GET COORDINATES
    coords = getCoordsGDF(gdf) # np.vstack((np.array(gdf.geometry.x), np.array(gdf.geometry.y))).T

    # AZIMUTH DESACRIPTORS
    azimuth_distances = azimuthDescriptors(coords, angle_step=angle_step, search_radius=azimuth_radius, overlap=overlap)

    # SAVE DESCRIPTORS
    gdf[['descriptors' + str(i) for i in range(np.array(azimuth_distances).shape[1])]] = np.array(azimuth_distances)
    return gdf, np.array(azimuth_distances)

"""
def azimuthDescriptors(coords, angle_step=30, start_angle=None, search_radii=[50], overlap=True):

    if not isinstance(search_radii, list):
        search_radii = [search_radii]

    divisor = 1 if overlap else 2

    if start_angle is None:
        azimuth_ranges = [(i - angle_step / 2, i + angle_step / 2) for i in np.arange(0, 360, angle_step // divisor)]
    else:
        azimuth_ranges = [(i, i + angle_step) for i in np.arange(start_angle, start_angle + 360, angle_step // divisor)]

    kdtree = KDTree(coords)  # Build KDTree once
    max_radius = max(search_radii)  # Largest radius for a single KDTree query

    def process_point(point):
        point_results = []

        # Get points within the max search radius only once
        indices = kdtree.query_ball_point(point, r=max_radius)
        neighbor_coords = coords[indices]

        # Calculate azimuths and distances once
        azimuths = calculate_azimuths(neighbor_coords, point)
        distances = np.linalg.norm(neighbor_coords - point, axis=1)

        for radius in search_radii:
            # Filter neighbors within the current radius
            mask_radius = distances <= radius
            radius_neighbors = neighbor_coords[mask_radius]
            radius_distances = distances[mask_radius]
            radius_azimuths = azimuths[mask_radius]

            radius_results = []
            for min_az, max_az in azimuth_ranges:
                # Filter based on azimuth range
                mask_az = (radius_azimuths >= min_az) & (radius_azimuths <= max_az)
                filtered_distances = radius_distances[mask_az]

                if filtered_distances.size > 0:
                    radius_results.append(np.mean(filtered_distances))
                else:
                    radius_results.append(-1 * radius)

            # Normalize radius results and append
            radius_results = 1 - np.array(radius_results) / radius
            point_results.extend(radius_results)

        return point_results

    # Use parallel processing to speed up processing of each point
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_point, coords), total=len(coords), leave=False))

    return results"""

def azimuthDescriptors_old(coords, angle_step=30, start_angle=None, search_radius=50, overlap=True):

    divisor = 1 if overlap else 2

    if start_angle is None:
        azimuth_ranges = [(i - angle_step / 2, i + angle_step / 2) for i in np.arange(0, 360, angle_step // divisor)]
    else:
        azimuth_ranges = [(i, i + angle_step) for i in np.arange(start_angle, start_angle+360, angle_step // divisor)]
    kdtree = KDTree(coords)  # Efficient nearest neighbor search
    results = []

    for i, point in tqdm(enumerate(coords), total=coords.shape[0], leave=False):
        point_results = []
        
        # Get points within the search radius around the current point
        indices = kdtree.query_ball_point(point, r=search_radius)
        neighbor_coords = coords[indices]
        
        # Vectorized azimuth calculation for neighbors
        azimuths = calculate_azimuths(neighbor_coords, point)
        distances = np.linalg.norm(neighbor_coords - point, axis=1)
        
        for min_az, max_az in azimuth_ranges:
            # Filter indices based on azimuth range
            mask = (azimuths >= min_az) & (azimuths <= max_az)
            filtered_distances = distances[mask]
            filtered_neighbors = neighbor_coords[mask]
            
            if filtered_neighbors.size > 0:
                # Find the closest point within the filtered neighbors
                point_results.append(np.mean(filtered_distances))
            else:
                point_results.append(-1 * search_radius)
                # point_results.append((None, None))  # No point in range

        point_results = 1 - np.array(point_results) / search_radius
        results.append(point_results)
    return results

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

def match_descriptors(reference_descriptors, target_descriptors, ratio_threshold=0.75, dist_thresh=2):
    """
    Matches descriptors and applies the Lowe ratio test.
    
    Args:
    reference_descriptors (ndarray): Descriptors from the reference image (n_ref x d).
    target_descriptors (ndarray): Descriptors from the target image (n_target x d).
    ratio_threshold (float): The threshold for the Lowe ratio test (default is 0.75).
    
    Returns:
    list of (ref_idx, tgt_idx) matches that pass the Lowe ratio test.
    """
    
    # Number of reference and target descriptors
    n_ref = reference_descriptors.shape[0]
    n_target = target_descriptors.shape[0]

    # Array to store matches
    matches = []
    
    # Vectorized distance calculation
    # Compute all pairwise distances between reference and target descriptors
    # This uses broadcasting to compute the Euclidean distances efficiently
    # dist_matrix will be of shape (n_ref, n_target)
    # dist_matrix = np.linalg.norm(reference_descriptors[:, np.newaxis, :] - target_descriptors[np.newaxis, :, :], axis=2)
    dist_matrix = cdist(reference_descriptors, target_descriptors, metric='cosine')
    
    # For each descriptor in the reference set
    for ref_idx in tqdm(range(n_ref), leave=False):
        # Sort the distances to find the two smallest distances
        sorted_indices = np.argsort(dist_matrix[ref_idx])
        closest_idx = sorted_indices[0]  # Closest descriptor in the target set
        second_closest_idx = sorted_indices[1]  # Second closest
        
        # Compute the distances to the two closest target descriptors
        d1 = dist_matrix[ref_idx, closest_idx]
        d2 = dist_matrix[ref_idx, second_closest_idx]

        
        # Apply the Lowe ratio test
        if d1 / d2 < ratio_threshold:
            # If the ratio passes the threshold, store the match (ref_idx, closest_idx)
            if d1 < dist_thresh and d2 < dist_thresh: 
                matches.append((ref_idx, closest_idx, d1, d2))
    
    return matches


def calcQuiver(ax, fromGDF, toGDF, color="black"):
    fromcoords = np.vstack((np.array(fromGDF.geometry.x), np.array(fromGDF.geometry.y))).T
    tocoords = np.vstack((np.array(toGDF.geometry.x), np.array(toGDF.geometry.y))).T

    uv = tocoords - fromcoords
    ax.quiver(fromcoords[:, 0], fromcoords[:, 1], uv[:, 0], uv[:, 1], angles='xy', color=color)

    return ax, uv

def normArry(arry):
    return (arry - np.min(arry)) / (np.max(arry) - np.min(arry))

def normCoords(A, B):
    offsets = np.mean(A, axis=0)
    return A - offsets, B - offsets

def most_popular_indices(values, eps=0.1):
    # Convert values to a 2D array as required by DBSCAN
    values_2d = np.array(values).reshape(-1, 1)
    
    # Cluster the values using DBSCAN
    db = DBSCAN(eps=eps, min_samples=2).fit(values_2d)
    
    # Find the most common cluster label, ignoring noise (-1 label)
    labels, counts = np.unique(db.labels_[db.labels_ != -1], return_counts=True)
    if len(counts) == 0:
        return []  # No clusters found
    
    most_common_label = labels[np.argmax(counts)]
    
    # Get indices of values in the most common cluster
    indices = [i for i, label in enumerate(db.labels_) if label == most_common_label]
    
    return indices

def most_popular_indices_2d(points, eps=0.1):
    # Convert the list of 2D points to a numpy array
    points_array = np.array(points)
    
    # Cluster the points using DBSCAN
    db = DBSCAN(eps=eps, min_samples=2).fit(points_array)
    
    # Find the most common cluster label, ignoring noise (-1 label)
    labels, counts = np.unique(db.labels_[db.labels_ != -1], return_counts=True)
    if len(counts) == 0:
        return []  # No clusters found
    
    most_common_label = labels[np.argmax(counts)]
    
    # Get indices of points in the most common cluster
    indices = [i for i, label in enumerate(db.labels_) if label == most_common_label]
    
    return indices

def get_non_outlier_indices(data, threshold=2):
    """
    Detects non-outlier indices based on a standard deviation threshold.

    Parameters:
    - data (array-like): The data array to check.
    - threshold (float): The number of standard deviations to use as the cutoff.

    Returns:
    - non_outlier_indices (numpy array): Indices of the data points that are not outliers.
    """
    # Calculate mean and standard deviation
    mean = np.mean(data)
    std_dev = np.std(data)
    
    # Determine which points are within the threshold
    non_outliers = np.abs(data - mean) <= threshold * std_dev
    print(mean, threshold * std_dev)
    
    # Return indices of non-outliers
    return np.where(non_outliers)[0]


def getDescriptors(gdf):
    # Filter columns that start with 'descriptors' and sort them
    descriptor_columns = sorted([col for col in gdf.columns if col.startswith('descriptors')],
                                key=lambda x: int(x.replace('descriptors', '')))
    
    if len(descriptor_columns) == 0:
        return None
    
    # Select only these columns and convert to a 2D array
    descriptor_array = gdf[descriptor_columns].to_numpy()
    
    return np.array(descriptor_array)

def getCoordsGDF(gdf):
    return np.vstack((np.array(gdf.geometry.x), np.array(gdf.geometry.y))).T

def calcDescriptors(gdf, angle_step, azimuth_radius, overlap=True):
    # GET COORDINATES
    coords = getCoordsGDF(gdf) # np.vstack((np.array(gdf.geometry.x), np.array(gdf.geometry.y))).T

    # AZIMUTH DESACRIPTORS
    azimuth_distances = azimuthDescriptors(coords, angle_step=angle_step, search_radius=azimuth_radius, overlap=overlap)

    # SAVE DESCRIPTORS
    gdf[['descriptors' + str(i) for i in range(np.array(azimuth_distances).shape[1])]] = np.array(azimuth_distances)
    return gdf, np.array(azimuth_distances)

def matching(tar_dataset, ref_dataset, angle_step=10, azimuth_radius=2000, ratio_threshold=0.8, num_retries=10, loosening_factor=0.01):

    # ALL STREETS AZIMUTHS
    ref_descriptors = getDescriptors(ref_dataset)

    # IF WE HAVEN'T CALCULATED THEM BEFORE
    if ref_descriptors is None:
        ref_dataset, ref_descriptors = calcDescriptors(ref_dataset, angle_step, azimuth_radius, overlap=True)

    tar_dataset, tar_descriptors = calcDescriptors(tar_dataset, angle_step, azimuth_radius, overlap=True)

    out = match_descriptors(tar_descriptors, ref_descriptors, ratio_threshold=ratio_threshold)

    counter = 0
    while len(out) < 0.1 * tar_dataset.shape[0]:
        print(f"Couldn't find with initial match params, rematching iteration {counter}; {len(out)} {tar_dataset.shape[0]}")
        counter = counter + 1
        out = match_descriptors(tar_descriptors, ref_descriptors, ratio_threshold=ratio_threshold+ loosening_factor * counter)
        if num_retries < counter:
            break


    return out, tar_dataset, ref_dataset


def matching_distances(tar_dataset, ref_dataset, angle_step=10, azimuth_radius=2000, ratio_threshold=0.8, match_radius=500, num_retries=10, loosening_factor=0.01):

    # ALL STREETS AZIMUTHS
    ref_descriptors = getDescriptors(ref_dataset)

    # IF WE HAVEN'T CALCULATED THEM BEFORE
    if ref_descriptors is None:
        ref_dataset, ref_descriptors = calcDescriptors(ref_dataset, angle_step, azimuth_radius, overlap=True)

    tar_dataset, tar_descriptors = calcDescriptors(tar_dataset, angle_step, azimuth_radius, overlap=True)

    # out = match_descriptors(tar_descriptors, ref_descriptors, ratio_threshold=ratio_threshold)
    out = match_descriptors_within_distance(ref_descriptors, tar_descriptors, getCoordsGDF(ref_dataset), getCoordsGDF(tar_dataset),#  reference_coords, target_coords, 
                      ratio_threshold=ratio_threshold, pixel_dist_thresh=match_radius)

    counter = 0
    while len(out) < 3: # 0.1 * tar_dataset.shape[0]:
        print(f"Couldn't find with initial match params, rematching iteration {counter}; {len(out)} {tar_dataset.shape[0]}")
        counter = counter + 1
        out = match_descriptors_within_distance(ref_descriptors, tar_descriptors, getCoordsGDF(ref_dataset), getCoordsGDF(tar_dataset),#  reference_coords, target_coords, 
                      ratio_threshold=ratio_threshold + loosening_factor * counter, pixel_dist_thresh=match_radius)
        if num_retries < counter:
            break

    return out, tar_dataset, ref_dataset

def match_descriptors_within_distance(reference_descriptors, target_descriptors, reference_coords, target_coords, 
                      ratio_threshold=0.75, dist_thresh=1e9, pixel_dist_thresh=500):
    """
    Matches descriptors using Euclidean distance, applies Lowe's ratio test,
    and finds the best match within a specified pixel distance threshold.
    
    Args:
    reference_descriptors (ndarray): Descriptors from the reference image (n_ref x d).
    target_descriptors (ndarray): Descriptors from the target image (n_target x d).
    reference_coords (ndarray): Coordinates of reference descriptors (n_ref x 2).
    target_coords (ndarray): Coordinates of target descriptors (n_target x 2).
    ratio_threshold (float): The threshold for the Lowe ratio test (default is 0.75).
    dist_thresh (float): Descriptor distance threshold.
    pixel_dist_thresh (float): Pixel distance threshold to restrict matches spatially.
    
    Returns:
    list of (ref_idx, tgt_idx, d1, d2) matches that pass the Lowe ratio test within the pixel distance threshold.
    """
    
    n_ref = reference_descriptors.shape[0]
    n_target = target_descriptors.shape[0]

    # Array to store matches
    matches = []
    
    # Calculate descriptor distances and pixel distances
    dist_matrix = cdist(reference_descriptors, target_descriptors, metric='cosine')
    pixel_dist_matrix = cdist(reference_coords, target_coords, metric='euclidean')
    
    # For each descriptor in the reference set
    for ref_idx in tqdm(range(n_ref), leave=False):
        # Get all target descriptors within the pixel distance threshold
        within_pixel_threshold = np.where(pixel_dist_matrix[ref_idx] < pixel_dist_thresh)[0]
        
        # Continue only if there are targets within the pixel distance
        if len(within_pixel_threshold) > 1:
            # Get descriptor distances for these target descriptors
            filtered_dists = dist_matrix[ref_idx, within_pixel_threshold]

            # Sort indices by descriptor distance
            sorted_indices = np.argsort(filtered_dists)
            closest_idx = within_pixel_threshold[sorted_indices[0]]
            second_closest_idx = within_pixel_threshold[sorted_indices[1]]
            
            # Descriptor distances
            d1 = dist_matrix[ref_idx, closest_idx]
            d2 = dist_matrix[ref_idx, second_closest_idx]
            
            # Apply Lowe's ratio test and descriptor distance threshold
            if d1 / d2 < ratio_threshold and d1 < dist_thresh and d2 < dist_thresh:
                matches.append((ref_idx, closest_idx, d1, d2, reference_coords[ref_idx], target_coords[closest_idx]))
    
    return matches

def calcQuiver(ax, fromGDF, toGDF, color="black"):
    fromcoords = np.vstack((np.array(fromGDF.geometry.x), np.array(fromGDF.geometry.y))).T
    tocoords = np.vstack((np.array(toGDF.geometry.x), np.array(toGDF.geometry.y))).T

    uv = tocoords - fromcoords
    ax.quiver(fromcoords[:, 0], fromcoords[:, 1], uv[:, 0], uv[:, 1], angles='xy', color=color)

    return ax, uv

def plotMatches(fromPoints, toPoints, dbscan_eps=0.04):
    fig, axs = plt.subplots(1, 2, figsize=(15,5))

    fromPoints.plot(ax=axs[0])
    toPoints.plot(ax=axs[0])
    axs[0], uv = calcQuiver(axs[0], fromPoints, toPoints, color="black")

    try:
        cx.add_basemap(axs[0])
    except:
        print("Error adding Basemap")

    angles = np.degrees(np.arctan2(uv[:, 1], uv[:, 0]))
    test = np.vstack((normArry(angles), normArry(np.sqrt(uv[:, 0] ** 2, uv[:, 1] ** 2 )))).T
    idx = most_popular_indices_2d(test, eps=dbscan_eps)

    axs[0], uv = calcQuiver(axs[0], fromPoints.iloc[idx], toPoints.iloc[idx], color="yellow")

    bins = np.linspace(-180, 180, 100)

    axs[1].hist(angles, bins, label="All")
    axs[1].hist(angles[idx], bins, label="Selected")
    axs[1].set_xlabel("Adjustment Bearing")
    axs[1].set_ylabel("Frequency (n)")

    return fig, axs, uv, idx


def matching_distances_loosen_distance(tar_dataset, ref_dataset, angle_step=10, azimuth_radius=2000, ratio_threshold=0.8, match_radius=2000, num_retries=10, loosening_factor=0.25):

    # ALL STREETS AZIMUTHS
    ref_descriptors = getDescriptors(ref_dataset)

    # IF WE HAVEN'T CALCULATED THEM BEFORE
    if ref_descriptors is None:
        ref_dataset, ref_descriptors = calcDescriptors(ref_dataset, angle_step, azimuth_radius, overlap=True)

    tar_dataset, tar_descriptors = calcDescriptors(tar_dataset, angle_step, azimuth_radius, overlap=True)

    # out = match_descriptors(tar_descriptors, ref_descriptors, ratio_threshold=ratio_threshold)
    out = match_descriptors_within_distance(ref_descriptors, tar_descriptors, getCoordsGDF(ref_dataset), getCoordsGDF(tar_dataset),#  reference_coords, target_coords, 
                      ratio_threshold=ratio_threshold, pixel_dist_thresh=match_radius, dist_thresh=4)

    counter = 0
    while len(out) < 3: # 0.1 * tar_dataset.shape[0]:
        counter = counter + 1
        print(f"Couldn't find with initial match params, rematching iteration {counter} with radius {match_radius + counter*match_radius*loosening_factor}; Currently {len(out)} matches, expecting {0.1 * tar_dataset.shape[0]}")
        
        out = match_descriptors_within_distance(ref_descriptors, tar_descriptors, getCoordsGDF(ref_dataset), getCoordsGDF(tar_dataset), 
                                                pixel_dist_thresh=match_radius + counter*match_radius*loosening_factor, ratio_threshold=0.95)
        if num_retries < counter:
            break

    return out, tar_dataset, ref_dataset

def iterativeAdjustFromMatching(matchresults, corners_curr=None, idx=None, im_corner_gdf=None, plot=False, verbose=False, dbscan_eps=0.04):
    if idx is None:
        idx = np.arange(len(matchresults), dtype=np.int32)
    
    if len(matchresults[0]) > 4:
        fromPoints = gpd.GeoDataFrame(geometry=gpd.points_from_xy([x[4][0] for x in matchresults], [x[4][1] for x in matchresults]))
        toPoints   = gpd.GeoDataFrame(geometry=gpd.points_from_xy([x[5][0] for x in matchresults], [x[5][1] for x in matchresults]))
    else:
        fromPoints = np.vstack((np.array(corners_curr.iloc[np.array(matchresults)[idx, 1]].geometry.x), np.array(corners_curr.iloc[np.array(matchresults)[idx, 1]].geometry.y))).T
        toPoints = np.vstack((np.array(im_corner_gdf.iloc[np.array(matchresults)[idx, 0]].geometry.x), np.array(im_corner_gdf.iloc[np.array(matchresults)[idx, 0]].geometry.y))).T
        fromPoints = gpd.GeoDataFrame(geometry=gpd.points_from_xy(fromPoints[:, 0], fromPoints[:, 1]))
        toPoints = gpd.GeoDataFrame(geometry=gpd.points_from_xy(toPoints[:, 0], toPoints[:, 1]))

    coordsA = getCoordsGDF(fromPoints)
    coordsB = getCoordsGDF(toPoints)

    coordsA, coordsB = normCoords(coordsA, coordsB)

    i = 0

    checker = True
    prev = np.where(idx)[0].shape[0]

    if plot:
        plotMatches(fromPoints, toPoints, dbscan_eps=dbscan_eps)
    while checker:
        initial = affineTransformation(coordsA[idx, 0], coordsA[idx, 1], coordsB[idx, 0], coordsB[idx, 1],verbose=False, )
        matrix = initial.matrix        
        coordsBprime = np.hstack((coordsB[idx], np.ones((coordsB[idx].shape[0], 1)))) @ np.linalg.inv(matrix).T


        distances = np.sqrt((coordsBprime[:, 0] - coordsA[idx, 0]) ** 2 + (coordsBprime[:, 1] - coordsA[idx, 1]) ** 2)
        uv = coordsB - coordsA
        angles = np.degrees(np.arctan2(uv[:, 1], uv[:, 0]))
        test = np.vstack((normArry(angles), normArry(np.sqrt(uv[:, 0] ** 2, uv[:, 1] ** 2 )))).T
        test[np.isnan(test)] = 0
        idx = most_popular_indices_2d(test, eps=dbscan_eps)

        counter = 0
        if np.where(idx)[0].shape[0] < 5:
            print("Loosening, under 5... ", end=" ")
            while np.where(idx)[0].shape[0] < 5:
                counter = counter + 1
                idx = most_popular_indices_2d(test, eps=dbscan_eps + 0.02 * counter)
                if counter > 10:
                    break
            print(f"Loosened {counter} times at {dbscan_eps + 0.02 * counter}")
        
            

        if np.where(idx)[0].shape[0] < prev:
            prev = np.where(idx)[0].shape[0]
        else: 
            checker = False
        
        if verbose:
            print(i)
            print(np.linalg.inv(matrix).T)
            fig, axs = plt.subplots(1, 2)
            axs[0].scatter(coordsA[idx, 0], coordsA[idx, 1], color='black')
            axs[0].scatter(coordsB[idx, 0], coordsB[idx, 1])
            axs[0].scatter(coordsBprime[:, 0], coordsBprime[:, 1], marker='x')
            axs[1].hist(distances, bins=50)
            axs[1].set_title(f"Iteration {i}")
            plt.show()

        i = i + 1

    return matrix, distances

