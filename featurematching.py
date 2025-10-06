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

def azimuthDescriptors_noconfidence(coords, angle_step=30, start_angle=None, search_radius=[50], overlap=True):
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
        results = list(tqdm(executor.map(process_point, coords), total=len(coords), leave=False, desc="Calculating Descriptors"))

    return np.array(results)  # Ensure output is array for consistent structu


def azimuthDescriptors(coords, angle_step=30, start_angle=None, search_radius=[50], overlap=True, confidences=None):
    divisor = 1 if overlap else 2

    if not isinstance(search_radius, list):
        search_radius = [search_radius]

    if start_angle is None:
        azimuth_ranges = [(i - angle_step / 2, i + angle_step / 2) for i in np.arange(0, 360, angle_step // divisor)]
    else:
        azimuth_ranges = [(i, i + angle_step) for i in np.arange(start_angle, start_angle + 360, angle_step // divisor)]

    kdtree = KDTree(coords)

    def process_point(index):
        point = coords[index]
        point_results = []
        for radius in search_radius:
            indices = kdtree.query_ball_point(point, r=radius)
            if len(indices) == 0:
                point_results.extend([-1] * len(azimuth_ranges))
                continue

            neighbor_coords = coords[indices]
            azimuths = calculate_azimuths(neighbor_coords, point)
            distances = np.linalg.norm(neighbor_coords - point, axis=1)

            if confidences is not None:
                neighbor_confs = confidences[indices]
            else:
                neighbor_confs = None

            radius_results = []
            for min_az, max_az in azimuth_ranges:
                mask = (azimuths >= min_az) & (azimuths <= max_az)
                filtered_distances = distances[mask]

                if filtered_distances.size > 0:
                    if neighbor_confs is not None:
                        weights = neighbor_confs[mask]
                        if np.sum(weights) > 0:
                            weights = weights / (np.sum(weights) + 1e-6)
                            weighted_mean = np.sum(filtered_distances * weights)
                            radius_results.append(weighted_mean)
                        else:
                            radius_results.append(np.mean(filtered_distances))  # fallback
                    else:
                        radius_results.append(np.mean(filtered_distances))
                else:
                    radius_results.append(-1 * radius)

            radius_results = 1 - np.array(radius_results) / radius
            point_results.extend(radius_results)

        return point_results

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_point, range(len(coords))), total=len(coords), leave=False, desc="Calculating Descriptors"))

    descriptors = np.array(results)
    quality = np.mean(descriptors != -1, axis=1)

    return descriptors, quality



def calcDescriptors_old(gdf, angle_step, azimuth_radius, overlap=True):
    # GET COORDINATES
    coords = getCoordsGDF(gdf) # np.vstack((np.array(gdf.geometry.x), np.array(gdf.geometry.y))).T

    # AZIMUTH DESACRIPTORS
    azimuth_distances = azimuthDescriptors(coords, angle_step=angle_step, search_radius=azimuth_radius, overlap=overlap)

    # SAVE DESCRIPTORS
    gdf[['descriptors' + str(i) for i in range(np.array(azimuth_distances).shape[1])]] = np.array(azimuth_distances)
    return gdf, np.array(azimuth_distances)

def calcDescriptors(dataset, angle_step, search_radius, overlap=True, confidences=None):
    coords = getCoordsGDF(dataset)
    descriptors, quality = azimuthDescriptors(coords, angle_step=angle_step, search_radius=search_radius, overlap=overlap, confidences=confidences)
    dataset['descriptor'] = list(descriptors)
    dataset['descriptor_quality'] = quality
    return dataset, descriptors

def azimuthDescriptors_old(coords, angle_step=30, start_angle=None, search_radius=50, overlap=True):

    divisor = 1 if overlap else 2

    if start_angle is None:
        azimuth_ranges = [(i - angle_step / 2, i + angle_step / 2) for i in np.arange(0, 360, angle_step // divisor)]
    else:
        azimuth_ranges = [(i, i + angle_step) for i in np.arange(start_angle, start_angle+360, angle_step // divisor)]
    kdtree = KDTree(coords)  # Efficient nearest neighbor search
    results = []

    for i, point in tqdm(enumerate(coords), total=coords.shape[0], leave=False, desc="Calculating Descriptors"):
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
    for ref_idx in tqdm(range(n_ref), leave=False, desc="Descriptor Matching"):
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
    # New style: descriptors stored as single list in one column
    if 'descriptor' in gdf.columns:
        desc_list = gdf['descriptor'].tolist()
        if isinstance(desc_list[0], list) or isinstance(desc_list[0], np.ndarray):
            return np.array(desc_list)

    # Fallback: old style, column-per-descriptor
    descriptor_columns = sorted(
        [col for col in gdf.columns if col.startswith('descriptors')],
        key=lambda x: int(x.replace('descriptors', ''))
    )

    if len(descriptor_columns) == 0:
        return None

    return gdf[descriptor_columns].to_numpy()


def getCoordsGDF(gdf):
    return np.vstack((np.array(gdf.geometry.x), np.array(gdf.geometry.y))).T

def calcDescriptors_old_no_quality(gdf, angle_step, azimuth_radius, overlap=True):
    # GET COORDINATES
    coords = getCoordsGDF(gdf) # np.vstack((np.array(gdf.geometry.x), np.array(gdf.geometry.y))).T

    # AZIMUTH DESACRIPTORS
    azimuth_distances = azimuthDescriptors(coords, angle_step=angle_step, search_radius=azimuth_radius, overlap=overlap)

    # SAVE DESCRIPTORS
    gdf[['descriptors' + str(i) for i in range(np.array(azimuth_distances).shape[1])]] = np.array(azimuth_distances)
    return gdf, np.array(azimuth_distances)

def calcDescriptors(gdf, angle_step, azimuth_radius, overlap=True, **kwargs):
    coords = getCoordsGDF(gdf)
    descriptors, quality = azimuthDescriptors(coords, angle_step=angle_step, search_radius=azimuth_radius, overlap=overlap, **kwargs)
    gdf['descriptor'] = list(descriptors)
    gdf['descriptor_quality'] = quality
    return gdf, descriptors

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

def match_descriptors_within_distance_no_mask(reference_descriptors, target_descriptors, reference_coords, target_coords, 
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
    for ref_idx in tqdm(range(n_ref), leave=False, desc="Descriptor Matching"):
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
            top_closest_idx = within_pixel_threshold[sorted_indices]
            top_grades_idx  = filtered_dists[sorted_indices]
            
            # Descriptor distances
            d1 = dist_matrix[ref_idx, closest_idx]
            d2 = dist_matrix[ref_idx, second_closest_idx]
            
            # Apply Lowe's ratio test and descriptor distance threshold
            if d1 / d2 < ratio_threshold and d1 < dist_thresh and d2 < dist_thresh:
                matches.append((ref_idx, closest_idx, d1, d2, reference_coords[ref_idx], target_coords[closest_idx], within_pixel_threshold, top_closest_idx, top_grades_idx))
    
    return matches

def match_descriptors_within_distance_noquality(reference_descriptors, target_descriptors, reference_coords, target_coords, 
                      ratio_threshold=0.75, dist_thresh=1e9, pixel_dist_thresh=500, overlap_penalty_factor=0.5):
    """
    Matches descriptors using cosine distance with overlap penalization,
    applies Lowe's ratio test, and finds the best match within a pixel distance threshold.
    """
    
    n_ref, d = reference_descriptors.shape
    n_target = target_descriptors.shape[0]

    # Build masks for valid bins
    ref_mask = (reference_descriptors != -1).astype(np.float32)
    tar_mask = (target_descriptors != -1).astype(np.float32)

    # Zero-fill invalid values
    ref_filled = np.where(reference_descriptors == -1, 0, reference_descriptors)
    tar_filled = np.where(target_descriptors == -1, 0, target_descriptors)

    # Cosine distances
    dist_matrix = cdist(ref_filled, tar_filled, metric='cosine')

    # Overlap matrix (fraction of valid dimensions)
    overlap_matrix = (ref_mask @ tar_mask.T) / d

    # Penalize low-overlap pairs
    dist_matrix += (1.0 - overlap_matrix) * overlap_penalty_factor

    # Pixel distance matrix
    pixel_dist_matrix = cdist(reference_coords, target_coords, metric='euclidean')

    matches = []

    for ref_idx in tqdm(range(n_ref), leave=False, desc="Descriptor Matching"):
        within_pixel_threshold = np.where(pixel_dist_matrix[ref_idx] < pixel_dist_thresh)[0]

        if len(within_pixel_threshold) > 1:
            filtered_dists = dist_matrix[ref_idx, within_pixel_threshold]
            sorted_indices = np.argsort(filtered_dists)

            closest_idx = within_pixel_threshold[sorted_indices[0]]
            second_closest_idx = within_pixel_threshold[sorted_indices[1]]
            top_closest_idx = within_pixel_threshold[sorted_indices]
            top_grades_idx = filtered_dists[sorted_indices]

            d1 = dist_matrix[ref_idx, closest_idx]
            d2 = dist_matrix[ref_idx, second_closest_idx]

            if d1 / d2 < ratio_threshold and d1 < dist_thresh and d2 < dist_thresh:
                matches.append((
                    ref_idx, closest_idx, d1, d2,
                    reference_coords[ref_idx], target_coords[closest_idx],
                    within_pixel_threshold, top_closest_idx, top_grades_idx
                ))

    return matches

def match_descriptors_within_distance_wrongdir(reference_descriptors, target_descriptors,
                                      reference_coords, target_coords,
                                      ratio_threshold=0.75, dist_thresh=1e9,
                                      pixel_dist_thresh=500, overlap_penalty_factor=0.5,
                                      target_descriptor_quality=None,
                                      quality_weight=0.2):
    """
    Matches descriptors using cosine distance with overlap and descriptor quality penalization.
    """
    n_ref, d = reference_descriptors.shape
    n_target = target_descriptors.shape[0]

    # Masks for valid bins
    ref_mask = (reference_descriptors != -1).astype(np.float32)
    tar_mask = (target_descriptors != -1).astype(np.float32)

    # Fill invalid bins with zero for cosine
    ref_filled = np.where(reference_descriptors == -1, 0, reference_descriptors)
    tar_filled = np.where(target_descriptors == -1, 0, target_descriptors)

    # Cosine distances
    dist_matrix = cdist(ref_filled, tar_filled, metric='cosine')

    # Overlap penalty
    overlap_matrix = (ref_mask @ tar_mask.T) / d
    dist_matrix += (1.0 - overlap_matrix) * overlap_penalty_factor

    # Descriptor quality penalty (if available)
    if target_descriptor_quality is not None:
        target_descriptor_quality = np.clip(target_descriptor_quality, 0, 1)
        quality_penalty = 1.0 - target_descriptor_quality
        dist_matrix += quality_penalty[None, :] * quality_weight

    # Pixel distance check
    pixel_dist_matrix = cdist(reference_coords, target_coords, metric='euclidean')

    matches = []
    for ref_idx in tqdm(range(n_ref), leave=False, desc="Descriptor Matching"):
        within_pixel_threshold = np.where(pixel_dist_matrix[ref_idx] < pixel_dist_thresh)[0]

        if len(within_pixel_threshold) > 1:
            filtered_dists = dist_matrix[ref_idx, within_pixel_threshold]
            sorted_indices = np.argsort(filtered_dists)

            closest_idx = within_pixel_threshold[sorted_indices[0]]
            second_closest_idx = within_pixel_threshold[sorted_indices[1]]
            top_closest_idx = within_pixel_threshold[sorted_indices]
            top_grades_idx = filtered_dists[sorted_indices]

            d1 = dist_matrix[ref_idx, closest_idx]
            d2 = dist_matrix[ref_idx, second_closest_idx]

            if d1 / d2 < ratio_threshold and d1 < dist_thresh and d2 < dist_thresh:
                matches.append((
                    ref_idx, closest_idx, d1, d2,
                    reference_coords[ref_idx], target_coords[closest_idx],
                    within_pixel_threshold, top_closest_idx, top_grades_idx
                ))
        elif len(within_pixel_threshold) == 1:
            only_idx = within_pixel_threshold[0]
            d1 = dist_matrix[ref_idx, only_idx]

            # Safety gates for sparse matches
            sufficient_overlap = overlap_matrix[ref_idx, only_idx] > 0.95
            good_quality = target_descriptor_quality is None or target_descriptor_quality[only_idx] > 0.90

            if d1 < dist_thresh and sufficient_overlap and good_quality:
                matches.append((
                    ref_idx, only_idx, d1, np.inf,
                    reference_coords[ref_idx], target_coords[only_idx],
                    within_pixel_threshold, [only_idx], [d1]
                ))

    return matches

def match_descriptors_within_distance(reference_descriptors, target_descriptors,
                                      reference_coords, target_coords,
                                      ratio_threshold=0.75, dist_thresh=1e9,
                                      pixel_dist_thresh=500, overlap_penalty_factor=0.5,
                                      target_descriptor_quality=None,
                                      quality_weight=0.2):
    """
    Matches target (typically detected) descriptors to reference (typically known/stable)
    using cosine distance with overlap and descriptor quality penalization.
    Returns match tuples with (reference_idx, target_idx, d1, d2, reference_coord, target_coord, ...)
    """
    n_ref, d = reference_descriptors.shape
    n_target = target_descriptors.shape[0]

    # Masks for valid bins
    ref_mask = (reference_descriptors != -1).astype(np.float32)
    tar_mask = (target_descriptors != -1).astype(np.float32)

    # Fill invalid bins with zero for cosine
    ref_filled = np.where(reference_descriptors == -1, 0, reference_descriptors)
    tar_filled = np.where(target_descriptors == -1, 0, target_descriptors)

    # Cosine distances (target rows, reference columns)
    dist_matrix = cdist(tar_filled, ref_filled, metric='cosine')  # (n_target, n_ref)

    # Overlap penalty
    overlap_matrix = (tar_mask @ ref_mask.T) / d
    dist_matrix += (1.0 - overlap_matrix) * overlap_penalty_factor

    # Descriptor quality penalty (only applies to target)
    if target_descriptor_quality is not None:
        target_descriptor_quality = np.clip(target_descriptor_quality, 0, 1)
        quality_penalty = 1.0 - target_descriptor_quality
        dist_matrix += quality_penalty[:, None] * quality_weight

    # Pixel distance matrix (target rows, reference columns)
    pixel_dist_matrix = cdist(target_coords, reference_coords, metric='euclidean')

    matches = []
    for tar_idx in tqdm(range(n_target), leave=False, desc="Descriptor Matching (Flipped)"):
        # within_pixel_threshold = np.where(pixel_dist_matrix[tar_idx] < pixel_dist_thresh)[0] # REFACTOR FOR SPEED BELOW
        within_pixel_threshold = (pixel_dist_matrix[tar_idx] < pixel_dist_thresh).nonzero()[0]

        if len(within_pixel_threshold) > 1:
            filtered_dists = dist_matrix[tar_idx, within_pixel_threshold]
            sorted_indices = np.argsort(filtered_dists)

            closest_idx = within_pixel_threshold[sorted_indices[0]]
            second_closest_idx = within_pixel_threshold[sorted_indices[1]]
            top_closest_idx = within_pixel_threshold[sorted_indices]
            top_grades_idx = filtered_dists[sorted_indices]

            d1 = dist_matrix[tar_idx, closest_idx]
            d2 = dist_matrix[tar_idx, second_closest_idx]

            if d1 / d2 < ratio_threshold and d1 < dist_thresh and d2 < dist_thresh:
                """
                COMMENTING TO SMARTIFY THE MATCHING PROCESS 
                AND KEEP BETTER TRACK
                matches.append((
                    closest_idx, tar_idx, d1, d2,
                    reference_coords[closest_idx], target_coords[tar_idx],
                    within_pixel_threshold, top_closest_idx, top_grades_idx
                ))"""

                overlap_val = overlap_matrix[tar_idx, closest_idx]
                quality_val = target_descriptor_quality[tar_idx] if target_descriptor_quality is not None else 1.0
                lowe_discriminability = 1.0 - (d1 / d2)

                matches.append({
                    "real_idx"          : closest_idx,
                    "image_idx"         : tar_idx,
                    "d1"                : d1,
                    "d2"                : d2,
                    "real_coords"       : reference_coords[closest_idx],
                    "image_coords"      : target_coords[tar_idx],
                    "within_comp"       : within_pixel_threshold,
                    "sorted_match_idx"  : top_closest_idx, 
                    "sorted_grades"     : top_grades_idx,
                    "qualities"         : (quality_val, overlap_val, lowe_discriminability)
                })

        elif False: # len(within_pixel_threshold) == 1:
            only_idx = within_pixel_threshold[0]
            d1 = dist_matrix[tar_idx, only_idx]

            overlap_val = overlap_matrix[tar_idx, only_idx]
            quality_val = target_descriptor_quality[tar_idx] if target_descriptor_quality is not None else 1.0

            # Stringent acceptance for single-match cases
            if d1 < dist_thresh and overlap_val > 0.5 and quality_val > 0.5:
                """
                COMMENTING TO SMARTIFY THE MATCHING PROCESS 
                AND KEEP BETTER TRACK
                matches.append((
                    only_idx, tar_idx, d1, np.inf,
                    reference_coords[only_idx], target_coords[tar_idx],
                    within_pixel_threshold, [only_idx], [d1]
                ))"""
                lowe_discriminability = 1.0

                matches.append({
                    "real_idx"          : closest_idx,
                    "image_idx"         : tar_idx,
                    "d1"                : d1,
                    "d2"                : np.inf,
                    "real_coords"       : reference_coords[closest_idx],
                    "image_coords"      : target_coords[tar_idx],
                    "within_comp"       : within_pixel_threshold,
                    "sorted_match_idx"  : [only_idx], 
                    "sorted_grades"     : [d1], 
                    "qualities"         : (quality_val, overlap_val, lowe_discriminability)
                })

    return matches


def calcQuiver(ax, fromGDF, toGDF, color="black"):
    fromcoords = np.vstack((np.array(fromGDF.geometry.x), np.array(fromGDF.geometry.y))).T
    tocoords = np.vstack((np.array(toGDF.geometry.x), np.array(toGDF.geometry.y))).T

    uv = tocoords - fromcoords
    ax.quiver(fromcoords[:, 0], fromcoords[:, 1], uv[:, 0], uv[:, 1], angles='xy', color=color)

    return ax, uv

def matching_distances_loosen_distance_noquality(tar_dataset, ref_dataset, angle_step=10, azimuth_radius=2000, ratio_threshold=0.8, match_radius=2000, num_retries=10, loosening_factor=0.25):

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
                                                pixel_dist_thresh=match_radius + counter*match_radius*loosening_factor, ratio_threshold=ratio_threshold)
        if num_retries < counter:
            break

    return out, tar_dataset, ref_dataset

def matching_distances_loosen_distance(tar_dataset, ref_dataset, angle_step=10, azimuth_radius=2000,
                                       ratio_threshold=0.8, match_radius=2000, num_retries=10,
                                       loosening_factor=0.25, quality_weight=0.2, overlap=True, confidences=None, recalc_target=False):
    # Descriptors for reference
    ref_descriptors = getDescriptors(ref_dataset)
    if ref_descriptors is None:
        ref_dataset, ref_descriptors = calcDescriptors(ref_dataset, angle_step, azimuth_radius, overlap=overlap)

    # Descriptors for target
    if recalc_target:
        tar_descriptors = getDescriptors(tar_dataset)
        tar_dataset, tar_descriptors = calcDescriptors(tar_dataset, angle_step, azimuth_radius, overlap=overlap, confidences=confidences)
    else:
        tar_descriptors = getDescriptors(tar_dataset)
        if tar_descriptors is None:
            tar_dataset, tar_descriptors = calcDescriptors(tar_dataset, angle_step, azimuth_radius, overlap=overlap, confidences=confidences)
    target_quality = np.array(list(tar_dataset['descriptor_quality']))  # Extract quality

    # Initial match attempt
    out = match_descriptors_within_distance(
        ref_descriptors, tar_descriptors,
        getCoordsGDF(ref_dataset), getCoordsGDF(tar_dataset),
        ratio_threshold=ratio_threshold,
        pixel_dist_thresh=match_radius,
        dist_thresh=4,
        target_descriptor_quality=target_quality,
        quality_weight=quality_weight
    )

    # Retry loop
    counter = 0
    while len(out) < 3:
        counter += 1
        new_radius = match_radius + counter * match_radius * loosening_factor
        print(f"Rematching iteration {counter} with radius {new_radius}; found {len(out)} matches")

        out = match_descriptors_within_distance(
            ref_descriptors, tar_descriptors,
            getCoordsGDF(ref_dataset), getCoordsGDF(tar_dataset),
            ratio_threshold=ratio_threshold,
            pixel_dist_thresh=new_radius,
            dist_thresh=4,
            target_descriptor_quality=target_quality,
            quality_weight=quality_weight
        )
        if counter >= num_retries:
            break

    return out, tar_dataset, ref_dataset