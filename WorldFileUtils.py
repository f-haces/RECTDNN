import numpy as np
import glob
from osgeo import gdal, osr
import pyproj
from affine import Affine
from shutil import copyfile

def get_geotransform_from_tfw(tfw_file):
    # READ TFW FILE AND RETURN GDAL GEOTRANSFORM
    with open(tfw_file, 'r') as f:
        tfw = [float(line) for line in f if line is not "\n"]

    geotransform = (tfw[4], tfw[0], tfw[1], tfw[5], tfw[2], tfw[3])
    
    # CHECK LAST OBSERVATION, IF IT'S BIGGER THAN 10e7 IT'S PROLLY STATE PLANE, ELSE UTM
    if np.abs(tfw[5]) > 1e7 :
        outcrs = "ESRI:102740"
    else:
        outcrs = "EPSG:32615"
    return geotransform, outcrs

def affine_to_geotransform(affine_obj):
    """Convert an affine object to a geotransform tuple.
    
    Parameters:
        affine_obj (affine.Affine): The affine object to convert.
    
    Returns:
        tuple: The geotransform tuple (x_origin, x_resolution, x_rotation, y_origin, y_rotation, y_resolution).
    """
    x_origin = affine_obj.c
    y_origin = affine_obj.f
    x_resolution = affine_obj.a
    y_resolution = affine_obj.e
    x_rotation = affine_obj.b
    y_rotation = affine_obj.d
    return (x_origin, x_resolution, x_rotation, y_origin, y_rotation, y_resolution)

def reproject_geotransform_old(geotransform, src_crs, dst_crs):
    src_proj = pyproj.Proj(src_crs)
    dst_proj = pyproj.Proj(dst_crs)
    
    #
    # Create a transformation object to convert between the input and output CRS
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs)

    # Transform the coordinates of the upper-left and lower-right corners of the input image
    upper_left = transformer.transform(geotransform[0], geotransform[3])
    lower_right = transformer.transform(geotransform[0] + geotransform[1] * geotransform[2],
                              geotransform[3] + geotransform[5] * geotransform[4])
    
    reprojected_geotransform = [upper_left[0], geotransform[1], geotransform[2], upper_left[1], geotransform[4], geotransform[5]]
    
    return reprojected_geotransform

def getDistWithTransform(transformer, x_res,):
    firstpoint  = transformer.transform(0, x_res)
    secondpoint = transformer.transform(0, 0)
    out = np.sqrt((firstpoint[0] - secondpoint[0]) ** 2 + (firstpoint[1] - secondpoint[1]) ** 2)
    out = np.sign(x_res) * out
    return out

def reproject_geotransform(geotransform, original_crs, target_crs):
    """Transform a geotransform to a different CRS using pyproj.

    Parameters:
        geotransform (tuple): The original geotransform to transform.
        original_crs (str): The original CRS of the geotransform.
        target_crs (str): The target CRS for the transformed geotransform.

    Returns:
        tuple: The transformed geotransform in the target CRS (x_origin, x_resolution, x_rotation, y_origin, y_rotation, y_resolution).
    """
    # Create a pyproj transformer object
    transformer = pyproj.Transformer.from_crs(original_crs, target_crs)

    # Transform the geotransform parameters to the target CRS
    x_origin, x_resolution, x_rotation, y_origin, y_rotation, y_resolution = geotransform
    x_origin_transformed, y_origin_transformed, z = transformer.transform(x_origin, y_origin, 0)

    # Transform the scale factors
    # x_res_transformed, y_res_transformed = transformer.transform(0, x_resolution), transformer.transform(0, y_resolution)
    # x_res_transformed, y_res_transformed = x_res_transformed[0], y_res_transformed[0]
    x_res_transformed = getDistWithTransform(transformer, x_resolution,)
    y_res_transformed = getDistWithTransform(transformer, y_resolution,)
    x_rotation        = getDistWithTransform(transformer, x_rotation,)
    y_rotation        = getDistWithTransform(transformer, x_rotation,)

    # Create a new geotransform with the transformed parameters
    transformed_geotransform = (x_origin_transformed, x_res_transformed, x_rotation,
                                y_origin_transformed, y_rotation, y_res_transformed)

    return transformed_geotransform

def get_affine_from_geotransform(geotransform):
    return Affine.from_gdal(*geotransform)

def AffineFromTFW(tfw_file, new_crs, old_crs=None):
    geotransform, detected_crs = get_geotransform_from_tfw(tfw_file)
    if old_crs is None:
        old_crs = detected_crs
    new_geotransform = reproject_geotransform(geotransform, old_crs, new_crs)
    print(new_geotransform)
    affine = get_affine_from_geotransform(new_geotransform)
    return affine
    
def AffineFromTIF(tif_file, new_crs):
    raster_ds = gdal.Open(tif_file)
    geotransform = raster_ds.GetGeoTransform()
    old_crs = raster_ds.GetProjection()
    new_geotransform = reproject_geotransform(geotransform, old_crs, new_crs)
    affine = get_affine_from_geotransform(new_geotransform)
    return affine

def getMatrixFromAffine(affine_transform):
    mt = affine_transform.to_gdal()
    
    matrix = np.array([
        [mt[1], mt[2], mt[0]],
        [mt[4], mt[5], mt[3]],
        [0, 0, 1]
    ])
    
    
    # matrix = np.array(coefficients_tuple).reshape((2, 3))
    # matrix = np.vstack([matrix, [0, 0, 1]])
    return(matrix)

def combineAffine(src_affine, dst_affine):
    return ~src_affine * dst_affine
    
def reprojectAffine(affine_obj, original_crs, target_crs):
    """Transform an affine object to a different CRS using pyproj.
    
    Parameters:
        affine_obj (affine.Affine): The original affine object to transform.
        original_crs (str): The original CRS of the affine object.
        target_crs (str): The target CRS for the transformed affine object.
    
    Returns:
        affine.Affine: The transformed affine object in the target CRS.
    """

    # CONVERT TO GEOTRANSFORM
    print("Inside function")
    geotrans = affine_to_geotransform(affine_obj)
    print(geotrans)
    
    # USE FUNCTIONS TO REPROJECT AND BACK TO AFFINE
    temp = reproject_geotransform(geotrans, original_crs, target_crs)
    print(temp)
    transformed_affine = get_affine_from_geotransform(temp)
    print(transformed_affine)
    
    return transformed_affine