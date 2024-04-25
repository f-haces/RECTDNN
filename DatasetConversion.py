import os, cv2, json, pickle, glob
import numpy as np
from tqdm.autonotebook import tqdm

def predictToRetrain(model, input_dir, pkl_dir, coco_dir,
                     target_size=None, 
                     original_shapes=None, 
                     category_labels=None,
                     categories=None
                     ):
    if target_size is None:
        target_size = 1024

    if original_shapes is None:
        original_shapes = []
    
    if category_labels is None:
        category_labels = {
            0 : "County",
            1 : "Tile",
            2 : "Box",
            3 : "Legend"
        }

    if categories is None:
        categories=[0, 1]

    original_shapes

    results = model(glob.glob(input_dir+"\\*"), imgsz=target_size,)

    createPickleDataset(results, pkl_dir)
    cocodata = toCOCO(coco_dir, pkl_dir, f"{coco_dir}/annotations", "", category_labels, categories)

    
def createPickleDataset(results, pkl_dir):
    # pkl_dir = f"{output_dataset_dir}/pkl"
    os.makedirs(pkl_dir, exist_ok=True)

    for i, result in enumerate(results[0]):

        fn = os.path.basename(result[0].path)
        
        curr_dict = {'labels' : result.boxes.cls.to('cpu').numpy().tolist(), 'boxes' : []}

        boxes = result.boxes.xyxy.cpu().numpy()
        # print(boxes)
        boxes[:, [0, 2]] = boxes[:, [0, 2]]
        boxes[:, [1, 3]] = boxes[:, [1, 3]]
        # print(boxes)
        curr_dict['boxes'] = np.vstack([boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 0]]).T
        
        
        with open(os.path.join(pkl_dir, fn[:-3]+"pkl"), 'wb') as handle:
            pickle.dump(curr_dict, handle)


def toCOCO(images_folder, pkl_folder, output_file, relative_image_path, category_labels, categories):
    
    def convertBBOXFormats(bbox):
        y_min, x_max, y_max, x_min = bbox
        width = x_max - x_min
        height = y_max - y_min
        
        if x_min+y_min+x_max+y_max != 0 and True:
            print(bbox)
            print(x_min, y_max, width, height)
            
        return x_min, y_min, width, height
        
    
    def convert(o):
        if isinstance(o, np.generic): return o.item()  
        raise TypeError
        
    # LIST FILES 
    fns       = os.listdir(images_folder)
    image_fns = [os.path.join(images_folder, fn) for fn in fns]
    pkl_fns   = [os.path.join(pkl_folder, fn[:-3]+"pkl") for fn in fns]
    
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Map category names to IDs
    category_id_map = {}
    for idx, category in enumerate(categories):
        category_id_map[category_labels[category]] = idx + 1  # COCO categories are 1-indexed
    
    # Add categories to COCO data
    for category in categories:
        coco_data['categories'].append({
            'id': category_id_map[category_labels[category]],
            'name': category_labels[category],
            'supercategory': ""
        })
    
    # Process images and annotations
    image_id = 1
    annotation_id = 1
    for i, image_path in tqdm(enumerate(image_fns), total=len(image_fns)):
        filename = fns[i]
    
        # READ IMAGE
        image = cv2.imread(image_path)
        height, width, _ = image.shape
    
        # READ PICKLE FILE
        pkl = pickle.load(open(pkl_fns[i], 'rb'))
    
        # Add image to COCO data
        coco_data['images'].append({
            'id': image_id,
            'file_name': f"{relative_image_path}{filename}",
            'width': width,
            'height': height
        })
    
        count = 0
    
        # Add annotations to COCO data
        for i, bbox in enumerate(pkl['boxes']):
    
            label = pkl["labels"][i]
    
            x, y, w, h = convertBBOXFormats(bbox)
    
            if x + y + w + h == 0:
                continue
    
            count = count + 1
            
            coco_data['annotations'].append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': category_id_map[category_labels[label]],
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,  # assuming annotations are not crowd
                'segmentation' : [],
            })
            annotation_id += 1
    
        image_id += 1
        
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, default=convert)

    return coco_data