import os, pickle, random

import numpy as np
from PIL import Image

from tqdm.autonotebook import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as T



class DetectionLoss(nn.Module):

    def __init__(self, num_classes):
        super(DetectionLoss, self).__init__()
        self.num_classes = num_classes
        self.regression_loss = nn.SmoothL1Loss(reduction='sum')
        self.classification_loss = nn.CrossEntropyLoss(reduction='sum')
        self.objectness_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, predictions, targets):
        # Unpack predictions and targets
        pred_boxes = predictions['boxes']
        pred_scores = predictions['scores']
        pred_labels = predictions['labels']
        
        gt_boxes = targets['boxes']
        gt_labels = targets['labels']
        
        # Match predicted boxes with ground truth boxes
        iou = box_iou(pred_boxes, gt_boxes)
        matched_gt_indices = iou.argmax(dim=1)  # Index of the ground truth box with highest IoU for each predicted box
        
        # Calculate bounding box regression loss only for matched boxes
        regression_loss = self.regression_loss(pred_boxes, gt_boxes[matched_gt_indices])
        
        # Calculate classification loss only for matched boxes
        classification_loss = self.classification_loss(pred_labels, gt_labels[matched_gt_indices])
        
        # Calculate objectness loss only for matched boxes
        objectness_loss = self.objectness_loss(pred_scores, torch.ones_like(matched_gt_indices, dtype=torch.float32))
        
        # Aggregate losses
        total_loss = regression_loss + classification_loss + objectness_loss
        
        return total_loss


class BBNN_dataset(Dataset):
    def __init__(self, image_dir, prediction_dir, transform=None):
        self.image_dir = image_dir
        self.prediction_dir = prediction_dir
        self.image_filenames = os.listdir(self.image_dir)
        self.transform = transform
        self.tensor = T.Compose([T.ToTensor()])
        
        # Load all images and predictions into memory
        self.data = []
        for image_filename in self.image_filenames:
            image_path = os.path.join(self.image_dir, image_filename)
            image = Image.open(image_path).convert('L')
            
            prediction_path = os.path.join(self.prediction_dir, image_filename[:-3] + 'pkl')
            
            with open(prediction_path, 'rb') as f:
                predictions = pickle.load(f)
            
            image_arry = np.asarray(image)
            
            masks = []
            for pred in predictions['boxes']:
                masks.append(self.tensor(self.create_mask(np.array(pred), image_arry)))
                
            image_arry = self.tensor(image_arry)
            
            predictions['boxes'] = np.array(predictions['boxes'])
            
            self.data.append({
                    'image'       : image, 
                    'image_arry'  : image_arry,
                    'predictions' : predictions,
                    'masks'       : masks
                })
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        working_data = self.data[idx].copy()
        
        # Apply transformations
        if self.transform:
            return self.transform(working_data.copy())
        else:
            return working_data

    def create_mask(self, bb, x):
        # BORROWED FROM https://jovian.ml/aakanksha-ns/road-signs-bounding-box-prediction
        """Creates a mask for the bounding box of same shape as image"""
        rows,cols,*_ = x.shape
        Y = np.zeros((rows, cols), dtype=np.byte)
        bb = bb.astype(np.int32)
        Y[bb[1]:bb[3], bb[0]:bb[2],] = 1.
        return Y
  
class RandomApplyToStack(torch.nn.Module):
    def __init__(self, transforms):
       super().__init__()
       self.transforms = transforms
    
    def mask_to_BB(self, Y):
        # BORROWED FROM https://jovian.ml/aakanksha-ns/road-signs-bounding-box-prediction
        cols, rows = np.nonzero(Y[0, :, :].numpy())
        if len(cols)==0: 
            return np.zeros(4, dtype=np.float32)
        top_row = np.min(rows)
        left_col = np.min(cols)
        bottom_row = np.max(rows)
        right_col = np.max(cols)
        # [left_col, top_row, right_col, bottom_row]
        return [left_col, bottom_row, right_col, top_row]

    def mask_to_BB_nonworking(self, A):
        # BORROWED FROM https://stackoverflow.com/questions/4808221/is-there-a-bounding-box-function-slice-with-non-zero-values-for-a-ndarray-in
        B = torch.nonzero(A)
        print(B.shape)
        (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1 
        return [xstart, ystart, xstop, ystop]

    def create_mask(self, bb, x):
        # BORROWED FROM https://jovian.ml/aakanksha-ns/road-signs-bounding-box-prediction
        """Creates a mask for the bounding box of same shape as image"""
        rows,cols,*_ = x.shape
        Y = np.zeros((rows, cols))
        bb = bb.astype(np.int)
        Y[bb[0]:bb[2], bb[1]:bb[3]] = 1.
        return Y
       
    def __call__(self, sample):
    
        # CREATE STACK OF IMAGES TO TRANSFORM
        imgs = [sample['image_arry']]
        imgs.extend(sample['masks'])
        imgs = torch.stack(imgs)
        print(imgs.shape)
        
        # TRANSFORM LIST OF IMAGES
        imgs = imgs.to("cuda")
        out  = self.transforms(imgs)
        out  = out.to("cpu")
        torch.cuda.empty_cache()
        
        output_sample = {}
        output_sample['image']  = out[0, :, :, :]
        output_sample['masks']       = out[1:, : , :, :]
        output_sample['predictions'] = {
                "boxes" : torch.Tensor([self.mask_to_BB(a) for a in out[1:, : , :, :]]),
                "labels": sample["predictions"]["labels"].copy()
            }
        
        return output_sample
    
    def forward(self, sample):
        return self.__call__(sample)
   
  
class RandomApplyToList(torch.nn.Module):
    def __init__(self, transforms):
       super().__init__()
       self.transforms = transforms
       
    def transformList(self, imgs):
        seed = np.random.randint(2147483647)
        out = []
        
        for img in imgs:
        
            '''
            img = img
            random.seed(seed)
            torch.manual_seed(seed)
            out.append(self.transforms(img))
            '''
            
            img = img.to("cuda")
            random.seed(seed)
            torch.manual_seed(seed)
            out.append(self.transforms(img).to("cpu"))
            torch.cuda.empty_cache()
            
            
        return out
    
    def mask_to_BB(self, Y):
        # BORROWED FROM https://jovian.ml/aakanksha-ns/road-signs-bounding-box-prediction
        cols, rows = np.nonzero(Y[0, :, :].numpy())
        if len(cols)==0: 
            return np.zeros(4, dtype=np.float32)
        top_row = np.min(rows)
        left_col = np.min(cols)
        bottom_row = np.max(rows)
        right_col = np.max(cols)
        # [left_col, top_row, right_col, bottom_row]
        return [left_col, bottom_row, right_col, top_row]

    def mask_to_BB_nonworking(self, A):
        # BORROWED FROM https://stackoverflow.com/questions/4808221/is-there-a-bounding-box-function-slice-with-non-zero-values-for-a-ndarray-in
        B = torch.nonzero(A)
        print(B.shape)
        (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1 
        return [xstart, ystart, xstop, ystop]

    def create_mask(self, bb, x):
        # BORROWED FROM https://jovian.ml/aakanksha-ns/road-signs-bounding-box-prediction
        """Creates a mask for the bounding box of same shape as image"""
        rows,cols,*_ = x.shape
        Y = np.zeros((rows, cols))
        bb = bb.astype(np.int)
        Y[bb[0]:bb[2], bb[1]:bb[3]] = 1.
        return Y
       
    def __call__(self, sample):
    
        # CREATE LIST TO TRANSFORM
        imgs = [sample['image_arry']]
        imgs.extend(sample['masks'])
        
        # TRANSFORM LIST OF IMAGES
        out  = self.transformList(imgs)
        
        output_sample = {}
        output_sample['image']  = out[0]
        output_sample['masks']       = out[1:]
        output_sample['predictions'] = {
                "boxes" : torch.Tensor(np.array([self.mask_to_BB(a) for a in out[1:]])),
                "labels": sample["predictions"]["labels"].copy()
            }
        
        return output_sample
        
    

'''        
class myTransforms(nn.Module): 
    def __init__(self):
        super().__init__()
        self.transforms = torch.nn.Sequential(
             T.RandomRotation(30),
             T.RandomCrop((2048, 2048))
        )
    def forward(self, x):
        return self.transforms(x)
    
scripted_myTransforms = torch.jit.script(myTransforms())
transform = RandomApplyToStack(scripted_myTransforms)
test_scripted = BBNN_dataset(image_dir, pred_dir, transform=transform)

class myTransforms(nn.Module): 
    def __init__(self):
        super().__init__()
        self.transforms = torch.nn.Sequential(
             T.RandomRotation(30),
             T.RandomCrop((2048, 2048))
        )
    def forward(self, x):
        return self.transforms(x)
    
scripted_myTransforms = torch.jit.script(myTransforms())
transform = RandomApplyToStack(scripted_myTransforms)
test_scripted = BBNN_dataset(image_dir, pred_dir, transform=transform)

class RandomApplyToList(torch.nn.Module):
    def __init__(self, transforms):
       super().__init__()
       self.transforms = transforms

    def __call__(self, imgs):
        seed = np.random.randint(2147483647)
        out = []
        
        for img in imgs:
            random.seed(seed)
            torch.manual_seed(seed)
            out.append(self.transforms(img))
        return out
        
        
class myTransform(object):
    
    def __init__(self, crop_size, max_angle):
        self.crop_size = crop_size
        self.max_angle = max_angle
    
    def random_flip(self, image, predictions):
        
        # Randomly flip horizontally
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            predictions['boxes'][:, [0, 2]] = image.width - predictions['boxes'][:, [2, 0]]

        return image, predictions
    
    def rotate_point(self, x, y, cx, cy, angle):
        # Rotate point (x, y) around center (cx, cy) by angle degrees
        angle_rad = np.radians(angle)
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        nx = cos_theta * (x - cx) - sin_theta * (y - cy) + cx
        ny = sin_theta * (x - cx) + cos_theta * (y - cy) + cy
        return nx, ny
    
    def random_rotation(self, img, preds):
        
        # Randomly select rotation angle
        angle = random.uniform(-self.max_angle, self.max_angle)

        # Perform rotation on both image and bounding boxes
        img = img.rotate(angle, expand=False)

        # Rotate bounding box coordinates
        cx = image.width / 2
        cy = image.height / 2
        
        boxes = []
        
        for box in preds['boxes']:
            x1, y1, x2, y2 = box
            x1, y1 = self.rotate_point(x1, y1, cx, cy, angle)
            x2, y2 = self.rotate_point(x2, y2, cx, cy, angle)
            boxes.append(x1, y1, x2, y2)
            
        preds['boxes'] = np.array(boxes)

        return img, preds
    
    def random_crop(self, image, predictions):
                
        # Randomly select top-left corner for cropping
        left = random.randint(0, image.width - self.crop_size)
        top = random.randint(0, image.height - self.crop_size)

        # Perform cropping on both image and bounding boxes
        image = image.crop((left, top, left + self.crop_size, top + self.crop_size))
        
        predictions['boxes'][:, [0, 2]] -= left
        predictions['boxes'][:, [1, 3]] -= top
        
        return image, predictions
    
        
    def __call__(self, sample):
        
        working_sample = sample.copy()
        
        image, predictions = working_sample['image'].copy(), working_sample['predictions'].copy()
        
        if random.random() < 0.5:
            image, predictions = self.random_crop(image, predictions)
        if random.random() < 0.0:
            image, predictions = self.random_rotation(image, predictions)
        if random.random() < 0.0:
            image, predictions = self.random_flip(image, predictions)
        return image, predictions
        
        
class RandomRotationWithBBoxes(object):
    def __init__(self, degrees, resample=False, expand=False, center=None):
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    def __call__(self, sample):
        image = sample['image']
        predictions = sample['predictions']

        angle = random.uniform(-self.degrees, self.degrees)
        image = transforms.functional.rotate(image, angle, self.resample, self.expand, self.center)

        if 'boxes' in predictions:
            boxes = predictions['boxes']
            h, w = image.size[::-1]
            cx = w / 2
            cy = h / 2

            # Convert PIL image to numpy array for easier manipulation
            np_image = np.array(image)

            # Rotate bounding boxes
            corners = np.stack([
                [boxes[:, 0], boxes[:, 1]],
                [boxes[:, 2], boxes[:, 1]],
                [boxes[:, 0], boxes[:, 3]],
                [boxes[:, 2], boxes[:, 3]]
            ], axis=0).astype(np.float64)

            # Move origin to image center
            corners[0] -= cx
            corners[1] -= cy

            # Apply rotation
            rotation_matrix = np.array([
                [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))],
                [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]
            ])
            print(rotation_matrix.shape)
            print(corners.shape)
            rotated_corners = np.matmul(rotation_matrix, corners)

            # Move origin back
            rotated_corners[0] += cx
            rotated_corners[1] += cy

            # Update box coordinates
            new_x1 = np.min(rotated_corners[0], axis=0)
            new_y1 = np.min(rotated_corners[1], axis=0)
            new_x2 = np.max(rotated_corners[0], axis=0)
            new_y2 = np.max(rotated_corners[1], axis=0)

            # Update box coordinates
            boxes[:, 0] = new_x1
            boxes[:, 1] = new_y1
            boxes[:, 2] = new_x2
            boxes[:, 3] = new_y2

            predictions['boxes'] = torch.tensor(boxes)

        return {'image': image, 'predictions': predictions}
        
        
        
        
        
        
        
        
        
        
        
        
        
# BORROWED FROM https://jovian.ml/aakanksha-ns/road-signs-bounding-box-prediction
def create_mask(bb, x):
    """Creates a mask for the bounding box of same shape as image"""
    rows,cols,*_ = x.shape
    Y = np.zeros((rows, cols))
    bb = bb.astype(np.int)
    Y[bb[0]:bb[2], bb[1]:bb[3]] = 1.
    return Y

def mask_to_bb(Y):
    """Convert mask Y to a bounding box, assumes 0 as background nonzero object"""
    cols, rows = np.nonzero(Y)
    if len(cols)==0: 
        return np.zeros(4, dtype=np.float32)
    top_row = np.min(rows)
    left_col = np.min(cols)
    bottom_row = np.max(rows)
    right_col = np.max(cols)
    return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)

def resize_image_bb(read_path,write_path,bb,sz):
    """Resize an image and its bounding box and write image to new path"""
    im = read_image(read_path)
    im_resized = cv2.resize(im, (int(1.49*sz), sz))
    Y_resized = cv2.resize(create_mask(bb, im), (int(1.49*sz), sz))
    new_path = str(write_path/read_path.parts[-1])
    cv2.imwrite(new_path, cv2.cvtColor(im_resized, cv2.COLOR_RGB2BGR))
    return new_path, mask_to_bb(Y_resized)

def create_bb_array(x):
    """Generates bounding box array from a train_df row"""
    return np.array([x[5],x[4],x[7],x[6]])
# modified from fast.ai
def crop(im, r, c, target_r, target_c): 
    return im[r:r+target_r, c:c+target_c]

# random crop to the original size
def random_crop(x, r_pix=8):
    """ Returns a random crop"""
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    return crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)

def center_crop(x, r_pix=8):
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    return crop(x, r_pix, c_pix, r-2*r_pix, c-2*c_pix)
        
def rotate_cv(im, deg, y=False, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
    """ Rotates an image by deg degrees"""
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c/2,r/2),deg,1)
    if y:
        return cv2.warpAffine(im, M,(c,r), borderMode=cv2.BORDER_CONSTANT)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)

def random_cropXY(x, Y, r_pix=8):
    """ Returns a random crop"""
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    xx = crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)
    YY = crop(Y, start_r, start_c, r-2*r_pix, c-2*c_pix)
    return xx, YY

def transformsXY(path, bb, transforms):
    x = cv2.imread(str(path)).astype(np.float32)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255
    Y = create_mask(bb, x)
    if transforms:
        rdeg = (np.random.random()-.50)*20
        x = rotate_cv(x, rdeg)
        Y = rotate_cv(Y, rdeg, y=True)
        if np.random.random() > 0.5: 
            x = np.fliplr(x).copy()
            Y = np.fliplr(Y).copy()
        x, Y = random_cropXY(x, Y)
    else:
        x, Y = center_crop(x), center_crop(Y)
    return x, mask_to_bb(Y) '''
        