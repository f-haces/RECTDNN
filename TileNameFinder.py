# NOTEBOOK IMPORTS
import warnings, re, traceback
import numpy as np
from datetime import datetime
from dateutil import parser
from fuzzywuzzy import process

# IMAGE IMPORTS
import cv2
from PIL import Image
from rapidfuzz.fuzz import partial_ratio_alignment


# CUSTOM UTILITIES
from IndexUtils import * 
from TileUtils import *
from ReadDate import extractDate

# TILED INFERENCE
import sahi

sahi.utils.cv.IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.tif']

Image.MAX_IMAGE_PIXELS = 933120000
warnings.filterwarnings("ignore")
initialize = False


def prepImageForReading(img):
    image = np.asarray(img, dtype=np.uint8)
    if np.max(image) < 255:
        image = image * 255
    return image

def resize_image(image, size=512):
    
    if image.ndim == 3:
        image = image[:, :, 0]

    # Get the current dimensions of the image
    height, width = image.shape
    
    # Calculate the scaling factor to resize the longest side to 512 pixels
    if height > width:
        scale_factor = size / height
    else:
        scale_factor = size / width
    
    # Compute new dimensions
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    
    # Resize the image using OpenCV
    resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    return resized_img

def extract_date(text):
    """
    # Step 1: Use regular expressions to look for date-like patterns
    # Looking for both numeric dates (e.g. 9/10/92) and written dates (e.g. April 2, 1990)
    date_regex = r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})|([A-Za-z]+\s+\d{1,2},\s+\d{4})'
    
    # Step 2: Search the text for any potential date patterns
    match = re.search(date_regex, text)
    
    if match:
        date_str = match.group(0)
        try:
            # Step 3: Parse the matched date string
            return parser.parse(date_str)
        except ValueError:
            pass  # If parsing fails, it will return None later on
    """
    # Step 4: Fallback: Try fuzzy parsing on the entire string if no match is found
    try:
        return parser.parse(re.sub(",", ", ", text).replace("|", ""), fuzzy=True)
    except:
        return None  # Return None if no valid date is found
    

'''
for i, (k, v) in tqdm(enumerate(gen_dict.items()), total=len(gen_dict.items())):
    if len(v['legend']) == 0:
        continue
    dates_found, text_found = extractDate(v['legend'][0]['data'], filter_values=['revised', 'effective'])

    print(dates_found)
    if dates_found is None:
        print(text_found)
'''    

def commonMisses(text):
    text = text.upper()
    text = re.sub('O', '0', text)
    text = re.sub('T', '1', text)
    text = re.sub('I', '1', text)
    text = re.sub('€', 'E', text)
    text = re.sub('Q', '0', text)
    text = re.sub('Z', '7', text)
    return text

def replace_last_letters(s, n=1):
    # Match any number followed by letters at the end of the string
    return re.sub(r'(\d)([a-zA-Z]+)$', lambda m: m.group(1) + m.group(2)[:n], s)

def prettifyName(text, key):
    if text[-2:] == " 8":
        text = text[:-2] + "B"
    text = text.replace(" ", "")
    text = re.sub('[^A-Za-z0-9_]+', '', text)
    text = re.sub(f'^.*?{key}', key, text)
    text = commonMisses(text)
    if text[-1] == '1':
        text = text[-1]
    score, s1_start, s1_end, s2_start, s2_end = partial_ratio_alignment(text, key)
    text = key + text[s1_end:s1_end+7] # HEURISTICALLY DETERMINED +7: 1 for prescript, 4 for ID, 1 for postcripst, 1 for slicing
    text = replace_last_letters(text)
    return text

def secondaryNameFinding(image_text, bbox, key, found_threshold, reader, verbose=0):
    cropped_image = image_text[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    if cropped_image.size == 0:
        return None, texts
    texts = reader.readtext(cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR), detail=0, width_ths=50)

    # REMOVE ONLY LETTERS
    texts = [s for s in texts if any(char.isdigit() for char in s)]

    # REMOVE ANY SPECIAL CHARACTERS
    texts = [re.sub('[^A-Za-z0-9_]+', '', text) for text in texts if len(re.sub('[^A-Za-z0-9_]+', '', text)) > 0]
    len_texts = [commonMisses(a) for a in texts if len(a) >= len(key)]

    # IF WE HAVE SOME TEXTS LONGER THAN KEY, THEN WE FOUND SOMETHING
    if not len(len_texts) == 0:
        text, val = process.extractOne(key, len_texts)
        if verbose > 1:
            print(text, val, len_texts)
        if val > found_threshold:
            return text, len_texts
        
        maybetext = max(len_texts, key=len)
        maybe = re.findall(r'\d{4}[a-zA-Z]', maybetext)
        maybe.extend(re.findall(r'\d{4}', maybetext))
        if len(maybe) > 0:
            if verbose > 1:
                print(f"Inferring Key: {key+maybe[0]}" )
            return key+maybe[0], len_texts
        
    # IF ALL ARE SHORTER THAN KEY, PROBABLY NO KEY PRINTED SO
    elif len(texts) > 0:
        text = commonMisses(''.join(texts))
        if verbose > 1:
            print("Could not find key " +  key+"_"+text)
        return key+"_"+text, texts

    return None, texts

def smartifyDict(gen_dict, basepath, reader,
                 found_threshold=60,
                 verbose=5,
                 smart_dict = None,
                 diagnostics=False,
                 ):
    """
    gen_dict  = From previous reading
    index_image = from CV2 reading

    """

    unsureCounter = 0
    totalCounter  = 0
    if smart_dict is None:
        smart_dict = {}

    for i, (k, v) in tqdm(enumerate(gen_dict.items()), total=len(gen_dict.items())):
        index_image = cv2.imread(os.path.join(basepath, k))
        key         = findIndexKey(k)

        if index_image.ndim == 3:
            index_image = index_image[:, :, 0]
        #kp, TPNN    = findKeypoints(index_image, model=TPNN)
        kp = v['DNN_outputs']['classifications']
        mask = cv2.dilate(kp[:, :, 2], (40, 40)) > 0
        image_text = cv2.GaussianBlur((255 - index_image) * mask, (11, 11), 1, 1)
        affine      = Affine(*v["output_transform"].flatten()[:6])
        
        if len(v['legend']) != 0:
            dates_found, _ = extractDate(v['legend'][0]['data'], reader, filter_values=['revised', 'effective'])
        else:
            dates_found, _ = None, None
        
        for ii, (kk, vv) in tqdm(enumerate(v['tile'].items()), total=len(v['tile'].items()), leave=False):
            try:
                totalCounter = totalCounter + 1
                image = prepImageForReading(vv['data'])
                texts = reader.readtext(cv2.cvtColor(resize_image(image, size=1024), cv2.COLOR_GRAY2BGR), detail=0, width_ths=50)
                texts = [a for a in texts if len(a) >= len(key)]
                bbox = vv['bbox']
                if not len(texts) == 0:
                    text, val = process.extractOne(key, texts)
                else: 
                    val = 0
                
                if val > found_threshold:
                    text = prettifyName(text, key)
                else:
                    if verbose > 1:
                        print("Attempting Secondary")
                    secondText, secondtextsfound = secondaryNameFinding(image_text, bbox.astype(np.int32), key, found_threshold, reader, verbose=verbose)
                    if secondText is None:
                        text = f"Unsure{unsureCounter}"
                        if verbose > 1:
                            print(secondtextsfound, vv['confidence'])
                        unsureCounter = unsureCounter + 1
                    else:
                        text = prettifyName(secondText, key)
                        if verbose > 1:
                            print(text, secondtextsfound)
    
                # CALCULATE COORDS FROM AFFINE 
                left, bottom = affine * (bbox[0], bbox[1])
                right, top   = affine * (bbox[2], bbox[3])

                # SAVE OUTPUT
                if text[-1].isalpha():
                    tilename = text[:-1]
                    inner_key = text[-1]
                else:
                    tilename = text
                    inner_key = 0
                
                currKeyCheck = smart_dict.get(tilename, None)
                if currKeyCheck is None:
                    smart_dict[tilename] = {}

                currInnerKeyCheck = smart_dict[tilename].get(inner_key, None)
                if currInnerKeyCheck is None:
                    smart_dict[tilename][inner_key] = {}
                    smart_dict[tilename][inner_key]['coords'] = []
                    smart_dict[tilename][inner_key]['indexes'] = []
                    smart_dict[tilename][inner_key]['pix'] = []
                    smart_dict[tilename][inner_key]['dates'] = []
                
                smart_dict[tilename][inner_key]['pix'].append(bbox)
                smart_dict[tilename][inner_key]['coords'].append(np.array([left, bottom, right, top]))
                smart_dict[tilename][inner_key]['indexes'].append(k)

                if "Unsure" in tilename:
                    smart_dict[tilename][0]['data'] = image
                    smart_dict[tilename][0]['confidence'] = vv['confidence']
                    smart_dict[tilename][0]['initialtexts'] = texts
                    smart_dict[tilename][0]['secondtexts'] = secondtextsfound
                elif diagnostics:
                    smart_dict[tilename][inner_key]['data'] = image
                    smart_dict[tilename][inner_key]['confidence'] = vv['confidence']
                    smart_dict[tilename][inner_key]['initialtexts'] = texts
                    # smart_dict[tilename][inner_key]['secondtexts'] = secondtextsfound

                if dates_found is None:
                    date = None
                else:
                    date = dates_found[0]
                smart_dict[tilename][inner_key]['dates'].append(date)
            
            except Exception:
                print(traceback.format_exc())
                continue
        if verbose > 0:
            print(f"{len(smart_dict)} Individual Tiles | Not Found: {unsureCounter} / {totalCounter} | Finished Processing: {k}")
        
    return smart_dict
    """
    try:
        with open("SmartDict.p", "wb" ) as f:
            pickle.dump(smart_dict, f)
    except Exception:
        print(traceback.format_exc())
        continue"""
