import re, cv2
import numpy as np
from fuzzywuzzy import process
from dateutil import parser


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
    

def extractDate(img, reader, filter_values=None, verbose=False, year_cutoff=2020, ini_year=1965, imsize=1024):
    try:
        image = prepImageForReading(img)
        texts = reader.readtext(cv2.cvtColor(resize_image(image, size=1024), cv2.COLOR_GRAY2BGR), detail=0, width_ths=50)
        texts_search = [s if re.search(r'[a-zA-Z]', s) else "" for s in texts]
        dates = [extract_date(a) for a in texts_search]

        if len(dates) == 0:
            return None, texts

        if filter_values is None:
            return dates, texts
        
        filtered_dates = []
        for a in np.where(np.array(dates) != None)[0].tolist():
            for b in filter_values:
                _, v = process.extractOne(b, texts[a-1].split(" "))
                if verbose:
                    print(texts[a], texts[a-1], b, v)
                # v = np.max([a, v])        
                if v > 70:
                    filtered_dates.append(dates[a])

        filtered_dates = [a for a in filtered_dates if a.year < year_cutoff]
        filtered_dates = [a for a in filtered_dates if a.year > ini_year]

        if len(filtered_dates) == 0:
            return None, texts

        return filtered_dates, texts
    except:
        return None, None
    

def extractDatesFromDict(mydict, keys, max_difference=3650):
    if not keys in list(mydict.keys()):
        print(f"Could not find key {keys} for current tile from options {list(mydict.keys())}, using closest date from all keys")
        keys=0

    if keys == 0:
        keys = list(mydict.keys())
    
    if not isinstance(keys, list):
        keys = [keys]

    out_dates = {}
    for key in keys:
        # for i in range(len(mydict[key]['dates'])):
        #     out_dates[mydict[key]['dates'][i]] = mydict[key]['coords'][i]
        out_dates[key] = mydict[key]
    
    return out_dates

def extractDatesFromDict_legacy(mydict, keys, max_difference=3650):
    if not keys in list(mydict.keys()):
        print(f"Could not find key {keys} for current tile from options {list(mydict.keys())}, using closest date from all keys")
        keys=0

    if keys == 0:
        keys = list(mydict.keys())
    
    if not isinstance(keys, list):
        keys = [keys]

    out_dates = {}
    for key in keys:
        for i in range(len(mydict[key]['dates'])):
            out_dates[mydict[key]['dates'][i]] = mydict[key]['coords'][i]
    
    return out_dates

def getBBOXClosestDate(date_dict, reference_date):
    # Find the key with the minimum difference to the reference_date
    closest_date = min(date_dict.keys(), key=lambda date: abs(date - reference_date))
    # Return the value associated with this closest date
    print(f"Using {closest_date} for tile date {reference_date}")
    return date_dict[closest_date], closest_date
