"""
Main config

"""
# CMD
PRINT_INTERVAL_SEC = 10 #Time interval (of the video) between command line output

# Detection toggle
DET_CAR_COL = True
DET_CAR_PERSON_COL = False
DET_CLOSE_DIS = True
DET_JAYWALKER = True

# Traffic signs/lights
USE_SIGN_TO_DET_MOV = True #Use traffic light/sign to detect camera movement
SHOW_SIGN_MOVEMENT = False #display the value used when determine "center displacement| width difference| height difference "


# Labelling
# User only one of these:
PRINT_OBJ_ID = False
PRINT_CLASS_LABEL = False # E.g. "Car:1 0.78" near the bounding box


PRINT_ANOMALY_LABEL = False # E.g. "Close Distance" near the bounding box

# Object detection
OMIT_SMALL = True #omit small bounding boxes of car/person
OMIT_SIGN = False  #omit traffic signs / lights
assert not(OMIT_SIGN and USE_SIGN_TO_DET_MOV), "Change OMIT_SIGN to False if you want to use them to detect the camera movement"

PERSON_MISS_TOLERATE = 2 # No. of frame tolerated, when the detection is missing, before the person object is ignored. 


# Damage detection
DO_PADDING = True #extend the bounding box before cropping
DO_ERASING = False #erase portion of image if overlapped with another bbox
IGNORE_SMALL = True #Do not detect small car
SHOW_PROB = True # Display the prob in the center of the bbox

DMG_SKIP_DET = False #skip some checking
DMG_SKIP_BASE = 1 #number of frame to skip

DMG_THRES = [0.9, 0.95] # use in pairs, if the prob is > thres, skip the classification of that bbox by n frames
DMG_SKIP_NO = [6, 12]

USE_AVG_PROB = False
USE_ADJUSTED_PROB = True
PROB_PERIOD = 5
WEIGHTED_PROB = False 
WEIGHT_THRES = 0.8


"""
Other (constant used)

"""
# Width/height > IS_SIDE_RATIO : the side of the car is facing the camera 
SIDE_THRES = 1.6

# 
COLL_HEIGHT_THRES = 0.1
COLL_HEIGHT_THRES_STRICT = 0.05
# Camera Moving Detection

