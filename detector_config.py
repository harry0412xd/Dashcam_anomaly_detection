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

# Labelling
PRINT_CLASS_LABEL = False # E.g. "Car:1 0.78" near the bounding box
PRINT_ANOMALY_LABEL = False # E.g. "Close Distance" near the bounding box

# Object detection
OMIT_SMALL = True #omit small bounding boxes of car/person
OMIT_SIGN = True #omit traffic signs / lights

# Damage detection
DO_PADDING = True
DO_ERASING = True
IGNORE_SMALL = True











"""
Other (constant used)

"""
# Width/height > IS_SIDE_RATIO : the side of the car is facing the camera 
IS_SIDE_RATIO = 1.6

# ---------------------------------------
# ------Collision
# ignore car with its height less than vid_height/this num
COLL_IGNORE_DENOMINATOR = 18

# 
COLL_HEIGHT_THRES = 0.1
COLL_HEIGHT_THRES_STRICT = 0.05

IOU_FALSE_THERS = 0.6
# Camera Moving Detection
