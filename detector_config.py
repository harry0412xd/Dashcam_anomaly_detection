# Width/height >1.5 : the side of the car is facing the camera 
IS_SIDE_RATIO = 1.5


# ---------------------------------------
# ------Collision
# ignore car with its height less than vid_height/this num
COLL_IGNORE_DENOMINATOR = 18

# 
COLL_HEIGHT_THRES = 0.1

IOU_FALSE_THERS = 0.6
# Camera Moving Detection


# Distance Detection
def get_roi_for_distance_detect(vid_width, vid_height):
    DIS_ROI_POINTS = [(vid_width//2, vid_height//2), 
                    (vid_width//8, vid_height),
                    (vid_width*7//8, vid_height)]
    return DIS_ROI_POINTS

            