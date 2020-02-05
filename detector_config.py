    

# Camera Moving Detection


# Distance Detection
def get_roi_for_distance_detect(vid_width, vid_height):
    DIS_ROI_POINTS = [(vid_width//2, vid_height//2), 
                    (vid_width//8, vid_height),
                    (vid_width*7//8, vid_height)]
    return DIS_ROI_POINTS

            