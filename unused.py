# input: boxes of a car
def detect_car_spin(recent_bboxes, out_frame=None):
    # need at least 2 box,
    if len(recent_bboxes)<2:
        return False
    change_counter = 0
    result = False
    prev_offset = 0 # first frame
    prev_ratio, prev_rate, rate = None, None, None
    color = (0,255,0)
    dev0 ,x0, y0 = None, None, None #for display
    # For the future boxes
    for i in range(len(recent_bboxes)):
        
        left, top, right, bottom = recent_bboxes[i][0]
        width, height = right-left, bottom-top

        # determine if the car is in the center of the frame
        center_x = (left+right)//2
        distance = abs(center_x - vid_width//2)

        forward_ratio = (distance/vid_width) + 1.2
        side_ratio = 2.4
        ratio_thres = (forward_ratio + side_ratio)/2

        # compute the frame interval between the 2 boxes
        frame_offset = recent_bboxes[i][1]
        frame_diff = frame_offset - prev_offset

        # interval between two boxes >1, split the w/h difference 
        if frame_diff>1:
            width_change = (width-prev_width)/frame_diff
            height_change = (height-prev_height)/frame_diff
            # the 1st should be the biggest
            ratio = (prev_width+width_change)/(prev_height+height_change)
        else:
            ratio = width/height

        
        # check if the car change its direction too fast
        dev = ratio - ratio_thres

        if dev0 is None:#for display
            dev0 = dev
            x0, y0 = (left+right)//2, (top+bottom)//2

        if prev_ratio is not None:
            rate = ratio - prev_ratio
            if abs(rate) > 0.2:
                print(f"{ratio} - {prev_ratio}")
                result = True
                color = (255,0,0)
                break            

        if prev_rate is not None:
            # check if the car change its direction frequently
            if abs(rate)>0.05 and rate*prev_rate<0: #changed
                change_counter +=1
                if change_counter>1:
                    result = True
                    color = (0,0,255)
                    break

        # mark as prev
        prev_width, prev_height = width, height
        prev_offset = frame_offset
        if rate is not None:
            prev_rate = rate
        prev_ratio = ratio
    
    if out_frame is not None:
        cv2.putText(out_frame, f"{dev0:.2f}", (x0 , y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return result