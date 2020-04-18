# This file contains all the unused code
# which may be abandoned due to poor result
# or time constraint


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

# car driving toward camera
def get_cdtc_list(frame_infos, car_list):
    cdtc_list = []
    for car in car_list:
        obj_id, _ = car
        future_bboxes = get_bboxes_by_id(frame_infos, obj_id)

        prev_width = None
        count, total = 0, 0
        for (bbox, _) in future_bboxes:
            width = bbox[2]-bbox[0]
            center = (bbox[2]+bbox[0])//2
            if prev_width is not None:
                total += 2
                if prev_width>width:
                    count += 1
                if center>prev_center:
                    count += 1
            else:
                bbox0 = bbox
            prev_width = width
            prev_center = center

        if total>0 and count/total > 0.5:
            cdtc_list.append((obj_id, bbox0))
    return cdtc_list


# if the person is in front of a car
# the car is driving toward camera
def detect_car_person_collison(id_to_info, cdtc_list):
    results =[]
    for person_id in id_to_info:
        cls_id, _, person_bbox = id_to_info[person_id]
        if class_names[cls_id] == 'person':
            person_height = person_bbox[3]-person_bbox[1]
            for (car_id, car_bbox) in cdtc_list:
                car_height = car_bbox[3]-car_bbox[1]
                if person_height>car_height:
                   diff_percent = (person_height-car_height)/car_height
                   if diff_percent<0.2:
                      #  check overlapped% of the person by the car
                        prop = compute_overlapped(person_bbox, car_bbox)
                        if prop>0.6:
                            results.append(person_id)
                            results.append(car_id)
    return results

def detect_car_person_collison_new(car_list, person_list, out_frame=None):
    results = []
    for person_id in person_list:
        person_bboxes = get_bboxes_by_id(person_id)
        for car_id in car_list:
            car_bboxes = get_bboxes_by_id(car_id)

            i ,j = 0, 0
            while i<len(person_bboxes) and j<len(car_bboxes):
                person_bbox, person_offset = person_bboxes[i]
                car_bbox, car_offset = car_bboxes[j]
                if person_offset==car_offset:
                    car_depth = estimate_depth_by_width(car_bbox, True)
                    person_depth = estimate_depth_by_width(person_bbox, False)

                    if abs(car_depth-person_depth)<5:
                        car_center_x, car_center_y = (car_bbox[2]+car_bbox[0])//2, (car_bbox[3]+car_bbox[1])//2
                        person_center_x, person_center_y = (person_bbox[2]+person_bbox[0])//2, (person_bbox[3]+person_bbox[1])//2
                        dist = euclidean_distance(car_center_x, person_center_x, car_center_y, person_center_y)
                        person_width = person_bbox[2]-person_bbox[0]
                        if dist< 2.5*person_width:
                            results.append(car_id)
                            results.append(person_id)
                    i += 1
                    j += 1 
                elif person_offset<car_offset:
                    i += 1
                elif person_offset>car_offset:
                    j += 1  
    return results

def detect_jaywalker(recent_bboxes, mean_shift, out_frame=None):
    # ROI = [(vid_width//10,vid_height), (vid_width//2,vid_height*3//7), (vid_width*9//10, vid_height) ]
    ROI = [(0,vid_height), (vid_width//2,vid_height*5//14), (vid_width, vid_height) ]
    y_thers_close = int(vid_height*0.65)
    y_thers_medium = int(vid_height*0.45)

    # draw demo line
    if out_frame is not None:
        cv2.polylines(out_frame, [np.array(ROI, dtype=np.int32)], False, (255,0,0))
        cv2.line(out_frame,(0, y_thers_close), (vid_width, y_thers_close), (255,0,0))
        cv2.line(out_frame,(0, y_thers_medium), (vid_width, y_thers_medium), (255,0,0))

    left, top, right, bottom = recent_bboxes[0][0]
    center_x, center_y = (left+right)//2, (top+bottom)//2

    # in checking range
    if bottom > y_thers_medium:
        if inside_roi(center_x, bottom, ROI):
            if bottom > y_thers_close :
                return True
            else:
                dist, max_dist = 0, 0
                for i in range(len(recent_bboxes)-1):
                    left, top, right, bottom = recent_bboxes[i+1][0]
                    cx, cy = (left+right)//2, (top+bottom)//2
                    
                    mean = mean_shift[0] if cx<vid_width//2 else mean_shift[1]
                    mean = 0
                    dist0 = cx - center_x
                    if dist0 < 0 : #box is moving left
                        dist += min(dist0 - mean, 0)
                    elif dist0 > 0 : #box is moving right
                        dist += max(dist0 - mean, 0)
                    if abs(dist)>max_dist:
                        max_dist = abs(dist)
                # if out_frame is not None:
                #     cv2.putText(out_frame, f"{(max_dist/vid_width):.2f} ", (center_x-10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                if max_dist > vid_width*0.4:
                  return True
    return False


# get mean bbox shift
def get_mean_shift(frames_infos, out_frame):
    # seperate frame by left/right portion
    lp_shift_list, rp_shift_list = [], []
    lp_left_count, lp_right_count= 0, 0
    rp_left_count, rp_right_count= 0, 0

    id_list = [*frames_infos[0]] #what id to look for
    for obj_id in id_list:
        i = 1
        while i< len(frames_infos) and not obj_id in frames_infos[i]: #find the next frame containing obj 
            i += 1
        if i == len(frames_infos): #not found
            continue
        _, _, box_cur = frames_infos[0][obj_id]
        _, _, box_next = frames_infos[i][obj_id]
        left, top, right, bottom = box_cur
        left_next, top_next, right_next, bottom_next = box_next
        x_diff = (right_next+left_next)//2 - (right+left)//2 
        x_diff = x_diff/i  #diff per frame
        # print(f"Obj {obj_id}: {x_diff}")
        # cv2.putText(out_frame, f"{x_diff:.2f} ", ((right+left)//2, (top+bottom)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        if (right+left)>vid_width: #Object in right portion
            if x_diff>0 :
                rp_right_count += 1
            elif x_diff<0 :
                rp_left_count += 1
            if not x_diff==0:
                rp_shift_list.append(x_diff)
        else:
            if x_diff>0 :
                lp_right_count += 1
            elif x_diff<0 :
                lp_left_count += 1
            if not x_diff==0:
                lp_shift_list.append(x_diff)

    left_mean = cal_weighted_mean(lp_shift_list, lp_left_count, lp_right_count)
    # cv2.putText(out_frame, f"{left_mean:.2f} ", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    right_mean = cal_weighted_mean(rp_shift_list, rp_left_count, rp_right_count)
    # cv2.putText(out_frame, f"{right_mean:.2f} ", (vid_width-50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return left_mean, right_mean


def cal_weighted_mean(shift_list, left_count, right_count):
    if (left_count+right_count)==0:
        return 0
    shift_total = 0
    weight_left = left_count/(left_count+right_count)
    weight_right = right_count/(left_count+right_count)
    weight_total = 0
    for n in shift_list:
      if n>0:
          weight_total += weight_right
          shift_total += n * weight_right
      else:
          weight_total += weight_left  
          shift_total += n * weight_left

    return shift_total/weight_total