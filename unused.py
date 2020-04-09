# Unused code



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