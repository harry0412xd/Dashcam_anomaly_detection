import sys
import argparse
from math import sqrt
from timeit import default_timer as timer
from collections import deque

import cv2
import numpy as np
import torch

from damage_detector import Damage_detector
from deeplabv3plus.deeplabv3plus import DeepLabv3plus
from yolov3.models import *
from yolov3.utils.utils import *
from yolov3.utils.datasets import *
from sort import *

import detector_config as DC
from utils import *

# model
damage_detector = None
dlv3 = None

#Global Variable
#Video properties : 
vid_width = 0
vid_height = 0
vid_fps = 0

device = None # cuda or cpu
class_names = None # list of classes for yolo
detection_boxes = None 
detection_size = 0

# Use to smoothen some detection
# extend the collision highlight 
#   key = {obj_id}_col, value = no. of frames marked as col
# extend the moving status 
#   key = is_moving, value = no. of frames marked as moving
# Skip damage checking for some frames 
#   key = {obj_id}_dmg, value = [no. of frames, dmg prob]
smooth_dict = {}

rider_list = []

def proc_frame(writer, frames, frames_infos, frame_no, prev_frame, prev_frame_info, test_writer=None):
    # start = timer()
    frame2proc = frames.popleft()
    out_frame = frame2proc.copy()
    id_to_info = frames_infos[0]

    # semantic seg
    if opt.ss:
        if (frame_no-1)%opt.ss_interval == 0:
            ss_mask = dlv3.predict(frame2proc)
        else:
            ss_mask = dlv3.get_last_result(frame2proc)
    # elif DC.DET_JAYWALKER:
        #compute the average shift in pixel of bounding box, in left/right half of the frame
        # left_mean, right_mean = get_mean_shift(frames_infos, out_frame)
 

    if DC.USE_SIGN_TO_DET_MOV:
        moved_count, sign_count, moved_signs = get_moving_sign_count(id_to_info, prev_frame_info, out_frame)
        do_frame_diff_check = False
        if moved_count/max(sign_count,1)>0.3:
            sign_is_moving, is_moving = True, True
        elif moved_count==0 and sign_count>=3:
            sign_is_moving, is_moving = False, False
        else:
            sign_is_moving = False
            do_frame_diff_check = True
    else:
        do_frame_diff_check = True


    test_frame = out_frame.copy() if opt.test=="moving" else None
    if do_frame_diff_check:
            is_moving = detect_camera_moving(frame2proc, prev_frame, out_frame=test_frame)

    if opt.test=="ss":
        test_frame = dlv3.create_overlay(out_frame)

    global smooth_dict
    # Smooth moving detection
    if is_moving:
        smooth_dict['is_moving'] = 1
    elif 'is_moving' in smooth_dict and smooth_dict['is_moving'] >0:
        smooth_dict['is_moving'] -= 1
        is_moving = True

    # car collision detect
    if DC.DET_CAR_PERSON_COL or DC.DET_CAR_COL:
        car_list, person_list = get_list_from_info(id_to_info)
    if DC.DET_CAR_COL:
        car_collision_id_list = detect_car_collision(car_list, out_frame)

    # # car-person collision detect
    # if DC.DET_CAR_PERSON_COL and not is_moving:
    #     cdtc_list = get_cdtc_list(frames_infos, car_list)
    #     car_person_collision_id_list = detect_car_person_collison_new(car_list, person_list, out_frame)

    # object-wise
    for obj_id in id_to_info:
        info = id_to_info[obj_id]
        class_id, score, bbox = info
        left, top, right, bottom = bbox
        class_name = class_names[class_id]
        properties = {}

        # estimate_depth_by_width(bbox, False, out_frame) #test output

        # if DC.DET_CAR_PERSON_COL and obj_id in car_person_collision_id_list:
        #     properties["car_person_crash"] = True

    # Jaywalker
        if class_name=="person":
            if obj_id in rider_list:
                class_name="rider"
            elif DC.DET_JAYWALKER:
                if opt.ss: # Use semantic segmentation to find people on traffic road
                    if is_moving:
                        properties["jaywalker"] = is_on_traffic_road(bbox, ss_mask, out_frame=out_frame)

                else: # Use pre-defined baseline
                    if is_moving and detect_jaywalker(bbox, out_frame=out_frame):
                        properties["jaywalker"] = True
    # ----Jaywalker end

        if is_car(class_name):
            # draw_future_center(frames_infos, obj_id, out_frame)
    # damage detection
            if opt.dmg_det:
                if DC.USE_AVG_PROB:
                    dmg_prob = damage_detector.get_avg_prob(obj_id, frame_no)
                elif DC.USE_ADJUSTED_PROB:
                    dmg_prob = damage_detector.get_adjusted_prob(obj_id, frame_no)
                else:
                    dmg_prob = damage_detector.detect(frame, bbox, id_to_info, frame_no, obj_id)

                if DC.DMG_SKIP_BASE>0:
                    obj_dmg_key = f"{obj_id}_dmg"
                    if obj_dmg_key in smooth_dict and smooth_dict[obj_dmg_key][0]>0:
                        smooth_dict[obj_dmg_key][0] -= 1
                        dmg_prob = smooth_dict[obj_dmg_key][1]
                    else:
                        dmg_prob = damage_detector.get_avg_prob(obj_id, frame_no)

                    # smooth indication and skip checking to make faster
                    skip_num = DC.DMG_SKIP_BASE
                    for i, dmg_thres in enumerate(DC.DMG_THRES):
                        if dmg_prob>dmg_thres: skip_num = DC.DMG_SKIP_NO[i]
                    smooth_dict[obj_dmg_key] = [skip_num, dmg_prob]

                if dmg_prob>=0.85:
                    properties["damaged"] = True
                if DC.SHOW_PROB:
                    cv2.putText(out_frame, f"{dmg_prob:.2f}", ((right+left)//2, (top+bottom)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # ----damage detection end

    # Car collision
            if DC.DET_CAR_COL:
                obj_col_key = f"{obj_id}_col"
                if obj_id in car_collision_id_list:
                    properties["collision"] = True
                    smooth_dict[obj_col_key] = vid_fps//6
                elif obj_col_key in smooth_dict and smooth_dict[obj_col_key] > 0:
                    properties["collision"] = True
                    smooth_dict[obj_col_key] -= 1
    # ----Car collision end

    # Car distance 
            if DC.DET_CLOSE_DIS and is_moving:
                # Detect lack of car distance
                is_close = detect_close_distance(bbox)
                properties['close_distance'] = is_close
    # ----Car distance end

    # traffic light/signs
        elif class_name=="traffic light":
            traffic_color = get_traffic_color(frame2proc, bbox, out_frame=None)
            properties["traffic_color"] = traffic_color
            
        elif DC.SHOW_SIGN_MOVEMENT and obj_id in moved_signs:
            properties["signs"] = moved_signs[obj_id]
    # ----traffic light/signs

        draw_bbox(out_frame, properties, class_name, obj_id, score, bbox)
        if opt.test:
            draw_bbox(test_frame, properties, class_name, obj_id, score, bbox)
# --- Objects iteration end

    if is_moving:
        if DC.USE_SIGN_TO_DET_MOV and sign_is_moving:
            moving_label = "sign_moving"
            moving_color = (255,200,0)
        else:
            moving_label = "moving"
            moving_color = (255,255,0)

        cv2.putText(out_frame, moving_label, (vid_width//2, vid_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, moving_color, 2)
        cv2.rectangle(out_frame, (5, 5), (vid_width-5, vid_height-5), (255,255,0), 5)
        if opt.test:
            cv2.putText(test_frame, moving_label, (vid_width//2, vid_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, moving_color, 2)
            cv2.rectangle(test_frame, (5, 5), (vid_width-5, vid_height-5), (255,255,0), 5)


    writer.write(out_frame)
    if test_writer is not None:
        test_writer.write(test_frame)
    frames_infos.popleft()
    # end = timer()
    # return (end-start)*1000
    return frame2proc, id_to_info


def get_list_from_info(all_info):
    car_list, person_list = [], []
    for obj_id in all_info:
        info = all_info[obj_id]
        class_id, _, bbox = info
        class_name = class_names[class_id]
        if is_car(class_name):
            car_list.append((obj_id, bbox))
        elif class_name=="person":
            person_list.append((obj_id, bbox))

    return car_list, person_list


# car list [(obj_id, bbox),]
# return list of obj_id (car that is colliding)
def detect_car_collision(car_list, out_frame):
    collision_list = []
    while len(car_list)>1:
        id1, bbox1 = car_list[0]
        left1, top1, right1, bottom1 = bbox1
        width1, height1 = right1-left1, bottom1-top1

        # ignore small box
        if height1<vid_height // 24:
            del car_list[0]
            continue

        if width1/height1 > DC.SIDE_THRES:
            is_side1 = True
        else:
            is_side1 = False

        i = 1 # the index for the second box 
        has_match = False
        while i<len(car_list):
            id2, bbox2 = car_list[i]
            left2, top2, right2, bottom2 = bbox2
            width2, height2 = right2-left2, bottom2-top2
            if width2/height2 > DC.SIDE_THRES:
                is_side2 = True
            else:
                is_side2 = False

            is_possible = False
            if is_side1 and is_side2:
                height_thres = 0.03
                iou_thres = 0.06
            else:  
                height_thres = 0.1
                if is_side1 or is_side2:
                    iou_thres = 0.09
                else: # is_side1 NOR is_side2:
                    iou_thres = 0.12

            # if they have about the same bottom(height)
            # 1: two sided car i.e. left/right potion of bbox overlap
            # 2: two forward car left/right side crash
            if (abs(bottom1-bottom2) / height1) < height_thres:
                # ((right1>right2 and left1>left2) or (right2>right1 and left2>left1)):
                    is_possible = True
                    
            # elif (is_side1 ^ is_side2):
            #             #similar height
            #     if abs(height1-height2)/height1 < 0.06 and \
            #        ((bottom2>bottom1 and top2>top1) or (bottom1>bottom2 and top1>top2)): # back car crash into front car, y-axis may not be similar
            #         is_possible = True
            #         iou_thres = 0.25

            if is_possible: 
                d1, d2 = estimate_depth_by_width(bbox1, True), estimate_depth_by_width(bbox2, True)
                d_diff = abs( (d1-d2) / ( (d1+d2)/2 ) )
                if d_diff < 0.15: #consider width to estimate depth
                    iou = compute_iou(bbox1, bbox2)
                    # cv2.putText(out_frame, f'{iou:.2f}', ((right1+left1+right2+left2)//4, (top1+bottom1+top2+bottom2)//4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                    overlapped_perc = max(compute_overlapped(bbox1, bbox2), compute_overlapped(bbox2, bbox1))# to exclude some false positive due to detection fault
                    if iou > iou_thres and overlapped_perc < 0.6: 
                        collision_list.append(id2)
                        del car_list[i]
                        has_match = True
                        i -= 1 # compensate the effect of removing element

            i += 1 #proceed to next box2
        if has_match:
            collision_list.append(id1)
        del car_list[0] #remove box1 anyway
    return collision_list
            

def check_is_rider(person_id, person_bbox, id_to_info):
    left, top, right, bottom = person_bbox
    center_x, center_y = (left+right)//2, (top+bottom)//2

    for obj_id in id_to_info:
        if obj_id != person_id:
            class_id2, _, bbox2 = id_to_info[obj_id]
            if class_id2==3 or class_id2==0: #motor or bike
                left2, top2, right2, bottom2 = bbox2
                center_x2, center_y2 = (left2+right2)//2, (top2+bottom2)//2
              
                if euclidean_distance(center_x,center_x2,center_y,center_2)<(bottom-top) or \
                   compute_overlapped(person_bbox, bbox2)>0.3:
                    rider_list.append(person_id)
                   



# Input bounding box of a person & mask from semantic segmentation
def is_on_traffic_road(bbox, ss_mask, out_frame=None):
    left, top, right, bottom = bbox
    center_x = (left+right)//2
    height = bottom - top
    # define check area
    left2, right2 = max(left, 0), min(right, vid_width)
    top2 = min(bottom-height//12, vid_height)
    bottom2 = min(bottom+height//12, vid_height)

    if top2==bottom2:
        return True

    total, road_count = 0, 0
    for y in range(top2, bottom2):
        for x in range(left2, right2):
            b,g,r = ss_mask[y][x]

            # is not person
            if not (r==220 and g==20 and b==60): 
                total +=1
                # is road
                if r==128 and g==64 and b==128:
                    road_count +=1
                # is padding
                elif r==255 and g==255 and b==255:
                    if center_x>vid_width//6 and center_x<vid_width*5//6:
                        return True
                    else:
                        road_count += 0.5

    if total >0:
        if out_frame is not None:
            cv2.putText(out_frame, f"{(road_count/total):.2f}", ((left+right)//2, (top+bottom)//2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
            cv2.rectangle(out_frame, (left2, top2), (right2, bottom2), (0,255,255), 1)
            out_frame[top2:bottom2, left2:right2] = ss_mask[top2:bottom2, left2:right2]
        if road_count/total>0.5:
            return True
    return False


def get_traffic_color(frame, bbox, out_frame=None):
    left, top, right, bottom = bbox
    left, top, right, bottom = max(left,0), max(top,0), min(right, vid_width), min(bottom, vid_height)
    img_size = (bottom-top)*(right-left)
    light_img = frame[top:bottom, left:right]
    # convert to hsv
    hsv = cv2.cvtColor(light_img, cv2.COLOR_BGR2HSV) 
    s,v = 60, 123
    # green
    lower_green = np.array([40,s,v]) 
    upper_green = np.array([95,255,255]) 
    green_mask = cv2.inRange(hsv, lower_green, upper_green) 
    # red
    lower_red1 = np.array([0,s,v]) 
    upper_red1 = np.array([20,255,255])
    lower_red2 = np.array([165,s,v]) 
    upper_red2 = np.array([180,255,255]) 
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = red_mask1+red_mask2
    # yellow
    lower_yellow = np.array([25,s,v]) 
    upper_yellow = np.array([35,255,255]) 
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow) 


    red_perc = cv2.countNonZero(red_mask)/img_size
    green_perc = cv2.countNonZero(green_mask)/img_size
    yellow_perc = cv2.countNonZero(yellow_mask)/img_size

    red_perc = red_perc if 0.6>red_perc>0.02 else 0
    green_perc = green_perc if 0.6>green_perc>0.02 else 0
    yellow_perc = yellow_perc if 0.6>yellow_perc>0.02 else 0

    if out_frame is not None:
        cv2.putText(out_frame, f"{red_perc:.2f}|{green_perc:.2f}|{yellow_perc:.2f}", ((right+left)//2, bottom+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    max_ = max(red_perc, green_perc, yellow_perc)
    if max_==0:
        return
    elif max_ == green_perc:
        return "green"
    elif max_ == red_perc:
        return "red"
    else:
        return "yellow"


def estimate_depth_by_width(bbox, is_car, out_frame=None):
    multiplier = 100 #make the score eaiser to read

    left, top, right, bottom = bbox
    width, height = right-left, bottom-top
    center_x = (left+right)//2
    center_y = (top+bottom)//2
    if is_car:      
        dist = abs(center_x - vid_width//2)
        factor = (dist/(vid_width//2) + 1)*1.2 - 0.2

        if width/height>=2:
            result = (width/2.4) / vid_width * multiplier
        else:
            result = (width/factor) / vid_width * multiplier   
    else:
        result = 1.6*width/ vid_width * multiplier

    result = multiplier - result #revert so that larger value = futher
    if out_frame is not None:
        cv2.putText(out_frame, f"{result:.2f} ", (center_x-5, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return result


# retrieve bounding boxes for an object in future n frames given obj_id
# return list of [bbox, x] , x = frame offset i.e. that frame is x frames after 
def get_bboxes_by_id(frames_infos, obj_id, length=None):
    bboxes_n_frameNum = []
    if length == None:
        length = len(frames_infos)
    else:
        length = min(len(frames_infos), length)

    for i in range(length):
        id_to_info = frames_infos[i]
        if obj_id in id_to_info:
          _, _, bbox = id_to_info[obj_id]
          bboxes_n_frameNum.append([bbox, i])
    return bboxes_n_frameNum

    
# draw bounding box on image given label and coordinate
def draw_bbox(image, properties, class_name, obj_id, score, bbox):
    left, top, right, bottom = bbox
    ano_label = ""
    thickness = vid_height//720+1
    font_size = vid_height/1080
    anomalies = [("collision", (0,0,255) ),
                 ("lost_control", (255,255,0) ),
                 ("damaged", (123,0,255) ),
                 ("close_distance", (70,255,255) ),
                 ("jaywalker", (0,123,255) ),
                 ("car_person_crash", (0,51,153)) #brown
                ]

    is_drawn = False

    # Anomalies
    for (name, color) in anomalies:
        if (name in properties) and properties[name]:
            cv2.rectangle(image, (left, top), (right, bottom), color, thickness)
            left, top, right, bottom = left+thickness, top+thickness, right-thickness, bottom-thickness
            is_drawn = True
            ano_label += f'{name} '

    # traffic light or sign
    if DC.SHOW_SIGN_MOVEMENT and "signs" in properties:
        color = (211,211,211) #gray
        dis, wdiff, hdiff= properties["signs"]
        cv2.putText(image, f"{dis:.2f} {wdiff:.2f} {hdiff:.2f}", ((right+left)//2, bottom+5), cv2.FONT_HERSHEY_SIMPLEX, font_size*0.8, color, thickness)
        

    # only traffic light
    if "traffic_color" in properties:
        color = properties["traffic_color"]
        if color=="red":
            color = (80,80,255)#red
        elif color=="green":
            color = (80,255,80)#green
        elif color=="yellow":
            color = (80,255,255)#yellow
        cv2.rectangle(image, (left, top), (right, bottom), color, max(1,thickness-1))
        is_drawn = True

    if not is_drawn:
          color = (0,255,0) # if not anomaly, use green
          cv2.rectangle(image, (left, top), (right, bottom), color, thickness)

    # print class name
    if DC.PRINT_OBJ_ID:
        cv2.putText(image, str(obj_id), ((right+left)//2, top-5), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), thickness)
    elif DC.PRINT_CLASS_LABEL:
        label = f'{class_name} {obj_id} : {score:.2f}'
        cv2.putText(image, label, ((right+left)//2, top-5), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), thickness)
    # print anomaly name
    if DC.PRINT_ANOMALY_LABEL:
        cv2.putText(image, ano_label, ((right+left)//2, top-5), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0,0,255), thickness)


def detect_close_distance(bbox, out_frame=None):   
    pts = [(vid_width//2, vid_height//2), 
           (vid_width//6, vid_height),
           (vid_width*5//6, vid_height)]

    left, top, right, bottom = bbox
    center_x, center_y = (left+right)//2, (top+bottom)//2

    # if (bottom> vid_height*8//9) or inside_roi(center_x, center_y, pts):
    if inside_roi((center_x, center_y), pts):
        width = right - left
        dist_score = ( 1-(width/vid_width) )**2
        if out_frame is not None:
            cv2.line(out_frame, pts[0], pts[1], (255,0,0))
            cv2.line(out_frame, pts[0], pts[2], (255,0,0))
            cv2.putText(out_frame, f"{dist_score:.2f}", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70,255,255), 2)
        if dist_score < 0.5:
            return True
        # elif dist_score < 0.5:
    return False


def detect_jaywalker(bbox, out_frame=None):
    pts = [(vid_width//2, vid_height//4), 
           (vid_width//8, vid_height),
           (vid_width*7//8, vid_height)]
    y_thres = int(vid_height*0.45)
    if out_frame is not None:
        cv2.line(out_frame, pts[0], pts[1], (255,0,0))
        cv2.line(out_frame, pts[0], pts[2], (255,0,0))
        cv2.line(out_frame, (0,y_thres), (vid_width,vid_height//3), (255,0,0))

    left, top, right, bottom = bbox
    height, bottom_center = bottom-top, ((left+right)//2, bottom)
    center_x = (left+right)//2

    if (bottom>y_thres) and (height>vid_height//4) and inside_roi(bottom_center, pts):
        return True
    return False

# Set camera movement detection area
# def set_move_det_area():
#     result = []
#     boxes_x = [0, vid_width*0.025,vid_width*0.225, vid_width*0.425, vid_width*0.625, vid_width*0.825]
#     boxes_y = [vid_height*0.05, vid_height*0.3]
#     box_width = int(vid_width*0.15)
#     box_height = int(vid_height*0.2)
#     for x in range(len(boxes_x)):
#         left, top = int(boxes_x[x]), int(boxes_y[0])
#         right, bottom = left+box_width, top+box_height
#         result.append([left, top, right, bottom])
#         if x==0 or x==4:
#             left, top = int(boxes_x[x]), int(boxes_y[1])
#             right, bottom = left+box_width, top+box_height
#             result.append([left, top, right, bottom])        
    
#     global detection_boxes, detection_size
#     detection_size = box_width*box_height
#     detection_boxes = result

# Set camera movement detection area
def set_move_det_area():
    result = []
    column = 6
    p = vid_width//column//10
    boxes_y = [0, vid_height//5]
    box_height = vid_height//6
    for x in range(0,column):
        left = x* vid_width//column + p
        right = (x+1)* vid_width//column - p
        top, bottom = int(boxes_y[0]), int(boxes_y[0])+box_height
        result.append([left, top, right, bottom])
        if x==0 or (x+1)==column:
            top, bottom = int(boxes_y[1]), int(boxes_y[1])+box_height
            result.append([left, top, right, bottom])        
    
    global detection_boxes, detection_size
    box_width = vid_width//column - 2*p
    detection_size = box_width*box_height
    detection_boxes = result


def get_moving_sign_count(cur_frame_info, prev_frame_info, out_frame=None):
    if prev_frame_info is None:
        return 0, 0, {} #first frame
    sign_count, moved_count = 0, 0
    moved_signs = {}
    for obj_id in cur_frame_info:
        class_id, _, bbox1 = cur_frame_info[obj_id]
        class_name = class_names[class_id]
        if class_name=="traffic light" or class_name=="traffic sign":
            if obj_id in prev_frame_info:
                sign_count += 1
                class_id, _, bbox2 = prev_frame_info[obj_id]
                left1, top1, right1, bottom1 = bbox1
                left2, top2, right2, bottom2 = bbox2

                width1, width2 = right1-left1, right2-left2
                height1, height2 = top1-bottom1,top2-bottom2
                center_x1, center_y1 = (left1+right1)//2, (top1+bottom1)//2
                center_x2, center_y2 = (left2+right2)//2, (top2+bottom2)//2

                width_diff, height_diff = abs((width1-width2)/width2), abs((height1-height2)/height2)
                dis = euclidean_distance(center_x1, center_x2, center_y1, center_y2)
                diag2 = sqrt(width2*width2+height2*height2)

                if DC.SHOW_SIGN_MOVEMENT:
                    moved_signs[obj_id] =  [dis/diag2, width_diff, height_diff] # for testing output

                # print(moved_signs[obj_id])
                # if dis/diag2 >0.05 or width_diff>0.03 or height_diff>0.03:
                if dis/diag2 >0.05 and not(width_diff>0.08 or height_diff>0.08):
                    sign_count += 1
                    moved_count += 1

    if out_frame is not None:
        cv2.putText(out_frame, f"{moved_count}/{sign_count}", (vid_width-70, vid_height-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    return moved_count, sign_count, moved_signs
                

# detect whether the camera is moving, return img? and boolean
def detect_camera_moving(cur_frame, prev_frame, out_frame=None):
    if prev_frame is None:
        return False

    score_thres = 0.01
    count = 0
    for box in detection_boxes:
        left, top, right, bottom = box
        # select out the box and convert to gray
        box_cur = cur_frame[top:bottom, left:right].copy()
        box_cur = cv2.cvtColor(box_cur, cv2.COLOR_BGR2GRAY)
        box_prev = prev_frame[top:bottom, left:right].copy()
        box_prev = cv2.cvtColor(box_prev, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(box_cur, box_prev)
        ret, result = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        score = cv2.countNonZero(result)/detection_size
        if score>score_thres:
            count+=1

        # testing purpose
        if out_frame is not None:
            result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            out_frame[top:bottom, left:right] = result_bgr
            c = (0,0,255) if score>score_thres else (0,255,0)
            cv2.rectangle(out_frame, (left, top), (right, bottom), c, 2)
            label = "%.3f" % score
            cv2.putText(out_frame, label, ((left+right)//2, (top+bottom)//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    # 8 boxes in total
    is_moving = count>2
    if is_moving:
        # testing purpose
        if out_frame is not None :
            cv2.putText(out_frame, "Is moving", (vid_width//2, vid_height-50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    
    return is_moving

# small func to help display progress
def sec2length(time_sec):
    m = int(time_sec//60)
    s = int(time_sec%60)
    if s<10:
        s= "0"+str(s)
    return f"{m}:{s}" 

# split into car & person
def split_bboxes(detections):
    person_bboxes, person_classes = [], []
    car_bboxes, car_classes = [], []
    sign_bboxes, sign_classes = [], []

    for (box, class_id) in detections:
        class_name = class_names[class_id]
        if class_name=="person":
            person_bboxes.append(box)
            person_classes.append(class_id)
        elif class_name=="traffic light" or class_name=="traffic sign":
            sign_bboxes.append(box)
            sign_classes.append(class_id)
        elif is_car(class_name):
            car_bboxes.append(box)
            car_classes.append(class_id)
    return car_bboxes, car_classes, person_bboxes, person_classes, sign_bboxes, sign_classes


#omit unwanted bounding boxes
def omit_bboxes(detections):
    # area_threshold = (vid_height//36)**2
    width_threshold, height_threshold = vid_width//30, vid_height//24
    omitted_count = 0

    i = 0
    while i<len(detections):
        bbox, class_id = detections[i]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        class_name = class_names[class_id]

        if DC.OMIT_SMALL and is_car(class_name) and \
            (width<width_threshold or height<height_threshold):
            # print(f"{classes[i]} {width}x{height}")
            del detections[i]
            omitted_count +=1
        elif DC.OMIT_SIGN and (class_name=="traffic light" or class_name=="traffic sign") :
            del detections[i]
            omitted_count +=1           
        else:
            i += 1
    # print(f"Omitted {omitted_count} boxes due to small size")
    return omitted_count
    
# yolo wrapper, return list of bounding boxes and list of corresponding classes(id)
def yolo_detect(frame, model):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(frame.transpose(2, 0, 1)).float().to(device)
    x = x.unsqueeze(0)
    _, _, h, w = x.size()

    ih, iw = (opt.img_size, opt.img_size)
    dim_diff = np.abs(h - w)
    pad1, pad2 = int(dim_diff // 2), int(dim_diff - dim_diff // 2)
    pad = (pad1, pad2, 0, 0) if w <= h else (0, 0, pad1, pad2)
    x = F.pad(x, pad=pad, mode='constant', value=127.5) / 255.0
    x = F.interpolate(x, size=(ih, iw), mode='bilinear')
    # x = F.upsample(x, size=(ih, iw), mode='bilinear')
    with torch.no_grad():
        detections = model.forward(x)
        detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)[0]
    results = []
    # bboxes = []
    # classes = []
    if detections is not None:
        detections = rescale_boxes(detections, opt.img_size, frame.shape[:2])
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
          #  bboxes.append([x1, y1, x2, y2, cls_conf.item()])
          #  classes.append(int(cls_pred))
          results.append([ [x1, y1, x2, y2, cls_conf.item()], int(cls_pred)])
    # return bboxes, classes
    return results


def is_car(class_name):
    return class_name=="car" or class_name=="bus" or class_name=="truck" \
           or class_name=="motor" or class_name=="rider"


def detect_damaged_car(id_to_info, frame, frame_no):
    for obj_id in id_to_info:
        class_id, score, bbox = id_to_info[obj_id]
        left, top, right, bottom = bbox
        dmg_height_thres, dmg_width_thres = 64, 64
        if not DC.IGNORE_SMALL or ((bottom-top)>dmg_height_thres and (right-left)>dmg_width_thres):
            damage_detector.detect(frame, bbox, id_to_info, frame_no, obj_id)


def track_video():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0)
    print(f"Using device: {device_name}")
    
    # load video
    video_path = opt.input
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    # get&set video prop
    global vid_width, vid_height, vid_fps
    vid_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_total_frame = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = vid.get(cv2.CAP_PROP_FPS)
    # if not vid_fps == int(vid_fps):
    #     old_fps = vid_fps
    #     vid_fps = round(vid_fps)
    #     print(f"Rounded {old_fps:2f} fps to {vid_fps}")
    video_length = sec2length(video_total_frame//vid_fps)
    
    # auto rename if no path is provided
    if opt.output == "":
        if not os.path.exists("video_out"):
            os.makedirs("video_out")  
        _filename = video_path.split("/")
        filename = _filename[len(_filename)-1]
        output_path = "video_out/" + filename.replace(".", "_output.")
    isOutput = True

    # init video writer
    if opt.x264:
        video_FourCC = cv2.VideoWriter_fourcc(*'x264')
        output_path = output_path.replace(".mp4", ".mkv")
        print(output_path)
    else:
        video_FourCC = cv2.VideoWriter_fourcc(*'mp4v')
    if output_path:
        print(f"Saving to: {output_path}, Size = {vid_width}x{vid_height},"
              f" fps = {vid_fps}, total frame = {video_total_frame}")
        out_writer = cv2.VideoWriter(output_path, video_FourCC, vid_fps, (vid_width, vid_height))

    print_interval = int(DC.PRINT_INTERVAL_SEC*vid_fps)

    # testing
    output_test = opt.test
    if opt.test:
        test_output_path =  output_path.replace("output", "test")
        test_writer = cv2.VideoWriter(test_output_path, video_FourCC, vid_fps, (vid_width, vid_height))
    else:
        test_writer = None

    if opt.save_path:
        save_result = True
        assert opt.result_path=="", "Save result & load result are both chosen."
        # _filename = opt.input.split("/")
        # result_filename = "detection_results/" + _filename[len(_filename)-1].split(".")[0] + ".txt"
        result_filename = opt.save_path
        det_result_file = open(result_filename, 'w')
    else:
        save_result = False

    if opt.result_path=="":
        is_use_result = False
    else:
        assert not save_result, "Save result & load result are both chosen."
        is_use_result = True
        all_results = load_det_result(opt.result_path)


    set_move_det_area()
    global class_names
    class_names = load_classes(opt.class_path)

    if not is_use_result:
      # init SORT tracker
        max_age = max(3,vid_fps//2)
        car_tracker = Sort(max_age=max_age, min_hits=1)
        person_tracker = Sort(max_age=max_age, min_hits=1)
        sign_trackers = Sort(max_age=max_age//2, min_hits=1)

        print("SORT initialized")
      # init yolov3 model
        if opt.weights_path == "model_data/YOLOv3_bdd/bdd.weights":
            if not os.path.isfile(opt.weights_path):
                print("Downloading YOLO weights ... ")
                url = "https://github.com/harry0412xd/Dashcam_anomaly_detection/releases/download/v1.0/bdd.weights"
                import urllib.request
                urllib.request.urlretrieve(url, "model_data/YOLOv3_bdd/bdd.weights")
        yolo_model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
        yolo_model.load_darknet_weights(opt.weights_path)
        yolo_model.eval()
        print("YOLO model loaded")

  # Car damage detection
    if opt.dmg_det:
        global damage_detector
        if DC.USE_AVG_PROB or DC.USE_ADJUSTED_PROB:
            damage_detector = Damage_detector(device, do_erasing=DC.DO_ERASING, do_padding=DC.DO_PADDING,
                                              side_thres=DC.SIDE_THRES, save_probs = True, prob_period=DC.PROB_PERIOD, weighted_prob=DC.WEIGHTED_PROB)
        else:
            damage_detector = Damage_detector(device, do_erasing=DC.DO_ERASING, do_padding=DC.DO_PADDING, side_thres=DC.SIDE_THRES)
  # Semantic segmentation
    if opt.ss:
        global dlv3
        # Create video writer for semantic segmentation result video
        if opt.ss_out:
            ss_output_path =  output_path.replace("output", "ss")
            ss_writer = cv2.VideoWriter(ss_output_path, video_FourCC, vid_fps, (vid_width, vid_height))
        else:
            ss_writer = None
        dlv3 = DeepLabv3plus(device, ss_writer, opt.ss_overlay)

    # Buffer
    buffer_size = vid_fps #store 1sec of frames
    prev_frames = deque()
    frames_infos = deque()
    
    # start iter frames
    in_frame_no, proc_frame_no = 0, 1
    prev_frame, prev_frame_info = None, None
    print("Start processing video ...")
    start = timer() #First
    while True:
        success, frame = vid.read()
        if not success: #end of video
            break
        in_frame_no += 1

        if is_use_result:
            id_to_info = use_det_result(all_results, in_frame_no)
        else:
            # Obj Detection
            obj_det_results = yolo_detect(frame, yolo_model)
            # print(f"results:{len(obj_det_results)}")
            omitted_count = omit_bboxes(obj_det_results)

            car_bboxes,car_classes, \
            person_bboxes, person_classes, \
            sign_bboxes, sign_classes = split_bboxes(obj_det_results)
            
            # tracker_infos is added to return link the class name & the object tracked
            car_trackers, car_tracker_infos = car_tracker.update(np.array(car_bboxes), np.array(car_classes))
            person_trackers, person_tracker_infos = person_tracker.update(np.array(person_bboxes), np.array(person_classes))
            sign_trackers, sign_tracker_infos = person_tracker.update(np.array(sign_bboxes), np.array(sign_classes))

            # join the trackers
            trackers = [*car_trackers, *person_trackers]
            tracker_infos =  [*car_tracker_infos, *person_tracker_infos]

            id_to_info = {} #key: id  value: info
            for c, d in enumerate(trackers):
                d = d.astype(np.int32) 
                left, top, right, bottom, obj_id = int(d[0]), int(d[1]), int(d[2]), int(d[3]), d[4]
                class_id, score = tracker_infos[c][0], tracker_infos[c][1]
                class_name = class_names[class_id]
                if is_car(class_name) and score <0 : #detection is missing
                    continue
                elif class_name=="traffic sign" or class_name=="traffic light" and score < -1:
                    continue
                elif class_name=="person" and score <= -(DC.PERSON_MISS_TOLERATE):
                    continue
                # add to dict
                info = [class_id, score, [left, top, right, bottom]]
                id_to_info[obj_id] = info

            if save_result:
                # print(f"saving: {in_frame_no}")
                save_det_result(det_result_file, id_to_info, in_frame_no)

        if opt.dmg_det and (DC.USE_AVG_PROB or DC.USE_ADJUSTED_PROB):
            detect_damaged_car(id_to_info, frame, in_frame_no) #wrapper function to iterate all obj

        prev_frames.append(frame)
        frames_infos.append(id_to_info)
        # frame buffer proc
        if len(prev_frames)>buffer_size:
            prev_frame, prev_frame_info = proc_frame(out_writer, prev_frames, frames_infos, proc_frame_no, 
                                                     prev_frame, prev_frame_info, test_writer=test_writer)

            proc_frame_no += 1

        if in_frame_no % print_interval == 0:
            end = timer()
            avg_s = (end-start)/print_interval
            fps = str(round(1/avg_s, 2))
            msg = f"[{sec2length(in_frame_no//vid_fps)}/{video_length}] avg_fps: {fps} time: {avg_s*1000}ms"
            print(msg)
            start = timer()

    # Process the remaining frames in buffer
    while len(frames_infos)>0:
        prev_frame, prev_frame_info = proc_frame(out_writer, prev_frames, frames_infos, proc_frame_no, 
                                                 prev_frame, prev_frame_info, test_writer=test_writer)
        proc_frame_no += 1

    end = timer()
    if in_frame_no % print_interval>0:
        avg_s = (end-start)/(in_frame_no % print_interval)
        fps = str(round(1/avg_s, 2))
        msg = f"[{sec2length(in_frame_no//vid_fps)}/{video_length}] avg_fps: {fps} time: {avg_s*1000}ms"
        print(msg)


    # release cv2 writer
    if isOutput:
        out_writer.release()
    if opt.test:
        test_writer.release()
    if opt.ss_out:
        ss_writer.release()

    if save_result:
        det_result_file.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # YOLO param
    parser.add_argument("--model_def", type=str, default="model_data/YOLOv3_bdd/bdd.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="model_data/YOLOv3_bdd/bdd.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="model_data/YOLOv3_bdd/classes.txt", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.55, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.25, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    # I/O
    parser.add_argument("--input", nargs='?', type=str, default="",help = "Video input path")
    parser.add_argument("--output", nargs='?', type=str, default="",  help = "[Optional] Video output path")
    parser.add_argument('--x264', action='store_true', default=False, help = "[Optional]Use x264")
    # Car damage detect
    # test
    parser.add_argument('--test', type=str, default="", help = "[Optional]Output testing video [moving/ss]")
    # Car damage detect
    parser.add_argument('--dmg_det', action='store_true', default=False, help = "[Optional]do damage classification")
    # Semantic segmentation
    parser.add_argument('--ss', action='store_true', default=False, help = "[Optional]Do semantic segmentation")
    parser.add_argument('--ss_out', action='store_true', default=False, help = "[Optional]Output semantic segmentation video")
    parser.add_argument('--ss_overlay', action='store_true', default=False, help = "[Optional]Overlay the result on the orignal video")
    parser.add_argument('--ss_interval', type=int, default=1, help="frame(s) between segmentations")
    # save/load detection&tracking results
    # parser.add_argument('--save_result', action='store_true', default=False, help = "[Optional]Output the Object detection/tracking results to a text file")
    parser.add_argument('--result_path', type=str, default="", help = "[Optional]Path of file which save the Object detection/tracking results")
    parser.add_argument('--save_path', type=str, default="", help = "[Optional]Path of where the Object detection/tracking results should be saved")
    
    opt = parser.parse_args()
    track_video()
