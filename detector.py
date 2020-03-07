import sys
import argparse
import math
from timeit import default_timer as timer
from collections import deque

import cv2
import numpy as np
import torch

from damage_detector import Damage_detector
damage_detector = None

from deeplabv3plus.deeplabv3plus import DeepLabv3plus

from yolov3.models import *
from yolov3.utils.utils import *
from yolov3.utils.datasets import *

from sort import *

import detector_config as DC


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


def proc_frame(writer, frames, frames_infos, frame_no, ss_masks=None, test_writer=None):
    start = timer()
    # print(f"{len(frames)} {len(frames_infos)} {len(ss_masks)}")
    frame2proc = frames.popleft()
    out_frame = frame2proc.copy()
    id_to_info = frames_infos[0]

    if ss_masks is not None:
        ss_mask = ss_masks.popleft()
        

    global class_names, damage_detector, smooth_dict
    global vid_width, vid_height, vid_fps

    # Detect whether the camera is moving
    if len(frames)>0:
        if test_writer:
            test_frame = out_frame.copy()
            is_moving = detect_camera_moving(frame2proc, frames[0], out_frame=test_frame)
            test_writer.write(test_frame)
        else:
            is_moving = detect_camera_moving(frame2proc, frames[0])
    else: #last frame
        is_moving = False
    
    # Smooth moving detection
    if is_moving:
        smooth_dict['is_moving'] = 1
    elif 'is_moving' in smooth_dict and smooth_dict['is_moving'] >0:
        smooth_dict['is_moving'] -= 1
        is_moving = True


    #compute the average shift in pixel of bounding box, in left/right half of the frame
    if ss_masks is None:
        left_mean, right_mean = get_mean_shift(frames_infos, out_frame)

    # car collision detect
    if DC.DET_CAR_PERSON_COL or DC.DET_CAR_COL:
        car_list, person_list = get_list_from_info(id_to_info)
    if DC.DET_CAR_COL:
        car_collision_id_list = detect_car_collision(car_list, out_frame)
    # car-person collision detect
    if DC.DET_CAR_PERSON_COL and not is_moving:
        cdtc_list = get_cdtc_list(frames_infos, car_list)
        car_person_collision_id_list = detect_person_car_collison(id_to_info, cdtc_list)
    # object-wise
    for obj_id in id_to_info:
        info = id_to_info[obj_id]
        class_id, score, bbox = info
        left, top, right, bottom = bbox
        class_name = class_names[class_id]
        ano_dict = {}

        if is_car(class_name):
            # draw_future_center(frames_infos, obj_id, out_frame)
    # damage detection
            if opt.dmg_det:
                # DAMAGE_SKIP_NUM = 2
                obj_dmg_key = f"{obj_id}_dmg"
                if obj_dmg_key in smooth_dict and smooth_dict[obj_dmg_key][0]>0:
                    smooth_dict[obj_dmg_key][0] -= 1
                    dmg_prob = smooth_dict[obj_dmg_key][1]

                else: 
                    # 720p : 90px | 1080p: 135px
                    dmg_height_thres, dmg_width_thres = vid_height//12, vid_width//24
                    if not DC.IGNORE_SMALL or ((bottom-top)>dmg_height_thres and (right-left)>dmg_width_thres) :
                        if DC.DO_PADDING:
                            if (right-left)/(bottom-top) > DC.IS_SIDE_RATIO :
                                x_pad, y_pad = (right-left)//8, (bottom-top)//12
                            else:
                                x_pad, y_pad = (right-left)//12, (bottom-top)//12
                        else:
                            x_pad, y_pad = 0,0

                        if DC.DO_ERASING:
                            dmg_prob = damage_detector.detect(frame2proc, bbox, padding_size=(x_pad, y_pad),
                                                              frame_info = id_to_info, erase_overlap=True, obj_id=obj_id)
                        else:
                            dmg_prob = damage_detector.detect(frame2proc, bbox, padding_size=(x_pad, y_pad))

                        # smooth indication and skip checking to make faster
                        if dmg_prob>0.97:
                            skip_num = 12
                        elif dmg_prob>0.95:
                            skip_num = 6
                        else:
                            skip_num = 3
                        smooth_dict[obj_dmg_key] = [skip_num, dmg_prob]

                    else:
                        dmg_prob = 0

                if dmg_prob>=0.93:
                    ano_dict['damaged'] = True
                cv2.putText(out_frame, f'{dmg_prob:.2f}', ((right+left)//2, (bottom+top)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    # ----damage detection end
            # print(obj_id)
            # if not (left<=0 or right>=vid_width or top<=0 or bottom>=vid_height):
            #     if detect_car_spin(get_bboxes_by_id(frames_infos, obj_id), out_frame):
            #         ano_dict['lost_control'] = True


    # Car collision
            if DC.DET_CAR_COL:
                obj_col_key = f"{obj_id}_col"
                if obj_id in car_collision_id_list:
                    ano_dict['collision'] = True
                    smooth_dict[obj_col_key] = vid_fps//6
                elif obj_col_key in smooth_dict and smooth_dict[obj_col_key] > 0:
                    ano_dict['collision'] = True
                    smooth_dict[obj_col_key] -= 1
    # ----Car collision end


    # Car distance 
            if DC.DET_CLOSE_DIS and is_moving:
                # Detect lack of car distance
                is_close = detect_close_distance(bbox)
                ano_dict['close_distance'] = is_close
    # ----Car distance end
            else: #is not moving
                if DC.DET_CAR_PERSON_COL and obj_id in car_person_collision_id_list:
                    ano_dict['collision'] = True
                # for (obj_id2, bbox) in cdtc_list:
                #     if obj_id == obj_id2:
                #         ano_dict["cdtc"] = True
                #         break

    # Jaywalker
        elif class_name=="person":
            if not is_moving:
                if DC.DET_CAR_PERSON_COL and obj_id in car_person_collision_id_list:
                    ano_dict['jaywalker_crashing'] = True
            if ss_masks is not None: # Use semantic segmentation to find people on traffic road
                if is_moving:
                    obj_on_road_key = f"{obj_id}_on_road"
                    if (frame_no-1)%opt.ss_interval == 0:
                        smooth_dict[obj_on_road_key] =  is_on_traffic_road(bbox, ss_mask)
                    elif obj_on_road_key in smooth_dict:
                            ano_dict['jaywalker'] = smooth_dict[obj_on_road_key]                       

            else: # Use pre-defined baseline
                if is_moving and detect_jaywalker(get_bboxes_by_id(frames_infos, obj_id), (left_mean, right_mean), out_frame):
                    ano_dict['jaywalker'] = True
    # ----Jaywalker end

        draw_bbox(out_frame, ano_dict, class_name, obj_id, score, bbox)
# --- Objects iteration end

    if is_moving:
        cv2.putText(out_frame, "moving", (vid_width//2, vid_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        cv2.rectangle(out_frame, (5, 5), (vid_width-5, vid_height-5), (255,255,0), 5)

    writer.write(out_frame)
    frames_infos.popleft()
    end = timer()
    return (end-start)*1000


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

# testing function
def draw_future_center(frames_infos, obj_id, out_frame):
    bboxes = get_bboxes_by_id(frames_infos, obj_id)
    for (bbox, _) in bboxes:
      center_x, center_y = (bbox[2]+bbox[0])//2, (bbox[3]+bbox[1])//2
      cv2.circle(out_frame,(center_x, center_y), 1, (0, 255, 255), -1)

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
def detect_person_car_collison(id_to_info, cdtc_list):
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





# car list [(obj_id, bbox),]
# return list of obj_id (car that is colliding)
def detect_car_collision(car_list, out_frame):
    collision_list = []
    global vid_width,vid_height
    while len(car_list)>1:
        id1, bbox1 = car_list[0]
        left1, top1, right1, bottom1 = bbox1
        box1_width, box1_height = right1-left1, bottom1-top1

        # ignore small box
        if box1_height<vid_height // DC.COLL_IGNORE_DENOMINATOR:
            del car_list[0]
            continue

        if box1_width/box1_height > DC.IS_SIDE_RATIO:
            is_side1 = True
        else:
            is_side1 = False

        i = 1 # the index for the second box 
        has_match = False
        while i<len(car_list):
            id2, bbox2 = car_list[i]
            left2, top2, right2, bottom2 = bbox2
            box2_width, box2_height = right2-left2, bottom2-top2

            if box2_width/box2_height > DC.IS_SIDE_RATIO:
                is_side2 = True
            else:
                is_side2 = False

            is_checked = False
            if is_side1 and is_side2:
                height_thres = DC.COLL_HEIGHT_THRES_STRICT
                iou_thres = 0.06
            else:  # is_side1 NOR is_side2:
                height_thres = DC.COLL_HEIGHT_THRES
                if is_side1 or is_side2:
                    iou_thres = 0.09
                else:
                    iou_thres = 0.12

            # if they have about the same bottom(height)
            # 1: two sided car i.e. left/right potion of bbox overlap
            # 2: two forward car left/right side crash
            if (abs(bottom1-bottom2) / box1_height) < height_thres and \
                ((right1>right2 and left1>left2) or (right2>right1 and left2>left1)):
                    is_checked = True

            # elif (abs(top1-top2) / box1_height) < (DC.COLL_HEIGHT_THRES_STRICT) and \
            #      (right1>right2 and left1>left2) or (right2>right1 and left2>left1):
            #         is_checked = True
            #         iou_thres = 0.25
       
            elif (is_side1 ^ is_side2):
                        #similar height
                if abs(box1_height-box2_height)/box2_height < height_thres and \
                   ((bottom2>bottom1 and top2>top1) or (bottom1>bottom2 and top1>top2)): # back car crash into front car, y-axis may not be similar
                    is_checked = True
                    iou_thres = 0.25

            if is_checked:   
                iou = compute_iou(bbox1, bbox2)
                # cv2.putText(out_frame, f'{iou:.2f}', ((right1+left1+right2+left2)//4, (top1+bottom1+top2+bottom2)//4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                if iou > iou_thres and iou < DC.IOU_FALSE_THERS: # to exclude some false positive due to detection fault
                    collision_list.append(id2)
                    del car_list[i]
                    has_match = True
                    i -= 1 # compensate the effect of removing element
            i += 1 #proceed to next box2
        if has_match:
            collision_list.append(id1)
        del car_list[0] #remove box1 anyway
    return collision_list
            

# Input bounding box of a person & mask from semantic segmentation
def is_on_traffic_road(bbox, ss_mask, out_frame=None):
    left, top, right, bottom = bbox
    # define check area
    height = bottom - top
    global vid_width, vid_height
    left2, right2 = max(left, 0), min(right, vid_width)
    top2 = min(bottom, vid_height)
    bottom2 = min(bottom+height//10, vid_height)

    # cv2.rectangle(out_frame, (left2, top2), (right2, bottom2), (0,255,255), 1)
    # out_frame[top2:bottom2, left2:right2] = ss_mask[top2:bottom2, left2:right2]

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
                    return True

    if total >0:
        if out_frame is not None:
            cv2.putText(out_frame, f"{(road_count/total):.2f}", ((left+right)//2, (top+bottom)//2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
        if road_count/total>0.5:
            return True
    return False


def detect_jaywalker(recent_bboxes, mean_shift, out_frame=None):
    global vid_height, vid_width

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

def compute_overlapped(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    return interArea / boxAArea


def compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


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
    
# draw bounding box on image given label and coordinate
def draw_bbox(image, ano_dict, class_name, obj_id, score, bbox):
    left, top, right, bottom = bbox
    ano_label = ""
    global vid_height
    thickness = vid_height//720+1
    font_size = vid_height/1080

    anomalies = [("collision", (0,0,255) ),
                 ("lost_control", (255,255,0) ),
                 ("damaged", (123,0,255) ),
                 ("close_distance", (70,255,255) ),
                 ("jaywalker_crashing", (0,100,255) ),
                 ("jaywalker", (0,123,255) )
                 ,("cdtc", (0,123,0)) #test only
                ]
    is_drawn = False
    for (name, color) in anomalies:
        if (name in ano_dict) and ano_dict[name]:
            cv2.rectangle(image, (left, top), (right, bottom), color, thickness)
            left, top, right, bottom = left+thickness, top+thickness, right-thickness, bottom-thickness
            is_drawn = True
            ano_label += f'{name} '

    # if not anomaly, use green
    if not is_drawn:
          cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), thickness)

    # print class name
    if DC.PRINT_CLASS_LABEL:
        label = f'{class_name} {obj_id} : {score:.2f}'
        cv2.putText(image, label, ((right+left)//2, top-5), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), thickness)
    # print anomaly name
    if DC.PRINT_ANOMALY_LABEL:
        cv2.putText(image, ano_label, ((right+left)//2, top-5), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0,0,255), thickness)


# To check whether a point(x,y) is within a triangle area of interest
# by computer the 3 traingles form with any 2 point & (x,y)
# and check if the total area of the 3 traingles equal to the triangle of interest
def inside_roi(x,y, pts):
    global vid_height, vid_width
    x1, y1 = pts[0]
    x2, y2 = pts[1]
    x3, y3 = pts[2]
    a0 = area(x1, y1, x2, y2, x3, y3)
    a1 = area(x, y, x2, y2, x3, y3)
    a2 = area(x1, y1, x, y, x3, y3)
    a3 = area(x1, y1, x2, y2, x, y)
    if a1+a2+a3 == a0:
      return True
    else:
      return False

def area(x1, y1, x2, y2, x3, y3): 
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0) 

def euclidean_distance(x1,x2,y1,y2):
    return math.sqrt( (x2-x1)**2 + (y2-y1)**2)

def detect_close_distance(bbox, out_frame=None):
    global vid_width, vid_height
    pts = [(vid_width//2, vid_height//2), 
           (vid_width//8, vid_height*8//9),
           (vid_width*7//8, vid_height*8//9)]

    left, top, right, bottom = bbox
    center_x, center_y = (left+right)//2, (top+bottom)//2

    if (bottom> vid_height*8//9) or inside_roi(center_x, center_y, pts):
        width = right - left
        dist_score = ( 1-(width/vid_width) )**2
        if out_frame is not None:
            cv2.putText(out_frame, f"{score:.2f}", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (70,255,255), 2)
        if dist_score < 0.5:
            return True
        # elif dist_score < 0.5:
    return False


# Set camera movement detection area
def set_move_det_area():
    result = []
    global vid_height, vid_width

    boxes_x = [vid_width*0.025,vid_width*0.225, vid_width*0.425, vid_width*0.625, vid_width*0.825]
    boxes_y = [vid_height*0.05, vid_height*0.3]
    box_width = int(vid_width*0.15)
    box_height = int(vid_height*0.2)
    for x in range(len(boxes_x)):
        left, top = int(boxes_x[x]), int(boxes_y[0])
        right, bottom = left+box_width, top+box_height
        result.append([left, top, right, bottom])
        if x==0 or x==4:
            left, top = int(boxes_x[x]), int(boxes_y[1])
            right, bottom = left+box_width, top+box_height
            result.append([left, top, right, bottom])        
    
    global detection_boxes, detection_size
    detection_size = box_width*box_height
    detection_boxes = result

   
# detect whether the camera is moving, return img? and boolean
def detect_camera_moving(cur_frame, prev_frame, out_frame=None):
    if prev_frame is None:
        # print("Last frame")
        return False
    threshold = 0.01
    global detection_boxes, detection_size

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
        percentage = cv2.countNonZero(result)/detection_size
        if percentage>threshold:
            count+=1

        # testing purpose
        if out_frame is not None:
            result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            return_img[top:bottom, left:right] = result_bgr
            cv2.rectangle(return_img, (left, top), (right, bottom), (0,255,0), 2)
            label = "%.3f" % percentage
            cv2.putText(out_frame, label, ((left+right)//2, (top+bottom)//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    # 8 boxes in total
    is_moving = count>2
    if is_moving:
        # testing purpose
        if out_frame is not None :
            global vid_width, vid_height
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
    person_bboxes, person_classes = [],[]
    car_bboxes, car_classes = [],[]

    for (box, class_id) in detections:
        class_name = class_names[class_id]
        if class_name=="person":
            person_bboxes.append(box)
            person_classes.append(class_id)
        else:
            car_bboxes.append(box)
            car_classes.append(class_id)
    return car_bboxes, car_classes, person_bboxes, person_classes 


#omit small bboxes since they are not accurate and useful enought for detecting anomaly
def omit_small_bboxes(detections):
    global vid_height
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
        elif DC.OMIT_SIGN and  (class_name=="traffic light" or class_name=="traffic sign") :
            del detections[i]
            omitted_count +=1           
        else:
            i += 1
    # print(f"Omitted {omitted_count} boxes due to small size")
    return omitted_count
    
# yolo wrapper, return list of bounding boxes and list of corresponding classes(id)
def yolo_detect(frame, model):
    global device
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
    return class_name=="car" or class_name=="bus" or class_name=="truck"

def track_video():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0)
    print(f"Using device: {device_name}")
    
    # load video
    video_path = opt.input
    output_path = opt.output
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    # get video prop
    global vid_width, vid_height, vid_fps
    vid_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_total_frame = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = vid.get(cv2.CAP_PROP_FPS)
    if not vid_fps == int(vid_fps):
        old_fps = vid_fps
        vid_fps = round(vid_fps)
        print(f"Rounded {old_fps:2f} fps to {vid_fps}")
    video_length = sec2length(video_total_frame//vid_fps)
    
    # init video writer
    video_FourCC = cv2.VideoWriter_fourcc(*'x264')
    isOutput = True if output_path != "" else False
    if isOutput:
        # print("!!! TYPE:", type(output_path), type(video_FourCC), type(vid_fps), type(video_size))
        print(f"Loaded video: {output_path}, Size = {vid_width}x{vid_height},"
              f" fps = {vid_fps}, total frame = {video_total_frame}")
        out_writer = cv2.VideoWriter(output_path, video_FourCC, vid_fps, (vid_width, vid_height))
    
    print_interval = DC.PRINT_INTERVAL_SEC*vid_fps

    # testing
    output_test = opt.test
    if output_test:
        test_output_path =  output_path.replace("output", "test")
        test_writer = cv2.VideoWriter(test_output_path, video_FourCC, vid_fps, (vid_width, vid_height))
    else:
        test_writer = None

  # global init
    set_move_det_area()

    global class_names
    class_names = load_classes(opt.class_path)
  # init SORT tracker
    max_age = max(3,vid_fps//2)
    car_tracker = Sort(max_age=max_age, min_hits=1)
    person_tracker = Sort(max_age=max_age, min_hits=1)
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
        damage_detector = Damage_detector(device)

    if opt.ss:
        # Create video writer for semantic segmentation result video
        if opt.ss_out:
            ss_output_path =  output_path.replace("output", "ss")
            ss_writer = cv2.VideoWriter(ss_output_path, video_FourCC, vid_fps, (vid_width, vid_height))
        else:
            ss_writer = None
        dlv3 = DeepLabv3plus(device, ss_writer, opt.ss_overlay)
        ss_masks = deque()
    else:
        ss_masks = None

    # Buffer
    buffer_size = vid_fps #store 1sec of frames
    prev_frames = deque()
    frames_infos = deque()
    
    
    # start iter frames
    in_frame_no, proc_frame_no = 0, 1
    print("Start processing video ...")
    start = timer() #First
    while True:
        success, frame = vid.read()
        if not success: #end of video
            break
        in_frame_no += 1

        # semantic seg
        if opt.ss:
            if (in_frame_no-1)%opt.ss_interval == 0:
                mask = dlv3.predict(frame)
            ss_masks.append(mask)

        # Obj Detection
        obj_det_results = yolo_detect(frame, yolo_model)
        omitted_count = omit_small_bboxes(obj_det_results)

        car_bboxes,car_classes, person_bboxes, person_classes = split_bboxes(obj_det_results)
        # tracker_infos is added to return link the class name & the object tracked

        car_trackers, car_tracker_infos = car_tracker.update(np.array(car_bboxes), np.array(car_classes))
        person_trackers, person_tracker_infos = person_tracker.update(np.array(person_bboxes), np.array(person_classes))

        # print(car_trackers, car_tracker_infos)
        # print(person_trackers, person_tracker_infos)

        trackers = [*car_trackers, *person_trackers]
        tracker_infos =  [*car_tracker_infos, *person_tracker_infos]

        id_to_info = {}
        for c, d in enumerate(trackers):
            d = d.astype(np.int32) 
            left, top, right, bottom = int(d[0]), int(d[1]), int(d[2]), int(d[3])
            obj_id = d[4]
            class_id, score = tracker_infos[c][0], tracker_infos[c][1]
            class_name = class_names[class_id]
            if score > -3: #detection is missing
                continue

            info = [class_id, score, [left, top, right, bottom]]
            id_to_info[obj_id] = info

        prev_frames.append(frame)
        frames_infos.append(id_to_info)
        # frame buffer proc
        if len(prev_frames)>buffer_size:
            proc_ms = proc_frame(out_writer, prev_frames, frames_infos, proc_frame_no, ss_masks=ss_masks, test_writer=test_writer)
            proc_frame_no += 1

        if in_frame_no % print_interval == 0:
            end = timer()
            # msg = f"[{sec2length(in_frame_no//vid_fps)}/{video_length}]"
                   # + f"  Found {len(bboxes)} boxes  | {omitted_count} omitted "

            avg_s = (end-start)/print_interval
            fps = str(round(1/avg_s, 2))
            msg = f"[{sec2length(in_frame_no//vid_fps)}/{video_length}] avg_fps: {fps} time: {avg_s*1000}ms"
            print(msg)
            start = timer()

        # if proc_frame_no% print_interval == 0:
            # print(f">> Processing frame {proc_frame_no}, time: {proc_ms:.2f}ms")

    # Process the remaining frames in buffer
    while len(frames_infos)>0:
        proc_frame(out_writer, prev_frames, frames_infos, proc_frame_no, ss_masks = ss_masks, test_writer=test_writer)
        proc_frame_no += 1
    end = timer()
    avg_s = (end-start)/(in_frame_no % print_interval)
    fps = str(round(1/avg_s, 2))
    msg = f"[{sec2length(in_frame_no//vid_fps)}/{video_length}] avg_fps: {fps} time: {avg_s*1000}ms"
    print(msg)

    # release cv2 writer
    if isOutput:
        out_writer.release()
    if output_test:
        test_writer.release()
    if opt.ss_out:
        ss_writer.release()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    # parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    # parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    # parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")

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
    # test
    parser.add_argument('--test', action='store_true', default=False, help = "[Optional]Output testing video")
    # Car damage detect
    parser.add_argument('--dmg_det', action='store_true', default=False, help = "[Optional]do damage classification")
    # Semantic segmentation
    parser.add_argument('--ss', action='store_true', default=False, help = "[Optional]Do semantic segmentation")
    parser.add_argument('--ss_out', action='store_true', default=False, help = "[Optional]Output semantic segmentation video")
    parser.add_argument('--ss_overlay', action='store_true', default=False, help = "[Optional]Overlay the result on the orignal video")
    parser.add_argument('--ss_interval', type=int, default=1, help="frame(s) between segmentations")

    opt = parser.parse_args()
    track_video()
