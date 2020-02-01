import sys
import argparse
import math
from timeit import default_timer as timer
from collections import deque

import cv2
import numpy as np
import torch

# from lane_SCNN.wrapper import Lane_model
# from Fast_SCNN.wrapper import *
# from ERFNet_CULane_PyTorch.wrapper import *
from damage_detector import Damage_detector
damage_detector = None

from yolov3.models import *
from yolov3.utils.utils import *
from yolov3.utils.datasets import *

from sort import *

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
#   key = {obj_id}_dmg, value = no. of frames
smooth_dict = {}

def proc_frame(writer, frames, frames_infos, test_writer=None):
    start = timer()

    frame2proc = frames.popleft()
    out_frame = frame2proc.copy()
    id_to_info = frames_infos[0]
    global class_names, damage_detector, smooth_dict
    global vid_width, vid_height, vid_fps

    # Detect whether the camera is moving
    if len(frames)>0
        _, is_moving = detect_camera_moving(frame2proc, frames[0])
    else: #last frame
        is_moving = False
    
    if is_moving:
        smooth_dict['is_moving'] = 1
    elif 'is_moving' in smooth_dict and smooth_dict['is_moving'] >0:
        smooth_dict['is_moving'] -= 1
        is_moving = True


    #compute the average shift in pixel of bounding box, in left/right half of the frame
    left_mean, right_mean = get_mean_shift(frames_infos, out_frame)

    collision_id_list = detect_car_collision(retrieve_all_car_info(frames_infos[0]))

    # object-wise
    for obj_id in id_to_info:
        info = id_to_info[obj_id]
        class_id, score, bbox = info
        left, top, right, bottom = bbox

        class_name = class_names[class_id]
        label = f'{class_name} {obj_id} : {score:.2f}'
        # print (f"  {label} at {left},{top}, {right},{bottom}")
        ano_dict = {"label": label}

        if class_name=="car" or class_name=="bus" or class_name=="truck":

            # damaged car - image classifier
            # [frame_count, dmg_prop]
            DAMAGE_SKIP_NUM = 6

            obj_dmg_key = f"{obj_id}_dmg"
            if obj_dmg_key in smooth_dict and smooth_dict[obj_dmg_key][0]>0:
                smooth_dict[obj_dmg_key][0] -= 1
                dmg_prob = smooth_dict[obj_dmg_key][1]
            else: 
                # 720p : 90px | 1080p: 135px
                dmg_height_thres, dmg_width_thres = vid_height//8, vid_height//16
                # if False:
                if (bottom-top)>dmg_height_thres and (right-left)>dmg_width_thres:
                    if (right-left)/(bottom-top) >1.3:
                        x_pad, y_pad = (right-left)//6, (bottom-top)//12
                    x_pad, y_pad = (right-left)//12, (bottom-top)//12
                    # x_pad, y_pad = 0,0
                    left2, top2, right2, bottom2 = max(left-x_pad,0), max(top-y_pad,0),\
                                                      min(right+x_pad, vid_width), min(bottom+y_pad, vid_height)

                    # Pass obj_id to output test image
                    det_class, dmg_prob = damage_detector.detect(frame2proc ,[left2, top2, right2, bottom2], obj_id=obj_id)
                    smooth_dict[obj_dmg_key] = [DAMAGE_SKIP_NUM, dmg_prob]
                else:
                    dmg_prob = 0

            if dmg_prob>0.8:
                ano_dict['damaged'] = True
                cv2.putText(out_frame, f'{dmg_prob:.2f}', ((right+left)//2, (bottom+top)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # Car collision
            obj_col_key = f"{obj_id}_col"
            if obj_id in collision_id_list:
                ano_dict['collision'] = True
                smooth_dict[obj_col_key] = vid_fps//6
            elif obj_col_key in smooth_dict and smooth_dict[obj_col_key] > 0:
                ano_dict['collision'] = True
                smooth_dict[obj_col_key] -= 1



            if is_moving:
                cv2.putText(out_frame, "moving", (vid_width//2, vid_height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                # Detect lack of car distance
                is_close = detect_close_distance(left, top, right, bottom)
                ano_dict['close_distance'] = is_close
                
        elif class_name=="person":
            if is_moving and detect_jaywalker(ret_bbox4obj(frames_infos, obj_id), out_frame, (left_mean, right_mean)):
                ano_dict['jaywalker'] = True

        draw_bbox(out_frame, ano_dict, left, top, right, bottom)
    # --- frame loop end

    writer.write(out_frame)
    frames_infos.popleft()
    end = timer()
    return (end-start)*1000



def retrieve_all_car_info(all_info):
    car_list = []
    for obj_id in all_info:
        info = all_info[obj_id]
        class_id, _, bbox = info
        class_name = class_names[class_id]
        if class_name=="car" or class_name=="bus" or class_name=="truck":
            car_list.append((obj_id, bbox))
    return car_list

# car list [(obj_id, bbox),]
# return list of obj_id (car that is colliding)
def detect_car_collision(car_list):
    collision_list = []
    global vid_width,vid_height
    while len(car_list)>1:
        id1, bbox1 = car_list[0]
        box1_width, box1_height = bbox1[2]-bbox1[0], bbox1[3]-bbox1[1]

        # ignore small box
        if box1_height<vid_height//18:
            del car_list[0]
            continue

        if box1_width/(box1_height)>1.5:
            is_side1 = True
        else:
            is_side1 = False
        i = 1 # the index for the second box 
        has_match = False
        while i<len(car_list):
            id2, bbox2 = car_list[i]
            box2_width, box2_height = bbox2[2]-bbox2[0], bbox2[3]-bbox2[1]
            if box2_width/(box2_height)>1.5:
                is_side2 = True
            else:
                is_side2 = False

            if is_side1 and is_side2:
                height_thres = 0.02
            else:
                height_thres = 0.1
# (abs(bbox1[1]-bbox2[1])/box1_height) < height_thres 
            # if they have about the same bottom(height)
            # 1: two sided car i.e. left/right potion of bbox overlap
            # 2: two forward car left/right side crash

            if (abs(bbox1[3]-bbox2[3])/box1_height) < height_thres: #similar y-level bottom
                if bbox2[2]>bbox1[0] or bbox1[1]>bbox2[0] or bbox2[3]>bbox1[1] or bbox1[3]>bbox2[1]:
                    is_checked = True
                    iou_thres = 0.1
            # back car crash into front car, y-axis may not be similar
            elif bbox2[3]>bbox1[1] or bbox1[3]>bbox2[1]:
                is_checked = True
                iou_thres = 0.4
                
            if is_checked:   
                iou = compute_iou(bbox1, bbox2)
                if iou > iou_thres and \
                    iou <0.6: # to exclude some false positive due to detection fault
                    collision_list.append(id2)
                    del car_list[i]
                    has_match = True
                    i -= 1 # compensate the effect of removing element
            i += 1 #proceed to next box2
        if has_match:
            collision_list.append(id1)
        del car_list[0] #remove box1 anyway
    return collision_list
            

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


def detect_jaywalker(recent_bboxes, frame, mean_shift):
    global vid_height, vid_width
    ROI = [(vid_width//10,vid_height), (vid_width//2,vid_height*3//7), (vid_width*9//10, vid_height) ]
    cv2.polylines(frame, [np.array(ROI, dtype=np.int32)], False, (255,0,0))
    cv2.line(frame,(0, vid_height//2), (vid_width, vid_height//2), (255,0,0))
    cv2.line(frame,(0, vid_height*7//10), (vid_width, vid_height*7//10), (255,0,0))
    left, top, right, bottom = recent_bboxes[0][0]
    center_x, center_y = (left+right)//2, (top+bottom)//2

    # bottom_center = (center_x, bottom)
    if not bottom<vid_height//2:
        if inside_roi(center_x, bottom, ROI):
            if bottom>vid_height*7//10 :
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
                cv2.putText(frame, f"{(max_dist/vid_width):.2f} ", (center_x-10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                if max_dist > vid_width*0.4:
                  return True
    return False


# retrieve bounding boxes for an object in future n frames given obj_id
# return list of [bbox, x] , x = frame offset i.e. that frame is x frames after 
def ret_bbox4obj(frames_infos, obj_id, length=None):
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
    cv2.putText(out_frame, f"{left_mean:.2f} ", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    right_mean = cal_weighted_mean(rp_shift_list, rp_left_count, rp_right_count)
    cv2.putText(out_frame, f"{right_mean:.2f} ", (vid_width-50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
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
def draw_bbox(image, ano_dict, left, top, right, bottom):
    global vid_height
    thickness = vid_height//720+1
    font_size = vid_height/1080
    label = ano_dict["label"]
    # (B,G,R)
    box_color = (0,255,0) # Use green as normal 

    ano_label = ""
    if ("close_distance" in ano_dict) and ano_dict["close_distance"]:
        box_color = (70,255,255) # yellow
        ano_label += "Close "
    
    if ("jaywalker" in ano_dict) and ano_dict["jaywalker"]:
        box_color = (0,123,255) #orange
        ano_label += "Jaywalker "

    if ("damaged" in ano_dict) and ano_dict["damaged"]:
        box_color = (123,0,255)
        ano_label += "Damaged "

    if ("collision" in ano_dict) and ano_dict["collision"]:
        box_color = (0,0,255)
        ano_label += "Collision "

    # elif ("accident" in ano_dict): 
    #     box_color = (255,255,255)
    cv2.rectangle(image, (left, top), (right, bottom), box_color, thickness)
    cv2.putText(image, label, (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0,255,0), thickness)
    if not ano_label=="":
        cv2.putText(image, ano_label, ((right+left)//2, top-5), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0,0,255), thickness)


# To check whether a point(x,y) is within a triangle area of interest
# by computer the 3 traingles form with any 2 point & (x,y)
# and check if the total area of the 3 traingles equal to the triangle of interest
def inside_roi(x,y, pts):
    global vid_height, vid_width
    # x1, y1 = vid_width//2, vid_height//2
    # x2, y2 = vid_width//8, vid_height
    # x3, y3 = vid_width*7//8, vid_height
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

def detect_close_distance(left, top, right, bottom):
    global vid_height, vid_width
    pts = [(vid_width//2, vid_height//2), 
          (vid_width//8, vid_height),
          (vid_width*7//8, vid_height) ]

    # ignore roi if the bbox is too big
    # since its center is hard to be within the roi
    if (right-left)>(vid_width//2) or inside_roi((left+right)//2, (top+bottom)//2, pts):
        box_center_x = (left+right)//2
        
        if box_center_x<vid_width//3:
            frame_center_x = vid_width//3
        elif box_center_x>vid_width//3*2:
            frame_center_x = vid_width//3*2
        else:
            frame_center_x = box_center_x

        dist = euclidean_distance(box_center_x,frame_center_x,bottom,vid_height)
        if dist<(vid_height//3) :
            # print(f"distance = {dist}")
            return True
    return False

# calculate the bboxes using video resolution for the detect_camera_moving func
def get_detection_boxes():
    result = []
    global vid_height, vid_width
    boxes_x = [vid_width*0.05, vid_width*0.85]
    boxes_y = [vid_height*0.05, vid_height*0.3, vid_height*0.55]
    box_width = int(vid_width*0.1)
    box_height = int(vid_height*0.15)

    for box_x in boxes_x:
        for box_y in boxes_y:
            left, top = int(box_x), int(box_y)
            right, bottom = left+box_width, top+box_height
            result.append([left, top, right, bottom])
            # print(left, top, right, bottom)

    # add two more on top side
    top = int(vid_height*0.05)
    bottom = top+box_height
    left = int(vid_width*0.3)
    result.append([left, top, left+box_width, bottom])
    left = int(vid_width*0.6)
    result.append([left, top, left+box_width, bottom])
    # return box size and list of bboxes pos
    # return box_width*box_height, result
    global detection_boxes, detection_size
    detection_size = box_width*box_height
    detection_boxes = result
            
# detect whether the camera is moving, return img? and boolean
def detect_camera_moving(cur_frame, prev_frame, should_return_img=False):
    if prev_frame is None:
        print("Last frame")
        return False
    threshold = 0.015
    global detection_boxes, detection_size

    if should_return_img:
        return_img = cur_frame.copy()
    else: 
        return_img = None

    count = 0
    for box in detection_boxes:
        left, top, right, bottom = box
        # select out the box and convert to gray
        box_cur = cur_frame[top:bottom, left:right].copy()
        box_cur = cv2.cvtColor(box_cur, cv2.COLOR_BGR2GRAY)
        box_prev = prev_frame[top:bottom, left:right].copy()
        box_prev = cv2.cvtColor(box_prev, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(box_cur, box_prev)
        ret, result = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
        percentage = cv2.countNonZero(result)/detection_size
        if percentage>threshold:
            count+=1

        # testing purpose
        if should_return_img:
            result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            return_img[top:bottom, left:right] = result_bgr
            cv2.rectangle(return_img, (left, top), (right, bottom), (0,255,0), 2)
            label = "%.3f" % percentage
            cv2.putText(return_img, label, ((left+right)//2, (top+bottom)//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    # 8 boxes in total
    is_moving = count>3
    if is_moving:
        # testing purpose
        if should_return_img:
            global vid_width, vid_height
            cv2.putText(return_img, "Is moving", (vid_width//2, vid_height-50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    
    # retur_img is None if not returning img
    return return_img, is_moving

# small func to help display progress
def sec2length(time_sec):
    m = int(time_sec//60)
    s = int(time_sec%60)
    if s<10:
        s= "0"+str(s)
    return f"{m}:{s}" 

#omit small bboxes since they are not accurate and useful enought for detecting anomaly
def omit_small_bboxes(bboxes,classes):
    global vid_height
    area_threshold = (vid_height//36)**2

    omitted_count = 0
    i = 0
    while i<len(bboxes):
        bbox = bboxes[i]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        class_name = class_names[classes[i]]
        if (class_name=="car" or class_name=="bus" or class_name=="truck")\
            and width*height<area_threshold:
            # print(f"{classes[i]} {width}x{height}")
            del bboxes[i]
            del classes[i]
            omitted_count +=1
        else:
            i += 1
    # print(f"Omitted {omitted_count} boxes due to small size")
    return omitted_count
    
# yolo wrapper, return list of bounding boxes and list of corresponding classes(id)
def yolo_detect(frame, model, opt):
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
    bboxes = []
    classes = []
    if detections is not None:
        detections = rescale_boxes(detections, opt.img_size, frame.shape[:2])
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
           bboxes.append([x1, y1, x2, y2, cls_conf.item()])
           classes.append(int(cls_pred))
    return bboxes, classes

# for testing
def draw_test_img(test_img):
    # draw the ROI of close dist detection
    x1, y1 = vid_width//2, vid_height//2
    x2, y2 = vid_width//8, vid_height
    x3, y3 = vid_width*7//8, vid_height
    pts = np.array([[x1,y1], [x2,y2], [x3,y3]], np.int32)
    cv2.polylines(test_img, [pts], True, (255,0,0))

    pts =[(0,vid_height), (vid_width,vid_height), (vid_width//2, vid_height//4) ]
    pts = np.array(pts,  np.int32)
    cv2.polylines(test_img, [pts], True, (0,0,255))

    out_test.write(test_img)

def track_video(opt):
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0)
    print(f"Using cuda device: {device_name}")
    
    # testing flag here:
    show_fps = True

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
    vid_fps = vid.get(cv2.CAP_PROP_FPS)
    video_total_frame = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    video_length = sec2length(video_total_frame//vid_fps)
    # init video writer
    video_FourCC = cv2.VideoWriter_fourcc(*'mp4v')
    isOutput = True if output_path != "" else False
    if isOutput:
        # print("!!! TYPE:", type(output_path), type(video_FourCC), type(vid_fps), type(video_size))
        print(f"Loaded video: {output_path}, Size = {vid_width}x{vid_height},"
              f" fps = {vid_fps}, total frame = {video_total_frame}")
        if not vid_fps == int(vid_fps):
            vid_fps = int(vid_fps)
            print(f"Rounded fps to {vid_fps}")
        out_writer = cv2.VideoWriter(output_path, video_FourCC, vid_fps, (vid_width, vid_height))
        
    output_test = True  
    if output_test:
        test_output_path =  output_path.replace("output", "test")
        test_writer = cv2.VideoWriter(test_output_path, video_FourCC, vid_fps, (vid_width, vid_height))

  # global init
    get_detection_boxes()
    global class_names
    class_names = load_classes(opt.class_path)
  # init SORT tracker
    max_age = max(3,vid_fps//2)
    mot_tracker = Sort(max_age=max_age, min_hits=1)
    print("SORT initialized")
  # init yolov3 model
    yolo_model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    yolo_model.load_darknet_weights(opt.weights_path)
    yolo_model.eval()
    print("YOLO model loaded")
    global damage_detector
    damage_detector = Damage_detector(device)
    # lane_model = Lane_model(device)
    # seg_model = Seg_model(device)
    # print("Fast-SCNN model loaded")
    # erf_model = Erf_model(device)

    # start iter frames
    in_frame_no, proc_frame_no = 0, 1
    buffer_size = vid_fps #store 1sec of frames
    prev_frames = deque()
    frames_infos = deque()

    while True:
        start = timer()
        success, frame = vid.read()
        if not success: #end of video
            break
        in_frame_no += 1
        # seg_img = seg_model.detect(frame)
        # print(seg_img.shape)
        # test_writer.write(seg_img)
        # erf_model.detect(frame, frame_no)

        # Obj Detection
        bboxes, classes = yolo_detect(frame, yolo_model, opt)
        omitted_count = omit_small_bboxes(bboxes, classes)
      
        # tracker_infos is added to return link the class name & the object tracked
        trackers, tracker_infos = mot_tracker.update(np.array(bboxes), np.array(classes))

        id_to_info = {}
        for c, d in enumerate(trackers):
            d = d.astype(np.int32) 
            left, top, right, bottom = int(d[0]), int(d[1]), int(d[2]), int(d[3])
            obj_id = d[4]
            class_id, score = tracker_infos[c][0], tracker_infos[c][1]
            class_name = class_names[class_id]
            if score == -1: #detection is missing
              continue

            info = [class_id, score, [left, top, right, bottom]]
            id_to_info[obj_id] = info

        # frame buffer proc
        if len(prev_frames)==buffer_size:
            proc_ms = proc_frame(out_writer, prev_frames, frames_infos, test_writer)
            proc_frame_no += 1
        prev_frames.append(frame)
        frames_infos.append(id_to_info)
        
        end = timer()
        msg = f"[{sec2length(in_frame_no//vid_fps)}/{video_length}]"+\
        f"  Found {len(bboxes)} boxes  | {omitted_count} omitted "
        #calculate fps by 1sec / time consumed to process this frame
        fps = str(round(1/(end-start),2))
        msg += (f"--fps: {fps}")
        print(msg)
        if proc_frame_no>1:
            print(f">> Processing frame {proc_frame_no}, time: {proc_ms:.2f}ms")

    # Process the remaining frames in buffer
    while len(frames_infos)>0:
        proc_ms = proc_frame(out_writer, prev_frames, frames_infos, test_writer)
        proc_frame_no += 1
        print(f">> Processing frame {proc_frame_no}, time: {proc_ms:.2f}ms")
    

    if isOutput:
        out_writer.release()
    if output_test:
        test_writer.release()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="model_data/bdd/bdd.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="model_data/bdd/bdd.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="model_data/bdd/classes.txt", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.55, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.25, help="iou thresshold for non-maximum suppression")
    # parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    # parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    # parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")

    parser.add_argument("--input", nargs='?', type=str, default="",help = "Video input path")
    parser.add_argument("--output", nargs='?', type=str, default="",  help = "[Optional] Video output path")
    opt = parser.parse_args()

    track_video(opt)
