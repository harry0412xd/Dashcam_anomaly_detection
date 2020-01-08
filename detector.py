import sys
import argparse
import math
from timeit import default_timer as timer
from collections import deque

import cv2
import numpy as np

from yolo import YOLO
from sort import *

# from utils.speed_est.speed_est import Speed_Est


#Global Variable
#Video properties : 
vid_width = 0
vid_height = 0


# draw bounding box on image given label and coordinate
def draw_bbox(image, ano_dict, left, top, right, bottom):
    global vid_height
    thickness = vid_height//720+1
    font_size = vid_height/1080
    label = ano_dict["label"]
    # (B,G,R)
    box_color = (0,255,0) # Use greem as normal 

    ano_label = ""
    if ("close_distance" in ano_dict) and ano_dict["close_distance"]:
        box_color = (0,0,255)
        ano_label += "Close distance "
    
    if ("jaywalker" in ano_dict) and ano_dict["jaywalker"]:
        box_color = (0,0,255)
        ano_label += "Jaywalker "
    
    cv2.rectangle(image, (left, top), (right, bottom), box_color, thickness)
    cv2.putText(image, label, (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0,255,0), thickness)
    if not ano_label=="":
        cv2.putText(image, ano_label, ((right+left)//2, top-5), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0,0,255), thickness)

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
        if width*height<area_threshold:
            # print(f"{classes[i]} {width}x{height}")
            del bboxes[i]
            del classes[i]
            omitted_count +=1
        else:
            i += 1
    # print(f"Omitted {omitted_count} boxes due to small size")
    return omitted_count

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

    return box_width*box_height, result
            

def detect_camera_moving(cur_frame, prev_frame, size, boxes, should_return_img=False):
    threshold = 0.015
    if should_return_img:
        return_img = cur_frame.copy()
    else: 
        return_img = None

    count = 0
    for box in boxes:
        left, top, right, bottom = box
        # select out the box and convert to gray
        box_cur = cur_frame[top:bottom, left:right].copy()
        box_cur = cv2.cvtColor(box_cur, cv2.COLOR_BGR2GRAY)
        box_prev = prev_frame[top:bottom, left:right].copy()
        box_prev = cv2.cvtColor(box_prev, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(box_cur, box_prev)
        ret, result = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
        percentage = cv2.countNonZero(result)/size
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
    
    return return_img, is_moving

def sec2length(time_sec):
    m = int(time_sec//60)
    s = int(time_sec%60)
    if s<10:
        s= "0"+str(s)
    return f"{m}:{s}" 

def track_video(yolo, video_path, output_path=""):
    show_fps = True
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    
    video_FourCC = cv2.VideoWriter_fourcc(*'mp4v')
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_total_frame = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    video_length = sec2length(video_total_frame//video_fps)
    global vid_width, vid_height
    vid_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    thickness = min((vid_width + vid_height) // 300, 3)
    detection_size, detection_boxes = get_detection_boxes()
    isOutput = True if output_path != "" else False
    if isOutput:
        # print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        print(f"Loaded video: {output_path}, Size = {vid_width}x{vid_height},"
              f" fps = {video_fps}, total frame = {video_total_frame}")
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, (vid_width, vid_height))
        
    # Testing purpose
    output_test = True
    if output_test:
        test_output_path =  output_path.replace("output", "test")
        out_test = cv2.VideoWriter(test_output_path, video_FourCC, video_fps, (vid_width, vid_height))

    # init SORT tracker
    max_age = max(3,video_fps//2)
    mot_tracker = Sort(max_age=max_age, min_hits=1)

    frame_no = 0
    # object_class_dict = {}

    
    buffer_size = video_fps//2 # store half second of frames
    prev_frames = deque(maxlen=buffer_size)
    frames_info = deque(maxlen=buffer_size)

    # est = None
    while True:
        start = timer()
        success, frame = vid.read()
        if not success:
            break
        frame_no += 1
        out_frame = frame.copy()

        if frame_no==1 :
            is_moving = True # Always treat the first frame as moving
            # speed = 0
            # est = Speed_Est(frame)
        else:
            # speed = est.predict(frame)
            test_img, is_moving = detect_camera_moving(frame, prev_frames[0], detection_size, detection_boxes)
            if output_test:
                test_img, is_moving = detect_camera_moving(frame, prev_frame[0], detection_size, detection_boxes, output_test)

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


        bboxes, classes = yolo.detect_image(frame)
        omitted_count = omit_small_bboxes(bboxes, classes)
        # print(f'Found {len(bboxes)} boxes for frame {frame_no}/{video_total_frame}')
        print(f"[{sec2length(frame_no//video_fps)}/{video_length}] [{frame_no}/{video_total_frame}]"+
                f"  Found {len(bboxes)} boxes  | {omitted_count} omitted  ")


        # tracker_infos is added to return link the class name & the object tracked
        trackers, tracker_infos = mot_tracker.update(np.array(bboxes), np.array(classes))

        for c, d in enumerate(trackers):
            d = d.astype(np.int32) 
            left, top, right, bottom = int(d[0]), int(d[1]), int(d[2]), int(d[3])
            obj_id = d[4]

            class_id = tracker_infos[c][0]
            class_name = yolo.class_names[class_id]
            score = tracker_infos[c][1]
            label = f'{class_name} {obj_id} : {score:.2f}'
            print (f"  {label} at {left},{top}, {right},{bottom}")

            ano_dict = {"label": label}
            # Anomaly binary classifiers :
            if is_moving:
                if class_name=="car" or class_name=="bus" or class_name=="truck":
                    is_close = detect_close_distance(left, top, right, bottom)
                    ano_dict['close_distance'] = is_close
                    if is_close :
                        print (f"Object {obj_id} is too close ")

                elif class_name=="person":
                    center_x, center_y = (left+right)//2, (top+bottom)//2

                    # left_area = [(0,0), (0,vid_height), (vid_width//4,0)]
                    # right_area = [(vid_width,0), (vid_width,vid_height), (vid_width//4*3,0)]
                    ROI = [(0,vid_height), (vid_width,vid_height), (vid_width//2, vid_height//4) ]

                    # if the camera is moving, any person in the middle should be abnormal
                    # if not (inside_roi(center_x, center_y, left_area) 
                    #         or inside_roi(center_x, center_y, right_area)):
                    if inside_roi(center_x, center_y, ROI):
                        ano_dict['jaywalker'] = True

            draw_bbox(out_frame, ano_dict, left, top, right, bottom)



        # cv2.putText(out_frame, str(speed), org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #                 fontScale=1.0, color=(255, 0, 0), thickness=2)
        if len(prev_frames)=buffer_size:
            frame2proc = prev_frames.pop()

        prev_frames.appendleft(frame) 

        end = timer()
        if show_fps:
            #calculate fps by 1sec / time consumed to process this frame
            fps = str(round(1/(end-start),2))
            print(f"--fps: {fps}")
            cv2.putText(out_frame, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.0, color=(255, 0, 0), thickness=2)

        if isOutput:
            out.write(out_frame)
    out.release()
    if output_test:
        out_test.release()
    yolo.close_session()

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str, default="model_data/bdd/bdd.h5",
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )
    parser.add_argument(
        '--anchors_path', type=str,default="model_data/bdd/anchors.txt",
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )
    parser.add_argument(
        '--classes_path', type=str,default="model_data/bdd/classes.txt",
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )
    parser.add_argument(
        '--gpu_num', type=int,default=1,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default="",
        help = "Video input path"
    )
    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )
    parser.add_argument(
        "--score", nargs='?', type=float, default="0.5",
        help = "Confidence thershold for YOLO detection"
    )
    parser.add_argument(
        "--iou", nargs='?', type=float, default="0.25",
        help = "IoU thershold for YOLO Non-max Suppression"
    )    

    FLAGS = parser.parse_args()
    if "input" in FLAGS:
        track_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
