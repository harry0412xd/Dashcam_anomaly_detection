"""
Run this to evaulate the performance of the car damage detection model

"""
import argparse
import torch
import cv2
import tqdm 

from damage_detector import Damage_detector
import detector_config as DC
from utils import load_det_result, use_det_result


def evaluate():
    video_path = opt.input
    output_path = opt.output
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    global vid_width, vid_height
    vid_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_FourCC = cv2.VideoWriter_fourcc(*'mp4v')
    vid_fps = vid.get(cv2.CAP_PROP_FPS)
    video_total_frame = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    out_writer = cv2.VideoWriter(output_path, video_FourCC, vid_fps, (vid_width, vid_height))

    all_results = load_det_result(opt.result_path)
    print("Start loading video...")
    pbar = tqdm.tqdm(total=video_total_frame)

    in_frame_no = 0
    while True:
        in_frame_no += 1
        pbar.update(1)
        success, frame = vid.read()
        if not success: #end of video
            break
        out_frame = frame.copy()


        id_to_info = use_det_result(all_results, in_frame_no)
        for obj_id in id_to_info:
            info = id_to_info[obj_id]
            class_id, score, bbox = info
            left, top, right, bottom = bbox

            # copy from main py file 2020/3/24

            # dmg_height_thres, dmg_width_thres = vid_height//12, vid_width//24
            dmg_height_thres, dmg_width_thres = 64, 64
            if not DC.IGNORE_SMALL or ((bottom-top)>dmg_height_thres and (right-left)>dmg_width_thres) :
                if DC.DO_PADDING:
                    if (right-left)/(bottom-top) > DC.IS_SIDE_RATIO :
                        x_pad, y_pad = (right-left)//8, (bottom-top)//12
                    else:
                        x_pad, y_pad = (right-left)//12, (bottom-top)//12
                else:
                    x_pad, y_pad = 0,0

                if DC.DO_ERASING:
                    dmg_prob = damage_detector.detect(frame, bbox, padding_size=(x_pad, y_pad),
                                                      frame_info = id_to_info, erase_overlap=True, obj_id=obj_id)
                else:
                    dmg_prob = damage_detector.detect(frame, bbox, padding_size=(x_pad, y_pad))


                if dmg_prob>=opt.dmg_thres: #positive
                    if get_damage_truth(obj_id, in_frame_no) is not None: # True positive
                        mode = 1
                    else:# False positive
                        mode = 2
                
                else:# negative
                    if get_damage_truth(obj_id, in_frame_no) is None: # True negative
                        mode = 3
                    else:# False negative
                        mode = 4

                update_metric(obj_id, mode)

            else:
                dmg_prob = -1
              
            draw_bbox(out_frame, obj_id, dmg_prob, bbox)
            # # to construct a metric_dict case_id -> true positive, true negative, false positive, false negative total
            # # accuracy = hit / total
            # if obj_id not in metric_dict:
            #     metric_dict[obj_id] = 0, end_frame_no-start_frame_no+1
            # else:
            #     hit, total = metric_dict[obj_id]
            #     metric_dict[obj_id] = hit, total + end_frame_no-start_frame_no+1

        out_writer.write(out_frame)
        
    compute_case_metric([0.25,0.50,0.75])

# draw bounding box on image given label and coordinate
def draw_bbox(image, obj_id, dmg_prob, bbox):
    left, top, right, bottom = bbox
    thickness = vid_height//720+1
    font_size = vid_height/1080

    if obj_id in obj_id_to_truth:
        if obj_id_to_truth[obj_id] is not None: # is damaged (truth)
            label_color = (123,0,255)
        else:
            label_color = (255,255,0)
    else:
        label_color = (0, 255, 0)
    
    if dmg_prob>=opt.dmg_thres:
        box_color = (123,0,255)
    else:
        box_color = (0, 255, 0)

    cv2.rectangle(image, (left, top), (right, bottom), box_color, thickness)

    # print id
    cv2.putText(image, str(obj_id), ((right+left)//2, top-5), cv2.FONT_HERSHEY_SIMPLEX, font_size, label_color, thickness)

    # print damage score
    cv2.putText(image, str(obj_id), ((right+left)//2, (top+bottom)//2), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), thickness)

    

def compute_case_metric(thres_list):
    case_count = {}
    for case_id in case_metric:
        total, true_positive, false_positive, true_negative, false_negative = case_metric[case_id]
        print(f"Case {case_id}: tp: {true_positive} fp: {false_positive} tn: {true_negative} fn:{false_negative}")
        acc = (true_positive + true_negative) / total
        if true_positive >0:
            recall = true_positive / (true_positive + false_negative)
            prec = true_positive / (true_positive + false_positive)
        else:
            recall=prec=0
        # print(f"Case {case_id}: prec: {prec} recall: {recall} acc: {acc}")
        print(f"----prec: {prec} recall: {recall} acc: {acc}")

        for thres in thres_list:
            if recall>thres:
                case_count[thres] = case_count[thres]+1 if (thres in case_count) else 1
    
    for thres in thres_list:
        print(f"Recall@{int(thres*100)}% = {case_count[thres]}/{len(case_metric)} = {case_count[thres]/len(case_metric)}")


def update_metric(obj_id, mode):
  global total_metric
  # mode 1: True positive, 2: False positive 3: True negative 4: False negative
  total_metric[0] += 1 
  total_metric[mode] += 1
  
  if obj_id in obj_id2case_id:
      case_id = obj_id2case_id[obj_id]
      if not case_id in case_metric:
          case_metric[case_id] = [0,0,0,0,0] # total, tp, fp, tn, fn
      case_metric[case_id][0] += 1
      case_metric[case_id][mode] += 1
  # else not in case

# for eval car damage detection
def load_damage_label(label_path):
    global obj_id_to_truth, obj_id2case_id
    last_frame_no = -1
    frame_results = []
    max_case_id = 0
    is_first_line = True
    with open(label_path, "r") as label_file:
        for line in label_file:
            if is_first_line:#ignore table header
                is_first_line = False
                continue
            _line = line.strip()
            # print(_line)
            case_id, obj_id, start_frame_no, end_frame_no, _, _ = _line.split(",")
            case_id = int(case_id)
            max_case_id = case_id if case_id>max_case_id else max_case_id

            if not obj_id in obj_id_to_truth:
                obj_id_to_truth[obj_id] = []
            obj_id_to_truth[obj_id].append([start_frame_no, end_frame_no])

            obj_id2case_id[obj_id] = case_id

    label_file.close()
    print(f"Loaded {len(obj_id_to_truth)} labels from {label_path}, {max_case_id} test cases.")

    # return obj_id_to_truth

def get_damage_truth(obj_id, frame_no):
    if obj_id in obj_id_to_truth:
        for (start_frame_no, end_frame_no) in obj_id_to_truth[obj_id]:

            if frame_no >= int(start_frame_no) and frame_no <= int(end_frame_no):
                return obj_id2case_id[obj_id]
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # I/O
    parser.add_argument("--input", nargs='?', type=str, default="video_in/Crash_damage.mp4",help = "Video input path")
    parser.add_argument("--output", nargs='?', type=str, default="video_out/Crash_damage_eval.mp4",  help = "[Optional] Video output path")
    parser.add_argument('--result_path', type=str, default="detection_results/Crash_damage.txt", help = "Path of file which save the Object detection/tracking results")
    parser.add_argument('--label_path', type=str, default="crash_damage_label.txt", help = "Path of label file indicating the damaged car(s)")
    parser.add_argument('--dmg_thres', type=float, default=0.8, help = "Thershold of confidence to consider a car as damaged")
    parser.add_argument("--device", type=str, default="cuda",  help = "Use cuda or cpu")
    opt = parser.parse_args()
    obj_id_to_truth, obj_id2case_id, case_metric = {}, {}, {}
    total_metric = [0,0,0,0,0]# total, tp, fp, tn, fn
    load_damage_label(opt.label_path)
    vid_width, vid_height = 0, 0
    damage_detector = Damage_detector(opt.device)
    print("Loaded damage model")
    evaluate()

    