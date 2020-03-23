"""
Run this to evaulate the performance of the car damage detection model

"""
import argparse
from utils import load_det_result, use_det_result
import torch
from damage_detector import Damage_detector



def evaluate():
    video_path = opt.input
    output_path = opt.output
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    all_results = load_det_result(opt.load_result)
    while True:
        success, frame = vid.read()
        if not success: #end of video
            break
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
                    dmg_prob = damage_detector.detect(frame2proc, bbox, padding_size=(x_pad, y_pad),
                                                      frame_info = id_to_info, erase_overlap=True, obj_id=obj_id)
                else:
                    dmg_prob = damage_detector.detect(frame2proc, bbox, padding_size=(x_pad, y_pad))
            else:
                dmg_prob = 0

            if dmg_prob>opt.dmg_thres: #positive
                if get_damage_truth is not None: # True positive

                else:# False positive
            else:
                if get_damage_truth is None: # True negative

                else:# False negative

                
            # to construct a metric_dict case_id -> true positive, true negative, false positive, false negative total
            # accuracy = hit / total
            if obj_id not in metric_dict:
                metric_dict[obj_id] = 0, end_frame_no-start_frame_no+1
            else:
                hit, total = metric_dict[obj_id]
                metric_dict[obj_id] = hit, total + end_frame_no-start_frame_no+1

def update_metric(obj_id, mode):
  # mode 1: True positive, 2: False positive 3: True negative 4: False negative

        
# for eval car damage detection
def load_damage_label(label_path):
    global obj_id_to_truth, obj_id2case_id = {}
    last_frame_no = -1
    frame_results = []
    max_case_id = 0
    with open(label_path, "r") as label_file:
        for line in label_file:
            _line = line.strip()
            case_id, obj_id, start_frame_no, end_frame_no, _, _ = _line.split(",")
            max_case_id = case_id if case_id>max_case_id else max_case_id

            if obj_id not in damaged_truth:
                obj_id_to_truth[obj_id] = []
            obj_id_to_truth[obj_id].append([case_id, start_frame_no, end_frame_no])

            obj_id2case_id[obj_id_to_truth] = case_id

    label_file.close()
    print(f"Loaded {len(damaged_truth)} labels from {label_path}, {max_case_id} test cases.")

    return obj_id_to_truth

def get_damage_truth(obj_id, frame_no):
    if obj_id in obj_id_to_truth:
        for instance in obj_id_to_truth[obj_id]:
            case_id, start_frame_no, end_frame_no = instance
            if frame_no >= start_frame_no and frame_no <= end_frame_no:
                return case_id
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # I/O
    parser.add_argument("--input", nargs='?', type=str, default="",help = "Video input path")
    parser.add_argument("--output", nargs='?', type=str, default="",  help = "[Optional] Video output path")
    parser.add_argument('--result_path', type=str, default="", help = "Path of file which save the Object detection/tracking results")
    parser.add_argument('--label_path', type=str, default="", help = "Path of label file indicating the damaged car(s)")
    parser.add_argument('--dmg_thres', type=float, default=0.8, help = "Thershold of confidence to consider a car as damaged")
    opt = parser.parse_args()
    obj_id_to_truth, obj_id2case_id = None, None
    load_damage_label(label_path):
    evaluate()

    