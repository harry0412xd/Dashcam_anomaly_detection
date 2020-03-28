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
from yolov3.utils.utils import load_classes
from detector import is_car

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
    lognPrint(f"Start loading {video_path}...")
    pbar = tqdm.tqdm(total=video_total_frame)

    frame_no = 0
    while True:
        frame_no += 1
        pbar.update(1)
        success, frame = vid.read()
        if not success: #end of video
            break
        out_frame = frame.copy()


        id_to_info = use_det_result(all_results, frame_no)
        for obj_id in id_to_info:
            info = id_to_info[obj_id]
            class_id, score, bbox = info
            if not is_car(class_names[class_id]):
                continue
            left, top, right, bottom = bbox

            # copy from main py file 2020/3/24

            # dmg_height_thres, dmg_width_thres = vid_height//12, vid_width//24
            dmg_height_thres, dmg_width_thres = 64, 64
            if not DC.IGNORE_SMALL or ((bottom-top)>dmg_height_thres and (right-left)>dmg_width_thres) :

                dmg_prob = damage_detector.detect(frame, bbox, id_to_info, frame_no, obj_id)

                for p_thres in p_thres_list:
                    if dmg_prob>p_thres: #positive
                        if get_damage_truth(obj_id, frame_no) is not None: # True positive
                            mode = 1
                        else:# False positive
                            mode = 2
                    
                    else:# negative
                        if get_damage_truth(obj_id, frame_no) is None: # True negative
                            mode = 3
                        else:# False negative
                            mode = 4
                    update_metric(obj_id, mode, p_thres)

            else:
                dmg_prob = -1
              
            draw_bbox(out_frame, obj_id, dmg_prob, bbox, frame_no)
            # # to construct a metric_dict case_id -> true positive, true negative, false positive, false negative total
            # # accuracy = hit / total
            # if obj_id not in metric_dict:
            #     metric_dict[obj_id] = 0, end_frame_no-start_frame_no+1
            # else:
            #     hit, total = metric_dict[obj_id]
            #     metric_dict[obj_id] = hit, total + end_frame_no-start_frame_no+1

        out_writer.write(out_frame)
        
    
    m_thres_list = [0.25,0.33,0.5,0.75]
    compute_case_metric(m_thres_list, p_thres_list)
    compute_total_metric()

# draw bounding box on image given label and coordinate
def draw_bbox(image, obj_id, dmg_prob, bbox, frame_no):
    left, top, right, bottom = bbox
    thickness = vid_height//720+1
    font_size = vid_height/1080

    # if obj_id in obj_id_to_truth:
    #     if obj_id_to_truth[obj_id] is not None: # is damaged (truth)

    label = f"#{obj_id}"

    case_id = None
    if obj_id in obj_id2case_id:
        case_id = obj_id2case_id[obj_id]

        if opt.dmg_thres in case_metrics:
            case_metric = case_metrics[opt.dmg_thres]
            if case_id in case_metric:
                total, tp, fp, tn, fn = case_metric[case_id]
                label += f"a:{tp+tn}/{total} |p: {tp}/{tp+fp} |r: {tp}/{tp+fn}"
        label = f"C{case_id}-" + label


    if case_id is not None:
        if get_damage_truth(obj_id, frame_no):
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
    cv2.putText(image, label, ((right+left)//2, top-5), cv2.FONT_HERSHEY_SIMPLEX, font_size*0.8, label_color, thickness)

    # print damage score
    cv2.putText(image, f"{dmg_prob:.2f}", ((right+left)//2, (top+bottom)//2), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), thickness)

    # print recall related

    
def compute_total_metric():
    for p_thres in p_thres_list:
        total_metric = total_metrics[p_thres]
        total, tp, fp, tn, fn = total_metric
        lognPrint(f"Results (all):  Acc:{tp+tn}/{total} |prec: {tp}/{tp+fp} |recall: {tp}/{tp+fn}")


def compute_case_metric(m_thres_list, p_thres_list):
    for p_thres in p_thres_list:

        prec_case_wise, recall_case_wise = {}, {}
        case_metric = case_metrics[p_thres]
        for case_id in case_metric:
            total, tp, fp, tn, fn = case_metric[case_id]
            # lognPrint(f"Case {case_id}: obj id: {case_id2obj_id[case_id]} ")
            # lognPrint(f"---- tp: {tp} fp: {fp} tn: {tn} fn:{fn}")
            lognPrint(f"Case {case_id}: obj id: {case_id2obj_id[case_id]}  ----  tp: {tp} fp: {fp} tn: {tn} fn:{fn}")
            log_csv("{case_id},{tp},{fp},{tn},{fn},{acc},{prec},{recall}")

            acc = (tp + tn) / total
            if tp >0:
                recall = tp / (tp + fn)
                prec = tp / (tp + fp)
            else:
                recall=prec=0
            # print(f"Case {case_id}: prec: {prec} recall: {recall} acc: {acc}")
            lognPrint(f"----prec: {prec} recall: {recall} acc: {acc}")

            for m_thres in m_thres_list:
                if recall>=m_thres:
                    recall_case_wise[m_thres] = recall_case_wise[m_thres]+1 if (m_thres in recall_case_wise) else 1
                if prec>=m_thres:
                    prec_case_wise[m_thres] = prec_case_wise[m_thres]+1 if (m_thres in prec_case_wise) else 1

        # prec
        for m_thres in m_thres_list:
            lognPrint(f"Precision@{int(m_thres*100)}% = {prec_case_wise[m_thres]}/{len(case_metric)} = {prec_case_wise[m_thres]/len(case_metric)}") # prec
            lognPrint(f"Recall@{int(m_thres*100)}% = {recall_case_wise[m_thres]}/{len(case_metric)} = {recall_case_wise[m_thres]/len(case_metric)}") # recall



def update_metric(obj_id, mode, p_thres):
  global total_metrics, case_metrics
  # mode 1: True positive, 2: False positive 3: True negative 4: False negative
  if p_thres in total_metrics:
      total_metric = total_metrics[p_thres]
  else:
      total_metric = [0,0,0,0,0]# total, tp, fp, tn, fn
  total_metric[0] += 1 
  total_metric[mode] += 1
  total_metrics[p_thres] = total_metric
  

  if obj_id in obj_id2case_id:
      case_id = obj_id2case_id[obj_id]
      if p_thres in case_metrics:
          case_metric = case_metrics[p_thres]
      else:
          case_metric = {}

      if not case_id in case_metric:
          case_metric[case_id] = [0,0,0,0,0] # total, tp, fp, tn, fn
      case_metric[case_id][0] += 1
      case_metric[case_id][mode] += 1
      case_metrics[p_thres] = case_metric
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

            # for convenience
            obj_id2case_id[obj_id] = case_id

            if case_id not in case_id2obj_id:
                case_id2obj_id[case_id] = []
            case_id2obj_id[case_id].append(obj_id)

    label_file.close()
    lognPrint(f"Loaded {len(obj_id_to_truth)} labels from {label_path}, {max_case_id} test cases.")

    # return obj_id_to_truth


def get_damage_truth(obj_id, frame_no):
    if obj_id in obj_id_to_truth:
        for (start_frame_no, end_frame_no) in obj_id_to_truth[obj_id]:

            if frame_no >= int(start_frame_no) and frame_no <= int(end_frame_no):
                return obj_id2case_id[obj_id]
    return None


# log and print
def lognPrint(text):
    print(text)
    log_path = "eval_results/" + opt.log
    with open(log_path, 'a') as log_file:
        log_file.write(text + '\n')


def log_csv(text):
    log_path = "eval_results/" + opt.log
    log_path.replace(".txt", ".csv")
    with open(log_path, 'a') as log_file:
        log_file.write(text + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # I/O
    parser.add_argument("--input", nargs='?', type=str, default="video_in/Crash_damage.mp4",help = "Video input path")
    parser.add_argument("--output", nargs='?', type=str, default="video_out/Crash_damage_eval.mp4",  help = "[Optional] Video output path")
    parser.add_argument('--result_path', type=str, default="detection_results/Crash_damage.txt", help = "Path of file which save the Object detection/tracking results")
    parser.add_argument('--label_path', type=str, default="crash_damage_label.txt", help = "Path of label file indicating the damaged car(s)")
    parser.add_argument('--dmg_thres', type=float, default=0.8, help = "Thershold of confidence to consider a car as damaged")
    parser.add_argument("--device", type=str, default="cuda",  help = "Use cuda or cpu")
    parser.add_argument("--log", type=str, default="eval_result.txt",  help = "Use cuda or cpu")
    parser.add_argument("--class_path", type=str, default="/content/Dashcam_anomaly_detection/model_data/YOLOv3_bdd/classes.txt",  help = "YOLO classes list")
    
  
    opt = parser.parse_args()
    obj_id_to_truth, obj_id2case_id, case_id2obj_id, case_metrics, total_metrics = {}, {}, {}, {}, {}
    load_damage_label(opt.label_path)
    vid_width, vid_height = 0, 0
    p_thres_list = [0.75, 0.8, 0.85, 0.9]
    assert opt.dmg_thres in p_thres_list, "Change the list above"
    class_names = load_classes(opt.class_path)

    damage_detector = Damage_detector(opt.device, do_erasing=DC.DO_ERASING, do_padding=DC.DO_PADDING, side_thres=DC.SIDE_THRES)
    lognPrint(f"Loaded Model weight: {damage_detector.get_checkpoint_path()}")
    lognPrint(f"Threshold: {opt.dmg_thres}")
    log_csv("case_id,tp,fp,tn,fn,acc,prec,recall")


    evaluate()

    