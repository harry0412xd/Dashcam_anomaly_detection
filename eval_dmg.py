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
        
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # I/O
    parser.add_argument("--input", nargs='?', type=str, default="",help = "Video input path")
    parser.add_argument("--output", nargs='?', type=str, default="",  help = "[Optional] Video output path")
    parser.add_argument('--load_result', type=str, default="", help = "Path of file which save the Object detection/tracking results")
    parser.add_argument('--label', type=str, default="", help = "Path of label file indicating the damaged car(s)")
    opt = parser.parse_args()
    evaluate()

    