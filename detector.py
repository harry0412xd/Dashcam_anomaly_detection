import sys
import argparse
from timeit import default_timer as timer

import cv2
import numpy as np

from yolo import YOLO
from sort import *


# draw bounding box on image given label and coordinate
def draw_bbox(image, label, left, top, right, bottom):
    cv2.rectangle(image, (left, top), (right, bottom), (0,255,0), 2)
    cv2.rectangle(image, (left, top), (right, bottom), (0,255,0), 2)
    cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

def track_video(yolo, video_path, output_path=""):
    show_fps = True
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = cv2.VideoWriter_fourcc(*'MP4V')
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_total_frame = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    thickness = min((vid_width + vid_height) // 300, 3)

    isOutput = True if output_path != "" else False
    if isOutput:
        # print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        print(f"Loaded video: {output_path}, Size = {vid_width}x{vid_height},"
              f" fps = {video_fps}, total frame = {video_total_frame}")
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, (vid_width, vid_height))

    # init SORT tracker
    max_age = max(3,video_fps//6)
    mot_tracker = Sort(max_age=max_age, min_hits=3)

    frame_no = 0
    # object_class_dict = {}
    while True:
        start = timer()
        success, frame = vid.read()
        if not success:
            break
        frame_no += 1
        bboxes, classes = yolo.detect_image(frame)
        print(f'Found {len(bboxes)} boxes for frame {frame_no}/{video_total_frame}')
        
        #omit small bboxes since they are not accurate and useful enought for detecting anomaly
        omit_small_box = True
        if omit_small_box:
            omitted_count = 0
            i = 0
            while i<len(bboxes):
                bbox = bboxes[i]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                if width*height<(vid_height//36)**2:
                    # print(f"{classes[i]} {width}x{height}")
                    del bboxes[i]
                    del classes[i]
                    omitted_count +=1
                else:
                    i += 1
            print(f"Omitted {omitted_count} boxes due to small size")

        trackers, tracker_infos = mot_tracker.update(np.array(bboxes), np.array(classes))
        for c, d in enumerate(trackers):
            d = d.astype(np.int32) 
            left, top, right, bottom = int(d[0]), int(d[1]), int(d[2]), int(d[3])
            obj_id = d[4]

            class_id = tracker_infos[c][0]
            class_name = yolo.class_names[class_id]
            score = tracker_infos[c][1]
            label = f'{class_name} {obj_id} : {score:.2f}'
            print (f"{label} at {left},{top}, {right},{bottom}")
            draw_bbox(frame, label, left, top, right, bottom)

        end = timer()
        if show_fps:
            #calculate fps by 1sec / time consumed to process this frame
            fps = str(round(1/(end-start),2))
            # print(f"fps: {fps}")
            cv2.putText(frame, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.0, color=(255, 0, 0), thickness=2)

        if isOutput:
            out.write(frame)
        # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        # cv2.imshow("result", result)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    yolo.close_session()

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str, default="bdd/bdd.h5",
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )
    parser.add_argument(
        '--anchors_path', type=str,default="bdd/anchors.txt",
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )
    parser.add_argument(
        '--classes_path', type=str,default="bdd/classes.txt",
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
