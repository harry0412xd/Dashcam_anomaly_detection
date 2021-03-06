import torch
import cv2
import math
import numpy as np
import sys
import os.path

from PIL import Image
from torchvision import transforms

from timm.models import create_model, resume_checkpoint, convert_splitbn_model
from timm.utils import *
from torchvision import models

class Damage_detector():
    def __init__(self, device, do_erasing=False, do_padding=False, side_thres=1.6, save_probs=False, 
                 prob_period=2, weighted_prob=False, conf_thres=0.8, very_conf_thres=0.9, output_test_image=False):
        # url = "https://github.com/harry0412xd/Dashcam_anomaly_detection/releases/download/v1.0/gluon_seresnext101_32x4d-244_checkpoint-69.pth.tar"
        # checkpoint_path = "model_data/gluon_seresnext101_32x4d-244_checkpoint-69.pth.tar"
        url = "https://github.com/harry0412xd/Dashcam_anomaly_detection/releases/download/v1.0/gluon_seresnext101_32x4d-244_checkpoint-228_cleaned.pth-c36f9352.pth"
        checkpoint_path = "model_data/gluon_seresnext101_32x4d-244_checkpoint-228_cleaned.pth-c36f9352.pth"
        if not os.path.isfile(checkpoint_path):
            torch.utils.model_zoo.load_url(url, model_dir="model_data/")
        model = create_model('gluon_seresnext101_32x4d', num_classes=2, checkpoint_path = checkpoint_path)

        model.to(device)
        model.eval()
        self.device = device
        self.model = model
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.checkpoint_path = checkpoint_path
        self.test_counter = {}
        self.output_test_image = output_test_image
        self.do_erasing = do_erasing
        self.do_padding = do_padding
        self.prob_period = prob_period # |<- n-th before --- current --- nth after->|
        self.side_thres = side_thres
        self.save_probs = save_probs
        self.weighted_prob = weighted_prob
        self.id2probs = {}
        self.conf_thres = conf_thres
        self.very_conf_thres = very_conf_thres


    def detect(self, frame, bbox, frame_info, frame_no, obj_id):

        # Extend the bounding box by a bit to include more pixels
        if self.do_padding:
            left, top, right, bottom = bbox
            if (right - left) / (bottom - top) > self.side_thres:
                x_pad, y_pad = (right - left) // 8, (bottom - top) // 12
            else:
                x_pad, y_pad = (right - left) // 12, (bottom - top) // 12
        else:
            x_pad, y_pad = 0, 0
        cropped_img = crop_and_pad(frame, bbox, (x_pad, y_pad))

        if self.do_erasing:
            assert frame_info is not None, "Other bounding boxes are necessary to find overlapped region. Pass the frame_info param."
            erase_overlapped(cropped_img, bbox, frame_info, (x_pad, y_pad))

        img_RGB = cv2.cvtColor(cropped_img,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_RGB)
        img = self.transform(img)
        x = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(x)
        damaged_prob = get_damaged_prob(output)

        # testing purpose : save the input image with prob printed
        if self.output_test_image:
            key = str(obj_id)
            if not (key in self.test_counter):
                self.test_counter[key] = 0
            self.test_counter[key] += 1
            counter = self.test_counter[key]
            # cv2.putText(cropped_img, f"{damaged_prob:.2f} ", ((right-left)//2, (bottom-top)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            cv2.imwrite(f'/content/test/obj{obj_id:04}_{counter:02}.jpg', cropped_img)

        # store all probs
        if self.save_probs:
            if not obj_id in self.id2probs:
                frame_no2prob = {}
            else:
                frame_no2prob = self.id2probs[obj_id]
            frame_no2prob[frame_no] = damaged_prob
            self.id2probs[obj_id] = frame_no2prob

        return damaged_prob


    def get_checkpoint_path(self):
        return self.checkpoint_path

    def get_avg_prob(self, obj_id, cur_frame_no):
        assert self.save_probs, "Need to save the probs in order to compute the average prob, pass save_probs=True when constructing the detector"
        total, count = 0.0, 0.0
        if obj_id in self.id2probs:
            frame_no2prob = self.id2probs[obj_id]
            remove = []
            for frame_no in frame_no2prob:
                if frame_no < cur_frame_no - self.prob_period:
                    remove.append(frame_no)
                elif frame_no <= cur_frame_no + self.prob_period:
                    damaged_prob = frame_no2prob[frame_no]
                    if self.weighted_prob :
                        if damaged_prob >= self.conf_thres:
                            weight = 1+ (damaged_prob-self.conf_thres)/(1-self.conf_thres)*0.5
                        else:
                            weight = 1- ((self.conf_thres-damaged_prob)**2) / ((1-self.conf_thres)**2) *0.5
                        # weight = max(0.6, damaged_prob+0.2)
                    else:
                        weight = 1

                    count += weight
                    total += damaged_prob*weight

            for frame_no in remove: del frame_no2prob[frame_no]
            if count==0:
                return 0
            return total/count
        return -1

    def get_adjusted_prob(self, obj_id, cur_frame_no):
        assert self.save_probs, "Need to save the probs in order to compute the adjusted prob, pass save_probs=True when constructing the detector"
        
        if obj_id in self.id2probs:
            frame_no2prob = self.id2probs[obj_id]
            remove = []
            total, count = 0.0, 0
            cur_prob, conf_count = -1, 0
            future_count, future_conf_count = 0, 0
            past_count, past_conf_count = 0, 0
            total_weight = 0
            for frame_no in frame_no2prob:
                # expired frame
                if frame_no < cur_frame_no - self.prob_period:
                    remove.append(frame_no)
                # within range
                elif frame_no <= cur_frame_no + self.prob_period:
                    damaged_prob= frame_no2prob[frame_no]

                    if self.weighted_prob :
                        if damaged_prob >= self.conf_thres:
                            weight = 1 + (damaged_prob-self.conf_thres)/(1-self.conf_thres) *0.5
                        else:
                            weight = 1 + ((self.conf_thres-damaged_prob)**2)/(self.conf_thres**2) *0.5
                        # weight = max(0.6, damaged_prob+0.2)
                    else:
                        weight = 1

                    count += 1
                    total_weight += weight
                    total += damaged_prob
                    # future frames
                    if cur_frame_no<frame_no:
                        future_count += 1
                        if damaged_prob>self.conf_thres:
                            future_conf_count += 1
                            conf_count += 1
                    #current
                    elif cur_frame_no==frame_no: 
                        cur_prob = damaged_prob
                        if damaged_prob>self.conf_thres:
                            conf_count += 1
                    #past
                    else: 
                        past_count += 1
                        if damaged_prob>self.conf_thres:
                            past_conf_count += 1
                            conf_count += 1
                        
            for frame_no in remove: del frame_no2prob[frame_no]

            if cur_prob>=self.conf_thres:
                if cur_prob>=self.very_conf_thres:
                    return cur_prob
                elif past_conf_count/max(1,past_count)>0.3 or \
                     future_conf_count/max(1,future_count)>0.3:
                    return cur_prob
                else:
                    return total/total_weight #total_weigh=count if not using weighted prob
            else:
                return cur_prob

        return -1

    def set_conf_thres(self, new_thes):
        self.conf_thres = new_thes



def get_damaged_prob(output):
    # is damaged car prob
    prob = torch.softmax(output, dim=1)[0, 1].item()
    return prob

def crop_and_pad(frame, bbox, padding_size):
    left, top, right, bottom = bbox
    x_pad, y_pad = padding_size
    h, w, _ = frame.shape
    left2, top2, right2, bottom2 = max(left-x_pad,0), max(top-y_pad,0),\
                                  min(right+x_pad, w), min(bottom+y_pad, h)

    return frame[top2:bottom2, left2:right2]


# this is for after padding
def erase_overlapped(cropped_img, target_bbox, frame_info, padding_size):
    left, top, right, bottom = target_bbox
    x_pad, y_pad = padding_size
    for obj_id in frame_info:
        _, _, bbox = frame_info[obj_id]
        left2, top2, right2, bottom2 = bbox

        from_right = left2>left and left2<right
        from_left = right2>left and right2<right
        from_bot = top2>top and top2<bottom
        from_top = bottom2>top and bottom2<bottom


        if (from_left or from_right ) and (from_top or from_bot):
            if from_left and from_right: #within
                erase_left =  left2 - left + x_pad
                erase_right = right2 - left + x_pad
            elif from_left:
                erase_left = 0
                erase_right = right2 - left + x_pad
            else:
                erase_left = left2 - left + x_pad
                erase_right = right - left + x_pad*2

            if from_top and from_bot:
                erase_top = top2 - top + y_pad
                erase_bot = bottom2 - top + y_pad
            elif from_top:
                erase_top = 0
                erase_bot = bottom2 - top + y_pad
            else:
                erase_top = top2-top + y_pad
                erase_bot = bottom - top + 2*y_pad

            cv2.rectangle(cropped_img, (erase_left, erase_top), (erase_right, erase_bot), (0,0,0), -1)





