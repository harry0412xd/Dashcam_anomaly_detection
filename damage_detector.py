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
    def __init__(self, device):
        # url = "https://github.com/harry0412xd/Dashcam_anomaly_detection/releases/download/v1.0/gluon_seresnext101_32x4d-244_checkpoint-69.pth.tar"
        # checkpoint_path = "model_data/gluon_seresnext101_32x4d-244_checkpoint-69.pth.tar"
        # if not os.path.isfile(checkpoint_path):
        #     torch.utils.model_zoo.load_url(url, model_dir="model_data/")

        # new data(5) 2020/3/17
        checkpoint_path = "/content/MyDrive/cls_model/20200317-083104-gluon_seresnext101_32x4d-224/model_best.pth.tar"
        model = create_model('gluon_seresnext101_32x4d', num_classes=2, checkpoint_path = checkpoint_path)

        # new data(6)
        # checkpoint_path = "/content/MyDrive/cls_model/train/20200321-165626-gluon_seresnext101_32x4d-224/averaged.pth"
        # model = create_model('gluon_seresnext101_32x4d', num_classes=2, checkpoint_path = checkpoint_path)
       
        # augmix test
        # checkpoint_path = "/content/MyDrive/cls_model/train/20200318-112718-gluon_seresnext101_32x4d-224/model_best.pth.tar"
        # model = create_model('gluon_seresnext101_32x4d', num_classes=2, checkpoint_path = checkpoint_path)

        # model = create_model('gluon_seresnext101_32x4d', num_classes=2)
        # convert_splitbn_model(model,3)
        # distribute_bn(model, 1, True)
        # resume_checkpoint(model, checkpoint_path)

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


    def detect(self, frame, bbox, padding_size= (0,0), frame_info=None, erase_overlap=False, obj_id=None):
        cropped_img = crop_and_pad(frame, bbox, padding_size)
        if erase_overlap:
            assert frame_info is not None, "Other bounding boxes are necessary to find overlapped region. Pass the frame_info param."
            erase_overlapped(cropped_img, bbox, frame_info, padding_size)
        img_RGB = cv2.cvtColor(cropped_img,cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img_RGB)
        img = self.transform(img)
        # img.save("test.jpg")
        
        x = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(x)
        damaged_prop = get_damaged_prop(output)

        # testing purpose : save the input image with prop printed
        if obj_id:
            key = str(obj_id)
            if not (key in self.test_counter):
                self.test_counter[key] = 0
            self.test_counter[key] += 1
            counter = self.test_counter[key]
            # cv2.putText(cropped_img, f"{damaged_prop:.2f} ", ((right-left)//2, (bottom-top)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            # cv2.putText(cropped_img, f"{get_whole_prop(output):.2f} ", ((right-left)//2, (bottom-top)//2 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.imwrite(f'/content/test/obj{obj_id:04}_{counter:02}.jpg', cropped_img)

        return damaged_prop

    def get_checkpoint_path(self):
        return self.checkpoint_path

def get_damaged_prop(output):
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
            



