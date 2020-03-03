import torch
import cv2
import math
import numpy as np
import sys
import os.path

from PIL import Image
from torchvision import transforms

import timm

class Damage_detector():
    def __init__(self, device):
        url = "https://github.com/harry0412xd/YOLOv3_anomaly_detection/releases/download/v1.0/gluon_seresnext101_32x4d-244_checkpoint-69.pth.tar"
        checkpoint_path = "model_data/gluon_seresnext101_32x4d-244_checkpoint-69.pth.tar"
        if not os.path.isfile(path):
            torch.utils.model_zoo.load_url(url, model_dir=path)
        # checkpoint_path = '/content/MyDrive/cls_model/train/20200209-152119-gluon_seresnext101_32x4d-224/checkpoint-69.pth.tar'

        model = timm.create_model('gluon_seresnext101_32x4d', num_classes=2, checkpoint_path = checkpoint_path)
        model.to(device)
        model.eval()
        self.device = device
        self.model = model
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.test_counter = {}

    def detect(self, frame, bbox, obj_id=None):
        
        left, top, right, bottom = bbox
        cropped_img = frame[top:bottom, left:right]
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
            cv2.putText(cropped_img, f"{damaged_prop:.2f} ", ((right-left)//2, (bottom-top)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            cv2.putText(cropped_img, f"{get_whole_prop(output):.2f} ", ((right-left)//2, (bottom-top)//2 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.imwrite(f'/content/test/obj{obj_id:04}_{counter:02}.jpg', cropped_img)

        return damaged_prop


def get_damaged_prop(output):
    # is damaged car prob
    prob = torch.softmax(output, dim=1)[0, 1].item()
    return prob 



