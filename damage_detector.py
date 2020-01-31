import torch
import cv2
import math
import numpy as np
import sys

from PIL import Image
from torchvision import transforms

import timm

class Damage_detector():
    def __init__(self, device):
        checkpoint_path = '/content/MyDrive/cls_model/train/20200131-094030-seresnext101_32x4d-224/model_best.pth.tar'
        model = timm.create_model('seresnext101_32x4d', num_classes=2, checkpoint_path = checkpoint_path)
        model.to(device)
        model.eval()
        self.device = device
        self.model = model
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.labels_map = ["whole", "damaged"]
        self.count = 0

    def detect(self, frame, bbox):
        
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

        
        self.count +=1
        cv2.putText(cropped_img, f"{damaged_prop:.2f} ", ((right-left)//2, (bottom-top)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imwrite(f'/content/test/{self.count:04}.jpg', cropped_img)


        return "damaged", damaged_prop

        # print('-----')
        # for idx in torch.topk(output, k=2).indices.squeeze(0).tolist():
        #     prob = torch.softmax(output, dim=1)[0, idx].item()
        #     print('{label:<20} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))



        # class_id = torch.topk(output, k=1).indices.squeeze(0)
        # prob = torch.softmax(output, dim=1)[0, class_id].item()
        # # print('{label:<20} ({p:.2f}%)'.format(label=labels_map[class_id], p=prob*100))
        # return labels_map[class_id], prob

def get_damaged_prop(output):
    # is damaged car prob
    prob = torch.softmax(output, dim=1)[0, 1].item()
    return prob 




