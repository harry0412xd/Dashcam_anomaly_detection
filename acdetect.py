import torch
import cv2
import math
import numpy as np
import sys

from PIL import Image
from torchvision import transforms

from efficientnet_pytorch import EfficientNet

class Accident_detector():
    def __init__(self, device):
        self.device = device

        self.model = EfficientNet.from_name('efficientnet-b4', override_params={'num_classes': 3}).to(device)
        checkpoint=torch.load("/content/MyDrive/cls_model/26jan/model_best2.pth.tar")
        print(checkpoint['epoch'])
        self.model.load_state_dict(checkpoint['state_dict'])

        self.transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.model.eval()

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

        # print('-----')
        labels_map = ["whole", "damaged", "FAS"]
        # for idx in torch.topk(output, k=2).indices.squeeze(0).tolist():
        #     prob = torch.softmax(output, dim=1)[0, idx].item()
        #     print('{label:<20} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))



        # # is damaged car prob
        # prob = torch.softmax(output, dim=1)[0, 0].item()
        # return prob 

        class_id = torch.topk(output, k=1).indices.squeeze(0)
        prob = torch.softmax(output, dim=1)[0, class_id].item()
        print('{label:<20} ({p:.2f}%)'.format(label=labels_map[class_id], p=prob*100))
        return labels_map[class_id], prob
        # if class_id==0 and prob>0.6:
        #     return True,'{label} ({p:.2f}%)'.format(label=labels_map[class_id], p=prob*100)
        # return False, '{label} ({p:.2f}%)'.format(label=labels_map[class_id], p=prob*100)





