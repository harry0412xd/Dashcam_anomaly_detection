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

        self.model = EfficientNet.from_name('efficientnet-b4').to(device)
        checkpoint=torch.load('/content/MyDrive/cls_model/21Jan - new dataset/model_best.pth.tar')
        self.model.load_state_dict(checkpoint['state_dict'])

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.model.eval()

    def detect(self, frame, bbox):
        # labels_map = ["damaged","whole"]
        left, top, right, bottom = bbox
        left = max(0, left)
        top = max(0, top)
        cropped_img = frame[top:bottom, left:right]
        img_RGB = cv2.cvtColor(cropped_img,cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img_RGB)
        img = self.transform(img)
        # img.save("test.jpg")
        
        x = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(x)

        # print('-----')
        # for idx in torch.topk(output, k=2).indices.squeeze(0).tolist():
        #     prob = torch.softmax(output, dim=1)[0, idx].item()
        #     print('{label:<20} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))

        class_id = torch.topk(output, k=1).indices.squeeze(0)
        prob = torch.softmax(output, dim=1)[0, class_id].item()
        # print('{label:<20} ({p:.2f}%)'.format(label=labels_map[class_id], p=prob*100))
        if class_id==0 and prob>0.7:
            return True
        return False




