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
        checkpoint=torch.load('/content/MyDrive/model_best.pth.tar')
        self.model.load_state_dict(checkpoint['state_dict'])

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.model.eval()

    def detect(self, frame, bbox):
        left, top, right, bottom = bbox
        left = max(0, left)
        top = max(0, top)
        cropped_img = frame[top:bottom, left:right]
        img_RGB = cv2.cvtColor(cropped_img,cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img_RGB)
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(x)
        index = output.data.cpu().numpy().argmax()
        print(index)
        return index==14




