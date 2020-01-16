import torch
import cv2
import math
import numpy as np

from PIL import Image
from torchvision import transforms
from Fast_SCNN.models.fast_scnn import get_fast_scnn
from Fast_SCNN.utils.visualize import get_color_pallete

class Seg_model():
    def __init__(self, device):
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.model = get_fast_scnn(dataset = 'citys', pretrained=True, root="./Fast_SCNN/weights", map_cpu=False).to(device)
        self.model.eval()

    def detect(self, frame):
        frame_RGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        img = Image.fromarray(frame_RGB).resize((w//32*32, h//32*32))
        img = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img)
        pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()
        mask = get_color_pallete(pred, 'citys').convert('RGB')

        mask_img = cv2.cvtColor(np.asarray(mask),cv2.COLOR_RGB2BGR)
        mask_img = cv2.resize(mask_img, (w, h))
        # output = cv2.addWeighted(frame, 0.5, mask_img, 0.5, 0)

        return mask_img

