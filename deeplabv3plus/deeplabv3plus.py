import torch
import cv2
import numpy as np

from PIL import Image
from torchvision import transforms

import deeplabv3plus.network as network

class DeepLabv3plus():
    def __init__(self, device):
          checkpoint_path = './deeplabv3plus/deeplabv3plus.py'

          model = network.deeplabv3plus_mobilenet(num_classes=19, output_stride = 16)
          checkpoint = torch.load(checkpoint_path)
          model.load_state_dict(checkpoint["model_state"])

          model.to(device)
          model.eval()


          self.device = device
          self.model = model
          self.transform = transforms.Compose([
              # transforms.Resize((224, 224)),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
          ])

          self.test_counter = {}