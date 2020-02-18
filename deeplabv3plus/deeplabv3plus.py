import torch
import cv2
import numpy as np
import os


from PIL import Image
from torchvision import transforms

import deeplabv3plus.network as network

from deeplabv3plus.cityscapes import Cityscapes_decoder
from deeplabv3plus.voc import Voc_decoder

class DeepLabv3plus():
    def __init__(self, device, video_writer=None):
          checkpoint_path = '/content/MyDrive/Code_to_test/best_deeplabv3plus_mobilenet_cityscapes_os16.pth'
          model = network.deeplabv3plus_mobilenet(num_classes=19, output_stride = 16)
          self.decoder = Cityscapes_decoder()

          # checkpoint_path = '/content/MyDrive/Code_to_test/best_deeplabv3plus_resnet101_voc_os16.pth'
          # model = network.deeplabv3plus_resnet101(num_classes=21, output_stride=16, pretrained_backbone=True)
          # self.decoder = Voc_decoder()          


          checkpoint = torch.load(checkpoint_path)
          model.load_state_dict(checkpoint["model_state"])

          model.to(device)
          model.eval()

          self.device = device
          self.model = model
          self.transform = transforms.Compose([
              transforms.Resize((512, 1024)),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
          ])
          self.counter = 0

          # Output result video if needed
          self.writer = video_writer

          

    

    def predict(self, frame, test_writer=None):
        height, width, _ = frame.shape
        resize_height = width//2
        crop_frame = frame[0:resize_height, 0:width]

        frame_RGB = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_RGB)
        img = self.transform(img)
        # img = img.to(self.device, dtype=torch.float32)

        x = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs  = self.model(x)

        preds = outputs.detach().max(dim=1)[1].cpu().numpy()
        pred = preds[0]
        pred = self.decoder.decode_target(pred).astype(np.uint8)

        # if not os.path.exists('results'):
        #     os.mkdir('results')
        # self.counter += 1
        # Image.fromarray(pred).save('results/%d_pred.png' % self.counter)
        if self.writer is not None:
            out_img = Image.fromarray(pred)
            out_img = cv2.cvtColor(np.asarray(out_img),cv2.COLOR_RGB2BGR)
            self.writer.write(out_img)
            


