"""
Wrapper function created to use the deeplabv3+ code
"""
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
    def __init__(self, device, video_writer=None, overlay=False):
        url = "https://github.com/harry0412xd/Dashcam_anomaly_detection/releases/download/v1.0/best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
        checkpoint_path = "model_data/best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
        if not os.path.isfile(checkpoint_path):
            torch.utils.model_zoo.load_url(url, model_dir="model_data/")
        # checkpoint_path = '/content/MyDrive/pretrain_weights/deeplabv3+/best_deeplabv3plus_mobilenet_cityscapes_os16.pth'      
        model = network.deeplabv3plus_mobilenet(num_classes=19, output_stride = 16, pretrained_backbone=False)
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
            # transforms.Resize((512, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        # self.counter = 0

        self.last_result = None

        # Output result video if needed
        self.writer = video_writer
        self.overlay = overlay

          
    def predict(self, frame):
        # resize and crop out the bottom
        height, width, _ = frame.shape
        resize_height = width//2
        pad_h =  height-resize_height
        crop_frame = frame[pad_h:height, 0:width]

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

        out_img = Image.fromarray(pred)
        out_img = cv2.cvtColor(np.asarray(out_img),cv2.COLOR_RGB2BGR)
        # print(out_img.shape)

        # resize back to original size with padding
        out_img = cv2.resize(out_img, (width, width//2))
        # padding_height = height - width//2
        out_img = cv2.copyMakeBorder(out_img, pad_h, 0, 0, 0, cv2.BORDER_CONSTANT, (255,255,255))

        if self.writer is not None:
            if self.overlay:
                overlay_img = cv2.addWeighted(frame, 0.3, out_img, 0.7, 0)
                self.writer.write(overlay_img)
            else:
                self.writer.write(frame)

        self.last_result = out_img #save last result
        return out_img
            

    
    def get_last_result(self, frame):
        if self.writer is not None:
            if self.overlay:
                overlay_img = cv2.addWeighted(frame, 0.3, self.last_result, 0.7, 0)
                self.writer.write(overlay_img)
            else:
                self.writer.write(frame)

        return self.last_result


    def create_overlay(self, frame):
        if frame is None: return None
        overlay_img = cv2.addWeighted(frame, 0.3, self.last_result, 0.7, 0)
        return overlay_img


