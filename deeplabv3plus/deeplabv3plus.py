import torch
import cv2
import numpy as np
import os
from collections import namedtuple

from PIL import Image
from torchvision import transforms

import deeplabv3plus.network as network
from deeplabv3plus.cityscapes import Cityscapes

class DeepLabv3plus():
    def __init__(self, device):
          checkpoint_path = '/content/MyDrive/Code_to_test/best_deeplabv3plus_mobilenet_cityscapes_os16.pth'

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
          self.counter = 0

    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]

    def predict(self, frame, test_writer=None):

        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_RGB)
        img = self.transform(img)
        # img = img.to(self.device, dtype=torch.float32)

        x = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs  = self.model(x)

        preds = outputs.detach().max(dim=1)[1].cpu().numpy()
        pred = preds[0]
        pred = self.decode_target(pred).astype(np.uint8)

        # if not os.path.exists('results'):
        #     os.mkdir('results')
        # self.counter += 1
        # Image.fromarray(pred).save('results/%d_pred.png' % self.counter)
        if test_writer is not None:
            out_img = Image.fromarray(pred)
            out_img = cv2.cvtColor(np.asarray(out_img),cv2.COLOR_RGB2BGR)
            test_writer.write(out_img)
            


    def decode_target(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        result = np.array(cls.train_id_to_color)[target]
        return result