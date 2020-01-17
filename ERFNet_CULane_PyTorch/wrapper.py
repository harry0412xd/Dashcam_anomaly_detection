import torch
import cv2
import math
import numpy as np
import sys

from PIL import Image
from torchvision import transforms

from ERFNet_CULane_PyTorch.models.erfnet import *
# from ERFNet_CULane_PyTorch.utils import transforms as tf
from ERFNet_CULane_PyTorch.prob_to_lines import *

class Erf_model():
    def __init__(self, device):
        self.device = device
        # args.dataset == 'CULane':
        num_class = 5
        ignore_label = 255

        self.model = ERFNet(num_class).to(self.device)
        input_mean = self.model.input_mean
        input_std = self.model.input_std

        self.model = torch.nn.DataParallel(self.model, device_ids=[0]).cuda()
        checkpoint = torch.load("ERFNet_CULane_PyTorch/trained/ERFNet_trained.tar")
        torch.nn.Module.load_state_dict(self.model, checkpoint['state_dict'])
        
        self.transform = transforms.Compose([
            transforms.Resize((976, 208)),
            transforms.ToTensor(),
            # transforms.Normalize(input_mean, input_std)
            # tf.GroupRandomScaleNew(size=(976, 208), interpolation=()),
            # tf.GroupRandomScaleNew(size=(976, 208), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
            # tf.GroupNormalize(mean=(input_mean, (0, )), std=(input_std, (1, ))),
        ])

        self.model.eval()

    def detect(self, frame, frame_no):
        frame_RGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        img = Image.fromarray(frame_RGB)
        img = self.transform(img).unsqueeze(0).to(self.device)
        
        output, output_exist = self.model(img)

        output = F.softmax(output, dim=1)
        
        pred = output.data.cpu().numpy()[0] # BxCxHxW
        pred_exist = output_exist.data.cpu().numpy()[0] # BxO
        np.set_printoptions(threshold=sys.maxsize)
        f = open(f"/content/test/{frame_no}.txt", "w")
        f.write(np.array2string(pred))
        f.close()
        prob_maps = []
        for num in range(4):
            # prob_map = (pred[num+1]*255).astype(int)
            # prob_maps.append(prob_map)
            # print(prob_map)
            if pred_exist[num] >0.3:
                pred_exist[num] = 1
            else:
                pred_exist[num] = 0

            
        print(GetLines(pred_exist, pred))
            # save_img = cv2.blur(prob_map,(9,9))
            # out_name = f"/content/test/{frame_no}_lane{num+1}.png"
            # cv2.imwrite(out_name, save_img)




