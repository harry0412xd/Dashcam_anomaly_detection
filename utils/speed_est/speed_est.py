# Modify from https://github.com/laa-1-yay/speed-estimation-of-car-with-optical-flow

# from model import CNNModel
from utils.speed_est.model import CNNModel
import cv2
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

class Speed_Est(object):

    def __init__(self, first_frame):
        self.model = CNNModel()
        self.model.load_weights("./model_data/bestCNNModel_flow.h5")
        
        self.prev_frame =  np.zeros_like(first_frame)
        self.flow_image_bgr_prev1 =  np.zeros_like(first_frame)
        self.flow_image_bgr_prev2 =  np.zeros_like(first_frame)
        self.flow_image_bgr_prev3 =  np.zeros_like(first_frame)
        self.flow_image_bgr_prev4 =  np.zeros_like(first_frame)

    def predict(self, next_frame):
        flow_image_bgr_next = convertToOptical(self.prev_frame, next_frame)
        flow_image_bgr = (self.flow_image_bgr_prev1 + self.flow_image_bgr_prev2
                          + self.flow_image_bgr_prev3 + self.flow_image_bgr_prev4 
                          + flow_image_bgr_next)/4

        curr_image = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)

        combined_image_save = 0.1*curr_image + flow_image_bgr

        #CHOOSE IF WE WANT TO TEST WITH ONLY OPTICAL FLOW OR A COMBINATION OF VIDEO AND OPTICAL FLOW
        combined_image = flow_image_bgr
        # combined_image = combined_image_save

        combined_image_test = cv2.normalize(combined_image, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        #CHOOSE IF WE WANT TO TEST WITH ONLY OPTICAL FLOW OR A COMBINATION OF VIDEO AND OPTICAL FLOW
        # combined_image_test = cv2.resize(combined_image, (0,0), fx=0.5, fy=0.5)
        # combined_image_test = cv2.resize(combined_image_test, (0,0), fx=0.5, fy=0.5)

# expected conv2d_1_input to have shape (240, 320, 3)
        combined_image_test = cv2.resize(combined_image_test, (320,240))


        combined_image_test = combined_image_test.reshape(1, combined_image_test.shape[0], combined_image_test.shape[1], combined_image_test.shape[2])

        prediction = self.model.predict(combined_image_test)

        # post prediction process
        self.prev_frame = next_frame
        self.flow_image_bgr_prev4 = self.flow_image_bgr_prev3
        self.flow_image_bgr_prev3 = self.flow_image_bgr_prev2
        self.flow_image_bgr_prev2 = self.flow_image_bgr_prev1
        self.flow_image_bgr_prev1 = flow_image_bgr_next

        return prediction[0][0]

def convertToOptical(prev_image, curr_image):

    prev_image_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    curr_image_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_image_gray, curr_image_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # flow = cv2.calcOpticalFlowFarneback(prev_image_gray, curr_image_gray, None, 0.5, 2, 15, 2, 5, 1.2, 0)

    hsv = np.zeros_like(prev_image)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_image_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return flow_image_bgr

      