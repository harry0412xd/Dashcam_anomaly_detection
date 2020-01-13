import torch
from lane_SCNN.model import SCNN
from lane_SCNN.utils.prob2lines import getLane
from lane_SCNN.utils.transforms import *

class Lane_model():
    def __init__(self, device):
        # exp0 : 512, 288
        # exp10 : 800, 288
        input_size = (800, 288)
        self.device = device

        self.model = SCNN(input_size=input_size, pretrained=False).to(device)
        save_dict = torch.load("lane_SCNN/exp10_best.pth", map_location=device)
        self.model.load_state_dict(save_dict['net'])
        self.model.eval()

        mean=(0.3598, 0.3653, 0.3662) # CULane mean, std
        std=(0.2573, 0.2663, 0.2756)
        # Imagenet mean, std
        # mean = (0.485, 0.456, 0.406)
        # std = (0.229, 0.224, 0.225)
        global transform_img, transform_to_net
        self.transform_img = Resize(input_size)
        self.transform_to_net = Compose(ToTensor(), Normalize(mean=mean, std=std))

    def detect(frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = transform_img({'img': img})['img']
        x = transform_to_net({'img': img})['img']
        x.unsqueeze_(0)
        if torch.cuda.is_available():
            x = x.cuda()
        seg_pred, exist_pred = model(x)[:2]
        seg_pred = seg_pred.detach().cpu().numpy()
        exist_pred = exist_pred.detach().cpu().numpy()
        seg_pred = seg_pred[0]
        exist = [1 if exist_pred[0, i] > 0.5 else 0 for i in range(4)]

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        lane_img = np.zeros_like(img)
        color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='uint8')
        coord_mask = np.argmax(seg_pred, axis=0)
        for i in range(0, 4):
            if exist_pred[0, i] > 0.5:
                lane_img[coord_mask == (i + 1)] = color[i]
        img = cv2.addWeighted(src1=lane_img, alpha=0.8, src2=img, beta=1., gamma=0.)
        # img = cv2.resize(img, (vid_width, vid_height))
        return img