
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import os
from models.experimental import attempt_load
from utils_det.datasets import LoadImages
from utils_det.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils_det.plots import plot_one_box
from utils_det.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

class DetectionYolo:

    def __init__(self, args):
        """
        Initialize AI modules
        """
        super(DetectionYolo, self).__init__()

        self.args = args

        self.initialize()
        self.dataloader_yolo()


    def initialize(self):
        # Initialize
        set_logging()
        self.device = select_device(self.args.device_det)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.args.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.args.img_size, s=self.stride)  # check img_size

        if self.args.no_trace:
            self.model = TracedModel(self.model, self.device, self.img_size)

        if self.half:
            self.model.half()  # to FP16
        
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        
        self.save_img = not self.args.nosave and not self.args.source.endswith('.txt')  # save inference images

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def dataloader_yolo(self):
        self.dataset = LoadImages(self.args.source, img_size=self.imgsz, stride=self.stride)

    def detect(self, path, img, im0s, vid_cap):

        objects = []
        if not self.args.surgeons_evaluation:
            
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = self.model(img, augment=self.args.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.args.conf_thres, self.args.iou_thres, classes=self.args.classes, agnostic=self.args.agnostic_nms)

            # Process detections
            for i, det in enumerate(pred):  # detections per image

                p, s, im0, frame = path, '', im0s, getattr(self.dataset, 'frame', 0)

                # p = Path(p)  # to Path
                # save_path = str(save_dir / p.name)  # img.jpg
                # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # # Print results
                    # for c in det[:, -1].unique():
                    #     n = (det[:, -1] == c).sum()  # detections per class
                    #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        dict_peroObject = {}
                        # print("xyxy", xyxy)
                        # print("################################ cls #########################################", cls)
                        # scalar_value = cls.item()
                        # print("################################ scalar_value #########################################", scalar_value)
                        # if int(scalar_value)==23:
                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        
                            # line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            # with open(txt_path + '.txt', 'a') as f:
                            #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        # if self.save_img:  # Add bbox to image
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=2)
                        dict_peroObject["class"] = self.names[int(cls)]
                        dict_peroObject["xyxy"] = xyxy
                        dict_peroObject["conf"] = conf
                        objects.append(dict_peroObject)
                # Save results (image with detections)
                if self.save_img:
                    if self.dataset.mode == 'image':
                        # cv2.imwrite(save_path, im0)
                        cv2.imwrite(os.path.join('output_image/' + path.split('/')[-1].split('.')[0], 'detection.png'), im0)
                        # print(f" The image with the result is saved in: {save_path}")
        else:

            label_surgeon_file = path.split('.')[0] + '.txt'
            with open(label_surgeon_file, 'r') as file:
                # Read all lines into a list
                lines = file.readlines()

            # Display the content of the file
            for line in lines:
                # print("line", line)
                splitted = line.split(' ')
                x1, y1, w, h = int(splitted[0]), int(splitted[1]), int(splitted[2]), int(splitted[3])
                dict_peroObject = {}
                dict_peroObject["class"] = 'Tumor'
                dict_peroObject["xyxy"] = x1, y1, x1+w, y1+h
                dict_peroObject["conf"] = 1
                objects.append(dict_peroObject)
        
        return objects
