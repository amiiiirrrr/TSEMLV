import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys
sys.path.append('models/DepthModels/UniK3D/')
sys.path.append('models/DetectionModels/yolov7/')
import torch
import torch.nn.functional as F
import numpy as np
import copy
import os
from src.config import Config
from src.detect_frame import DetectionYolo
import cv2
import matplotlib.pyplot as plt

from src.unik_depth import UnikDepther

__author__ = "Seyed Amir Mousavi"
__credits__ = ["Amir Mousavi"]
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Seyed Amir Mousavi"
__email__ = "seyedamir.mousavi@ghent.ac.kr"
__status__ = "Research"


class Run_Unik3d:
    """
    Run class
    get all AI modules together and create a sensible procedure
    """

    def __init__(self, args):
        """
        Initialize AI modules
        """
        self.args = args
        self.detector = DetectionYolo(self.args)
        self.dataset = self.detector.dataset
        self.unikDeptherObj = UnikDepther(self.args)

    def start_inferencing(self):
        
        if self.args.output_path is not None:
            os.makedirs(self.args.output_path, exist_ok=True)
        for path, img, im0s, vid_cap in self.dataset:
            # print('path', path)
            # if 'AGRMNPRZ_P4_27162' not in path:
            #     continue
            dict_result = {}
            height, width, _ = im0s.shape
            if (height != 572) and (width != 720):
                im0s = cv2.resize(im0s, (720, 572))
            img_depth = im0s.copy()
            self.img_original = im0s.copy()

            path_image = path.split('/')[-1]
            name_image = path_image.split('.')[0]
            self.path_save = os.path.join(self.args.output_path, name_image)
            # print("path_save", self.path_save)
            if self.path_save is not None:
                os.makedirs(self.path_save, exist_ok=True) 

            prediction_depth = self.unikDeptherObj.unik_depther(path)
            self.unikDeptherObj.vis_unik_depther(prediction_depth, self.path_save)

if __name__ == '__main__':
    conf_obj = Config()
    args = conf_obj.get_args()
    obj_run = Run_Unik3d(args)
    obj_run.start_inferencing()

