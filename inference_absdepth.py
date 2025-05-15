import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys
sys.path.append('models/DepthModels/MiDaS/')
sys.path.append('models/SegmentationModels/mmsegmentation_mask2former/')
sys.path.append('models/DetectionModels/yolov7/')
import torch
import torch.nn.functional as F
import numpy as np
import copy
import os
from src.config import Config
from run import DepthEstimator
from src.detect_frame import DetectionYolo
import cv2
from src.segment_frames_mask2former import SegmentMMSegmentation
import matplotlib.pyplot as plt

from src.utils.camera_callibration import CameraCallibration
from src.utils.depth_utils import depthUtils
from src.utils.common import calculate_average_image_oneChannel_v2, calculate_average_image_oneChannel_v1
from src.utils.seg_utils import SegmentationUtils
from src.utils.visual_utils import create_segmentation_SI_Tumor_v1, visualize_function, dimmed_mask
from src.evaluation import calculate_mae
# from src.unik_depth import UnikDepther
from src.measure_tumor_v1.measure_tumor_v1 import measureTumor_V1
from src.measure_tumor_v2.find_scale import ScaleDepth
from src.measure_tumor_v2.measure_tumor_v2 import MeasureTumor
from src.measure_tumor_v2.instrument_pose import estimate_instrument_pose, fit_shaft_lines

__author__ = "Seyed Amir Mousavi"
__credits__ = ["Amir Mousavi"]
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Seyed Amir Mousavi"
__email__ = "seyedamir.mousavi@ghent.ac.kr"
__status__ = "Research"

class Run:
    """
    Run class
    get all AI modules together and create a sensible procedure
    """

    def __init__(self, args):
        """
        Initialize AI modules
        """
        super(Run, self).__init__()

        self.args = args
        self.detector = DetectionYolo(self.args)
        self.segmentor = SegmentMMSegmentation(self.args)
        self.depther = DepthEstimator(self.args)
        self.dataset = self.detector.dataset
        self.camcal = CameraCallibration(self.args)
        self.segUtilObj = SegmentationUtils(self.args, self.segmentor)
        self.measureTumor_obj_v1 = measureTumor_V1(self.args)
        self.depthUtilsObj = depthUtils(self.args)
        self.depthScaleObj = ScaleDepth(self.args)
        self.measureTumorObj = MeasureTumor(self.args)
        # self.unikDeptherObj = UnikDepther(self.args)
        self.list_results = []
        
    def start_inferencing(self):
        
        K_matrix = self.camcal.K_matrix

        if self.args.output_path is not None:
            os.makedirs(self.args.output_path, exist_ok=True)
        for path, img, im0s, vid_cap in self.dataset:
            print('path', path)
            if 'ANLLN_P1_16631' not in path:
                continue
            dict_result = {}
            height, width, _ = im0s.shape
            # print('height', height)
            # print('width', width)
            if (height != 576) and (width != 720):
                im0s = cv2.resize(im0s, (720, 572))
            img_segmentation = im0s.copy()
            img_depth = im0s.copy()
            self.img_original = im0s.copy()

            with torch.no_grad():
                object_boxes = self.detector.detect(path, img, im0s, vid_cap)
            # print("object_boxes", object_boxes) 
            if len(object_boxes) == 0:
                continue

            largest_tumor, box_diameter = self.segUtilObj.find_biggestTumor_v1(object_boxes)

            if len(largest_tumor) == 0:
                continue

            path_image = path.split('/')[-1]
            name_image = path_image.split('.')[0]
            dict_result["image_name"] = name_image 
            self.path_save = os.path.join(self.args.output_path, name_image)
            # print("path_save", self.path_save)
            if self.path_save is not None:
                os.makedirs(self.path_save, exist_ok=True) 

            mask_visualize_tumor, mask_cal_tumor, center_tumor_box = self.segUtilObj.create_detection_mask_v1(largest_tumor, height, width)
            
            visual, segment_map = self.segmentor.segment(img_segmentation)
            # print('visual', visual.shape)
            # print('segment_map', segment_map.shape)
            si_mask, exist_SI = self.segUtilObj.create_segmentation_mask_v2(segment_map[0])
            # print("si_mask.shape", si_mask.shape)

            if exist_SI and (len(largest_tumor) > 0):
                # mask_SI_tumor, new_tumor_box = self.segUtilObj.create_mask_SI_tumor_v1(segment_map[0], largest_tumor)
                cv2.imwrite(os.path.join(self.path_save, 'img_original.png'), self.img_original)
                # cv2.imwrite(os.path.join(self.path_save, 'segment_map.png'), visual)

                #For MiDaS
                # prediction_depth, depth_map_visualize, raw_depth255, idepth = self.depther.run(img_depth, os.path.join(self.path_save, name_image))

                # For Unik3d
                prediction_depth = np.load(f"{self.path_save}/predicted_depth_unik3d_normalized.npy")

                # undistorted_si_mask = self.camcal.undistort_image(self.img_original)
                # cv2.imwrite(os.path.join(self.path_save, 'undistorted_si_mask.png'), undistorted_si_mask)
                # undistorted_prediction_depth = self.camcal.undistort_image(prediction_depth)
                # cv2.imwrite(os.path.join(self.path_save, 'undistorted_prediction_depth.png'), undistorted_prediction_depth)
                # undistorted_segment_map = self.camcal.undistort_image(segment_map)
                # cv2.imwrite(os.path.join(self.path_save, 'undistorted_segment_map.png'), undistorted_segment_map)

                line_1_coeffs, line_2_coeffs, viz_copy = fit_shaft_lines(si_mask, self.img_original, self.path_save)
                # cv2.imwrite(os.path.join(self.path_save, 'viz_copy.png'), viz_copy)

                if line_1_coeffs is not None and line_2_coeffs is not None:

                    P_c, v_L, viz_copy = estimate_instrument_pose(K_matrix, self.args.instrument_radius_mm, line_1_coeffs, line_2_coeffs, viz_copy)
                    # cv2.imwrite(os.path.join(self.path_save, 'viz_copy.png'), viz_copy)

                    _, _, absolute_depth_map = self.depthScaleObj.estimate_scale_and_depth_map(K_matrix, P_c, v_L, si_mask, prediction_depth, self.path_save)

                    estimated_tumor_size_results = self.measureTumorObj.measure_box_dimensions(K_matrix, absolute_depth_map, largest_tumor)
                    # for key, value in estimated_tumor_size_results.items():
                    #     print(f"    {key}: {value:.2f}")

                    dict_result["horizontal length"] = estimated_tumor_size_results['horizontal'] 
                    dict_result["vertical length"] = estimated_tumor_size_results['vertical']  
                    dict_result["diagonal"] = estimated_tumor_size_results['diagonal']   
                    self.list_results.append(dict_result)


        MAE = calculate_mae(self.args.ground_truth_path, self.list_results)
        print("MAE", MAE)


if __name__ == '__main__':
    conf_obj = Config()
    args = conf_obj.get_args()
    obj_run = Run(args)
    obj_run.start_inferencing()
