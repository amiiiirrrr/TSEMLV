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
from src.measure_tumor_v1.measure_tumor_v1 import measureTumor_V1

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
        self.list_results = []
        
    def start_inferencing(self):
        
        if self.args.output_path is not None:
            os.makedirs(self.args.output_path, exist_ok=True)
        for path, img, im0s, vid_cap in self.dataset:
            dict_result = {}
            height, width, _ = im0s.shape
            img_segmentation = im0s.copy()
            img_original = im0s.copy()
            image_visualize2 = im0s.copy()
            img_depth = im0s.copy()
            self.img_original = im0s.copy()

            with torch.no_grad():
                object_boxes = self.detector.detect(path, img, im0s, vid_cap)
            # print("object_boxes", object_boxes) 
            if len(object_boxes) > 0:
                largest_tumor, box_diameter = self.segUtilObj.find_biggestTumor_v1(object_boxes)
                # print("largest_tumor", largest_tumor) 
                if len(largest_tumor) > 0:

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
                    biggestSI_segment_map, exist_SI, cnt = self.segUtilObj.create_segmentation_mask_v1(segment_map[0])
                    # print("biggestSI_segment_map.shape", biggestSI_segment_map.shape)
                    # print("exist_SI", exist_SI) 
                    
                    if exist_SI and (len(largest_tumor) > 0):
                        mask_SI_tumor, new_tumor_box = self.segUtilObj.create_mask_SI_tumor_v1(segment_map[0], largest_tumor)
                        segmentation_SI_tumor = create_segmentation_SI_Tumor_v1(img_original, visual, mask_SI_tumor)
                        # cv2.imwrite(os.path.join(self.path_save, 'img_cropped.png'), img_cropped)
                        # cv2.imwrite(os.path.join(self.path_save, 'img_depthhere2.png'), img_depth)
                        prediction_depth, depth_map_visualize, raw_depth255, idepth = self.depther.run(img_depth, os.path.join(self.path_save, name_image))
                        # plt.imsave(os.path.join(self.path_save, 'prediction_depth.png'), prediction_depth)
                        # print("prediction_depth.shape", prediction_depth.shape)

                        normalized_depth = self.depthUtilsObj.normalize_depth_midas(prediction_depth)

                        self.diameter_SI_pixel, rect_SI, img_visualize_SI = self.measureTumor_obj_v1.minAreaRect_SI(cnt, biggestSI_segment_map, height, width)
                        Ps1, Ps2 = self.measureTumor_obj_v1.minAreaRect_SI2(biggestSI_segment_map)
                        SI_box = (*Ps1, *Ps2)
                        avg_depth_SI, avg_depth_tumor = self.depthUtilsObj.depth_tumor_SI_v1(self.path_save, mask_cal_tumor, rect_SI, normalized_depth, depth_map_visualize)

                        # diameter_Tumor_box = self.find_distances_diameters(self.diameter_SI_pixel, avg_depth_tumor, avg_depth_SI, box_diameter)
                        real_W_obj, real_H_obj, Z_ref, Z_obj = self.measureTumor_obj_v1.find_distances_diameters_v3(largest_tumor, SI_box, normalized_depth)

                        # print("diameter_SI_pixel:", self.diameter_SI_pixel)
                        # print("distance_Tumor:", distance_Tumor)
                        visualize_function(self.path_save, image_visualize2, img_visualize_SI, biggestSI_segment_map, rect_SI, real_W_obj, real_H_obj, Z_ref, Z_obj, largest_tumor, mask_visualize_tumor)
                        diameter = np.sqrt(real_W_obj**2 + real_H_obj**2)
                        # print("diameter", diameter) 
                        dict_result["horizontal length"] = real_W_obj 
                        dict_result["vertical length"] = real_H_obj 
                        dict_result["diagonal"] = diameter 
                        self.list_results.append(dict_result)

                        # cv2.imwrite(os.path.join(self.path_save, 'mask_biggest_tumor.png'), mask_visualize_tumor)
                        dimmed_mask_tumor = dimmed_mask(mask_visualize_tumor, img_original)
                        # cv2.imwrite(os.path.join(self.path_save, 'dimmed_mask_tumor.png'), dimmed_mask_tumor)
                        cv2.imwrite(os.path.join(self.path_save, 'segmentation.png'), visual)
                        # cv2.imwrite(os.path.join(self.path_save, 'biggestSI_segment_map.png'), biggestSI_segment_map)
                        
                        cv2.imwrite(os.path.join(self.path_save, 'img_original.png'), img_original)
                        # cv2.imwrite(os.path.join(self.path_save, 'im0s.png'), im0s)
                        # cv2.imwrite(os.path.join(self.path_save, 'adjusted_brighness.png'), img_depth)
                        if new_tumor_box:
                            # cv2.imwrite(os.path.join(self.path_save, 'mask_SI_tumor.png'), mask_SI_tumor)
                            blend_image = self.segUtilObj.blend_images_v1(segmentation_SI_tumor, img_visualize_SI, new_tumor_box)
                            cv2.imwrite(os.path.join(self.path_save, 'blend_image.png'), blend_image)
        MAE = calculate_mae(self.args.ground_truth_path, self.list_results)
        print("MAE", MAE)


if __name__ == '__main__':
    conf_obj = Config()
    args = conf_obj.get_args()
    obj_run = Run(args)
    obj_run.start_inferencing()