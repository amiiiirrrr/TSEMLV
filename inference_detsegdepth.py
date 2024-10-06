import sys
sys.path.append('MiDaS/')
sys.path.append('mmsegmentation_mask2former/')
sys.path.append('yolov7/')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from torch.utils.data import Sampler
import torch.nn.functional as F
import time
import pickle
import numpy as np
from torchvision.transforms import Lambda
import argparse
import copy
import random
import numbers
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
#from NLBlock import NLBlockimport os, subprocess
import os, subprocess
from config import Config
from run import *
from midas.model_loader import default_models, load_model
from detect_frame import DetectionYolo
import math
from segment_frames_mask2former import SegmentMMSegmentation
# from segment_frames_segformer import SegmentMMSegmentation
import matplotlib.pyplot as plt
import pandas as pd

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
        
        self.list_results = []
        
    def start_inferencing(self):
        
        if self.args.output_path is not None:
            os.makedirs(self.args.output_path, exist_ok=True)
        for path, img, im0s, vid_cap in self.dataset:
            dict_result = {}
            self.height, self.width, _ = im0s.shape
            img_segmentation = im0s.copy()
            img_original = im0s.copy()
            image_visualize2 = im0s.copy()
            img_depth = im0s.copy()
            self.img_original = im0s.copy()
            # img_depth = self.clahe(img_depth)
            # print("img_depth.shape", img_depth.shape)

            with torch.no_grad():
                object_boxes = self.detector.detect(path, img, im0s, vid_cap)
            # print("object_boxes", object_boxes) 
            if len(object_boxes) > 0:
                largest_tumor, box_diameter = self.find_biggestTumor(object_boxes)
                # print("largest_tumor", largest_tumor) 
                if len(largest_tumor) > 0:

                    path_image = path.split('/')[-1]
                    name_image = path_image.split('.')[0]
                    dict_result["image_name"] = name_image 
                    self.path_save = os.path.join(self.args.output_path, name_image)
                    # print("path_save", self.path_save)
                    if self.path_save is not None:
                        os.makedirs(self.path_save, exist_ok=True) 

                    mask_visualize_tumor, mask_cal_tumor, center_tumor_box = self.create_detection_mask(largest_tumor)
                    
                    visual, segment_map = self.segmentor.segment(img_segmentation)
                    # print('visual', visual.shape)
                    # print('segment_map', segment_map.shape)
                    biggestSI_segment_map, exist_SI = self.create_segmentation_mask(segment_map[0])
                    # print("biggestSI_segment_map.shape", biggestSI_segment_map.shape)
                    # print("exist_SI", exist_SI) 
                    
                    if exist_SI and (len(largest_tumor) > 0):
                        mask_SI_tumor, new_tumor_box = self.create_mask_SI_tumor(segment_map[0], largest_tumor)
                        segmentation_SI_tumor = self.create_segmentation_SI_Tumor(img_original, visual, mask_SI_tumor)                        # cv2.imwrite(os.path.join(self.path_save, 'img_cropped.png'), img_cropped)
                        # cv2.imwrite(os.path.join(self.path_save, 'img_depthhere2.png'), img_depth)
                        prediction_depth, depth_map_visualize, raw_depth255, idepth = self.depther.run(img_depth, os.path.join(self.path_save, name_image))
                        # plt.imsave(os.path.join(self.path_save, 'prediction_depth.png'), prediction_depth)
                        # print("prediction_depth.shape", prediction_depth.shape)

                        # https://github.com/isl-org/MiDaS/issues/4
                        normalized_depth = self.normalize_depth(prediction_depth)

                        self.diameter_SI_pixel, rect_SI, img_visualize_SI = self.minAreaRect_SI(biggestSI_segment_map)
                        Ps1, Ps2 = self.minAreaRect_SI2(biggestSI_segment_map)
                        SI_box = (*Ps1, *Ps2)
                        avg_depth_SI, avg_depth_tumor = self.depth_tumor_SI(mask_cal_tumor, rect_SI, normalized_depth, depth_map_visualize)

                        real_W_obj, real_H_obj, Z_ref, Z_obj = self.find_distances_diameters_v3(largest_tumor, SI_box, normalized_depth)

                        # print("real_W_obj:", real_W_obj)
                        # print("real_H_obj:", real_H_obj)
                        # print("Z_ref:", Z_ref)
                        # print("Z_obj:", Z_obj)
                        # print("diameter_SI_pixel:", self.diameter_SI_pixel)
                        # print("distance_Tumor:", distance_Tumor)
                        self.visualize_function(image_visualize2, img_visualize_SI, biggestSI_segment_map, rect_SI, real_W_obj, real_H_obj, Z_ref, Z_obj, largest_tumor, mask_visualize_tumor)
                        diameter = np.sqrt(real_W_obj**2 + real_H_obj**2)
                        # print("diameter", diameter) 
                        dict_result["horizontal length"] = real_W_obj 
                        dict_result["vertical length"] = real_H_obj 
                        dict_result["diagonal"] = diameter 
                        self.list_results.append(dict_result)

                        # cv2.imwrite(os.path.join(self.path_save, 'mask_biggest_tumor.png'), mask_visualize_tumor)
                        dimmed_mask_tumor = self.dimmed_mask(mask_visualize_tumor, img_original)
                        # cv2.imwrite(os.path.join(self.path_save, 'dimmed_mask_tumor.png'), dimmed_mask_tumor)
                        cv2.imwrite(os.path.join(self.path_save, 'segmentation.png'), visual)
                        # cv2.imwrite(os.path.join(self.path_save, 'biggestSI_segment_map.png'), biggestSI_segment_map)
                        
                        cv2.imwrite(os.path.join(self.path_save, 'img_original.png'), img_original)
                        # cv2.imwrite(os.path.join(self.path_save, 'im0s.png'), im0s)
                        # cv2.imwrite(os.path.join(self.path_save, 'adjusted_brighness.png'), img_depth)
                        if new_tumor_box:
                            # cv2.imwrite(os.path.join(self.path_save, 'mask_SI_tumor.png'), mask_SI_tumor)
                            blend_image = self.blend_images(segmentation_SI_tumor, img_visualize_SI, new_tumor_box)
                            cv2.imwrite(os.path.join(self.path_save, 'blend_image.png'), blend_image)
        MAE = self.calculate_mae()
        print("MAE", MAE)
    
    def blend_images(self, frame, img_visualize_SI, tumor_box):
        # Blend img_visualize_SI with frame to brighten the SI

        x1 = int(tumor_box[0])
        x2 = int(tumor_box[2])
        y1 = int(tumor_box[1])
        y2 = int(tumor_box[3])

        mask = np.zeros_like(frame)
        cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
        mask_inv = cv2.bitwise_not(mask)
        darkened_frame = cv2.bitwise_and(frame, mask_inv)
        brightened_tumors = cv2.bitwise_and(frame, mask)
        brightened_tumors_and_SI = cv2.addWeighted(brightened_tumors, 1.0, img_visualize_SI, 1, 0)
        brightened_tumors_and_SI = cv2.convertScaleAbs(brightened_tumors_and_SI, alpha=1.5, beta=0)
        combined_frame = cv2.add(darkened_frame, brightened_tumors_and_SI)
        # combined_frame = cv2.add(combined_frame, blended_frame)
        return combined_frame

    def find_biggestTumor(self, boxes):
        # Initialize variables to store the largest diameter and corresponding box
        largest_diameter = 0
        largest_box = []
        # print("boxesboxesboxesboxesboxesboxesboxesboxesboxesboxesboxes", boxes)
        for dict_ in boxes:
            if dict_['class']=='Tumor':
                # Iterate through each box
                x1, y1, x2, y2 = dict_['xyxy']
                width = int(x2-x1)
                height = int(y2-y1)
                diameter = np.sqrt(width**2 + height**2)
                
                # Check if the current diameter is larger than the largest diameter found so far
                if diameter > largest_diameter:
                    largest_diameter = diameter
                    largest_box = dict_['xyxy']

        return largest_box, largest_diameter
    
    def create_detection_mask(self, box):

        x1 = int(box[0])
        x2 = int(box[2])
        y1 = int(box[1])
        y2 = int(box[3])
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        center_box = (int(center_x), int(center_y))
        # print("center_box", center_box)
        # Create a mask with zeros
        mask_visualize = np.zeros((self.height, self.width))
        mask_cal = np.zeros((self.height, self.width))

        # Set the pixels within the box to 1
        mask_visualize[y1:y2, x1:x2] = 255
        mask_cal[y1:y2, x1:x2] = 1

        return mask_visualize, mask_cal, center_box
    
    def create_segmentation_mask(self, segment_map):
        '''
        create a mask for surgical instrument
        '''

        # color_SI = self.segmentor.PALETTE[self.segmentor.CLASSES.index("SI")]
        class_number_SI = self.segmentor.CLASSES.index("SI")

        mask_segmentation = np.where(segment_map!=int(class_number_SI),0,255)
        # dimmed_mask_SI = self.dimmed_mask(mask_segmentation, self.img_original)
        # cv2.imwrite(os.path.join(self.path_save, 'dimmed_mask_SI.png'), dimmed_mask_SI)
        if np.mean(mask_segmentation)==0:
            exist_SI = False
            mask_segmentation = None
        else:
            exist_SI = True
            mask_segmentation = self.find_biggest_SI(mask_segmentation)

        return mask_segmentation, exist_SI

    def create_mask_SI_tumor(self, segment_map, intended_tumor):
        '''
        Create a mask for surgical instrument and tumor.
        Find the tumor that overlaps with the intended tumor box and return the new bounding box coordinates.
        '''
        x1 = int(intended_tumor[0])
        x2 = int(intended_tumor[2])
        y1 = int(intended_tumor[1])
        y2 = int(intended_tumor[3])
        
        # Retrieve class indices for SI and Tumor
        class_number_SI = self.segmentor.CLASSES.index("SI")
        class_number_Tumor = self.segmentor.CLASSES.index("Tumor")

        # Create masks where the class matches are set to white (255) and others to black (0)
        mask_segmentation_SI = np.where(segment_map == class_number_SI, 255, 0).astype(np.uint8)
        mask_segmentation_Tumor = np.where(segment_map == class_number_Tumor, 255, 0).astype(np.uint8)

        # Find contours of all tumor regions
        contours, _ = cv2.findContours(mask_segmentation_Tumor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize variables to track the largest overlap
        max_overlap = 0
        best_tumor_bbox = None

        # Define the intended tumor bounding box
        intended_bbox = (x1, y1, x2, y2)

        def bbox_overlap_area(bbox1, bbox2):
            # Calculate overlap area
            x1_max = max(bbox1[0], bbox2[0])
            y1_max = max(bbox1[1], bbox2[1])
            x2_min = min(bbox1[2], bbox2[2])
            y2_min = min(bbox1[3], bbox2[3])
            overlap_width = max(0, x2_min - x1_max)
            overlap_height = max(0, y2_min - y1_max)
            return overlap_width * overlap_height

        # Iterate over all detected tumor contours
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            tumor_bbox = (x, y, x + w, y + h)
            # print("intended_bbox",intended_bbox)
            # print("tumor_bbox",tumor_bbox)
            overlap_area = bbox_overlap_area(intended_bbox, tumor_bbox)
            # print("overlap_area",overlap_area)
            
            if overlap_area > max_overlap:
                max_overlap = overlap_area
                best_tumor_bbox = tumor_bbox

        if best_tumor_bbox:
            x1_new, y1_new, x2_new, y2_new = best_tumor_bbox

            # Create a mask for the best overlapping tumor
            filtered_tumor_mask = np.zeros_like(mask_segmentation_Tumor)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                bbox = (x, y, x + w, y + h)
                if bbox == best_tumor_bbox:
                    cv2.drawContours(filtered_tumor_mask, [contour], -1, 255, thickness=cv2.FILLED)
                    break

            # Combine the SI and filtered Tumor masks
            mask_segmentation = np.maximum(mask_segmentation_SI, filtered_tumor_mask)
            mask_segmentation = cv2.cvtColor(np.uint8(mask_segmentation), cv2.COLOR_GRAY2BGR)
            return mask_segmentation, (x1_new, y1_new, x2_new, y2_new)
        else:
            return None, None
    
    def create_segmentation_SI_Tumor(self, img_original, visual, combined_mask):
        # Initialize a copy of the original image to modify
        merged_image = img_original.copy()

        darkening_factor = 0.5  # This reduces the brightness to 50%
        merged_image = (merged_image * darkening_factor).astype(img_original.dtype)
        # Create a boolean mask where the combined mask is 255 (white)
        mask_boolean = combined_mask == 255

        # Apply the mask to copy SI and Tumor segments from 'visual' to 'merged_image'
        merged_image[mask_boolean] = visual[mask_boolean]

        return merged_image


    def find_biggest_SI(self, mask_SI):

        mask_SI_rgb = cv2.cvtColor(np.uint8(mask_SI), cv2.COLOR_GRAY2BGR)
        
        # cv2.imwrite(os.path.join(self.path_save, 'mask_SI_rgb.png'), mask_SI_rgb)

        mask_SI = cv2.cvtColor(mask_SI_rgb, cv2.COLOR_BGR2GRAY)

        # cv2.imwrite(os.path.join(self.path_save, 'mask_SI_rgb.png'), mask_SI)

        # ret,thresh = cv2.threshold(mask_SI,127,255,0)
        contours,_ = cv2.findContours(mask_SI, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        self.cnt = max(contours, key=cv2.contourArea)

        mask_SI_rgb_copy = np.zeros_like(mask_SI)
        mask_SI_rgb_copy = cv2.cvtColor(np.uint8(mask_SI_rgb_copy), cv2.COLOR_GRAY2BGR)  # Convert to RGB
        img_with_biggest_SI = cv2.drawContours(mask_SI_rgb_copy, [self.cnt], 0, (255, 255, 255), 10)
        # cv2.imwrite('img_with_biggest_SI.png', img_with_biggest_SI)
        return img_with_biggest_SI


    def normalize_depth(self, depth_image):
        '''
        normalize the depth map
        '''
        # Find the maximum pixel value in the image
        max_pixel_value = np.max(depth_image)

        # Normalize the image based on the maximum pixel value
        normalized_image = depth_image / max_pixel_value

        # Ensure the pixel values are in the range [0, 1]
        # normalized_image = np.clip(normalized_image, 0, 1)
        normalized_image = np.exp(normalized_image - 1)
        return normalized_image

    def depth_tumor_SI(self, mask_tumor, rect, idepth, depth_map_visualize):
        '''
        this function returns two value
        depth_value_SI: the depth value of the region that we find the biggest tumor
        depth_value_Tumor: the depth value of the region that we calculate the SI diameter based on its center in rect

        the depth value could be the average of depth values in that region.

        (center(x, y), (width, height), angle of rotation) = rect
        '''
        
        # cv2.imwrite(os.path.join(self.path_save, 'here2.png'), idepth)

        #################################### Depth mask Tumor ####################################
        # mask_tumor = cv2.cvtColor(np.uint8(mask_tumor), cv2.COLOR_GRAY2BGR)
        # mask_tumor = cv2.cvtColor(mask_tumor, cv2.COLOR_BGR2GRAY)
        mask_tumor = np.uint8(mask_tumor)
        # cv2.imwrite(os.path.join(self.path_save, 'here.png'), mask_tumor)

        # Apply the mask to the RGB image
        tumor_depth = cv2.bitwise_and(idepth, idepth, mask=mask_tumor)

        # cv2.imwrite(os.path.join(self.path_save, 'tumor_depth.png'), tumor_depth)

        avg_depth_tumor = self.calculate_average_image_oneChannel(tumor_depth)

        #################################### Depth mask SI ####################################
        ((c_x, c_y), (width_shape, height_shape), angle) = rect

        # Create a new image with the same size as the original image and filled with zeros
        SI_img = np.zeros_like(idepth)
        SI_img_visualize = np.zeros_like(depth_map_visualize)

        # Calculate the top-left and bottom-right corners of the square
        top_left = (int(c_x) - self.args.depth_avg_area, int(c_y) - self.args.depth_avg_area)
        bottom_right = (int(c_x) + self.args.depth_avg_area, int(c_y) + self.args.depth_avg_area)

        # Draw a filled square with size 10x10 centered in the new image
        cv2.rectangle(SI_img, top_left, bottom_right, (255, 255, 255), -1)
        
        cv2.rectangle(SI_img_visualize, top_left, bottom_right, (255, 255, 255), -1)
        # cv2.imwrite(os.path.join(self.path_save, 'SI_depth.png'), SI_img)

        # Copy the original image's square of size 10x10 centered into the new image
        SI_img[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1] = idepth[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1]
        # cv2.imwrite(os.path.join(self.path_save, 'here3.png'), SI_img)
        SI_img_visualize[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1] = depth_map_visualize[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1]
        # Save the result
        # cv2.imwrite(os.path.join(self.path_save, 'SI_depth.png'), SI_img_visualize)

        avg_depth_SI = self.calculate_average_image_oneChannel(SI_img)

        return avg_depth_tumor, avg_depth_SI
    
    def calculate_average_image_oneChannel(self, img):
        # Calculate the sum of the pixel values for each color channel
        non_zero_pixels = np.nonzero(img)
        average = np.mean(img[non_zero_pixels])
        return average

    def minAreaRect_SI(self, mask_biggestSI):
        '''
        https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
        (center(x, y), (width, height), angle of rotation) = rect

        references:
            https://stackoverflow.com/questions/28710337/opencv-lines-passing-through-centroid-of-contour-at-given-angles
        '''
        mask_SI = cv2.cvtColor(mask_biggestSI, cv2.COLOR_BGR2GRAY)
        img = mask_biggestSI.copy()
        x,y,w,h = cv2.boundingRect(self.cnt)
        
        # img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        # compute rotated rectangle (minimum area)
        rect = cv2.minAreaRect(self.cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # draw minimum area rectangle (rotated rectangle)
        # img = cv2.drawContours(img,[box],0,(0,255,255),2)

        # rank based on their y. after sorting extract y1 and y0. the line between these two dots is what we are looking for
            
        ref = mask_SI.copy()
        # cv2.drawContours(ref, contours, 0, 255, 1)
        cv2.drawContours(ref, self.cnt, 0, 255, 1)
        # cv2.imwrite(os.path.join(self.path_save, 'ref.png'), ref)

        tmp = np.zeros_like(mask_SI)

        ((c_x, c_y), (width_shape, height_shape), angle) = rect
        
        # # print("angleangleangleangle", angle)
        # if angle < 45:
        #     theta = (90-angle) * np.pi/180.0
        # if angle >= 45:
        #     theta = (180-angle) * np.pi/180.0
            
        # sorted_box = sorted(box, key=lambda box: box[1])
        # tmp = self.draw_parallel_line(tmp, p1=[c_x, c_y], p2=sorted_box[1], p3=sorted_box[0])

        tmp = self.find_vertical_dots(tmp, p=[c_x, c_y], rectangle=box)
        # cv2.imwrite(os.path.join(self.path_save, 'tmp.png'), tmp)
        # cv2.line(tmp, (int(c_x), int(c_y)),
        #    (int(int(c_x)+np.cos(theta)*self.width),
        #     int(int(c_y)-np.sin(theta)*self.height)), 255, 1)
        
        (row, col) = np.nonzero(np.logical_and(tmp, ref))

        tmp_and = np.logical_and(tmp, ref)
        tmp_rgb = cv2.cvtColor(np.uint8(tmp_and*255), cv2.COLOR_GRAY2BGR)
        # cv2.circle(tmp_rgb,(int(c_x), int(c_y)), radius=0, color=(0, 0, 255), thickness=3)
        # cv2.circle(tmp_rgb,(col[0],row[0]), radius=0, color=(0, 0, 255), thickness=3)
        # cv2.circle(tmp_rgb,(col[-1],row[-1]), radius=0, color=(0, 0, 255), thickness=3)
        # cv2.imwrite(os.path.join(self.path_save, 'tmp_rgb.png'), tmp_rgb)
        
        out_visualize = img + tmp_rgb
        cv2.line(out_visualize, (col[-1],row[-1]), (col[0],row[0]), (0, 0, 255), 10)
        # cv2.imwrite(os.path.join(self.path_save, 'out_visualize.png'), out_visualize)
        ############################################ calculate the length of the SI ############################################
        # (col[0],row[0])
        # (c_x, c_y)
        # length1 = 2 * (np.sqrt((c_x - col[0]) ** 2 + (c_y - row[0]) ** 2))
        length = (np.sqrt((col[-1] - col[0]) ** 2 + (row[-1] - row[0]) ** 2))
        # print("length1", length1)
        # print("length2", length2)

        return length, rect, out_visualize
    
    def minAreaRect_SI2(self, mask_biggestSI):
        '''
        https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
        (center(x, y), (width, height), angle of rotation) = rect

        references:
            https://stackoverflow.com/questions/28710337/opencv-lines-passing-through-centroid-of-contour-at-given-angles
        '''
        mask_SI = cv2.cvtColor(mask_biggestSI, cv2.COLOR_BGR2GRAY)
        # compute rotated rectangle (minimum area)
        rect = cv2.minAreaRect(self.cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        
        ref = mask_SI.copy()
        # cv2.drawContours(ref, contours, 0, 255, 1)
        cv2.drawContours(ref, self.cnt, 0, 255, 1)

        tmp = np.zeros_like(mask_SI)

        ((c_x, c_y), (width_shape, height_shape), angle) = rect

        tmp = self.find_vertical_dots(tmp, p=[c_x, c_y], rectangle=box)
        
        (row, col) = np.nonzero(np.logical_and(tmp, ref))
        
        point1 = (col[0],row[0])
        point2 = (col[-1],row[-1])

        return point1, point2

    def find_vertical_dots(self, image, p, rectangle):
        '''
        get a point and a rectangle
        rectangle contains 4 points that we need to find two vertical lines.
        first we find the closest point (out of rectangle points) to the image center
        then we create three lines from the closest point to the other points.
        then we try to find two vertical lines. then we have these two vertical lines
        and then we can find the smallest line. then we have p1 and p2
        '''
        # print("rectanglerectanglerectanglerectangle", rectangle)
        # Calculate the center of the image
        center_x, center_y = self.width / 2, self.height / 2

        # Initialize variables to track the closest point and its distance
        closest_point = None
        closest_distance = float('inf')  # Initialize with positive infinity

        # Iterate through the four dots and calculate distances
        for dot in rectangle:
            x, y = dot
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

            # Check if this point is closer than the previous closest point
            if distance < closest_distance:
                closest_point = dot
                closest_distance = distance
        
        # print("closest_pointclosest_pointclosest_pointclosest_point", closest_point)
        # Create a new list without the specified element
        new_rectangle = [point for point in rectangle if not np.array_equal(point, closest_point)]
        # print("new_rectanglenew_rectanglenew_rectanglenew_rectangle", new_rectangle)

        
        # Initialize variables to store the perpendicular lines
        perpendicular_lines = []

        closest_point = (x, y) = tuple(closest_point)

        dot1, dot2 = self.find_perpendicular_dots(x, y, new_rectangle)
        
        p2, p3 = self.find_smallest_line(image, closest_point, dot1, dot2)

        # print("perpendicular_lines", perpendicular_lines)
        image = self.draw_parallel_line(image, p, p2, p3)

        return image

    def find_smallest_line(self, image, closest_point, dot1, dot2):
        '''
        take three point. we create two lines between central point to dot1 and dot2
        then we try to find the smalest line.
        '''
        len_side1 = np.sqrt((closest_point[0] - dot1[0])**2 + (closest_point[1] - dot1[1])**2)  
        len_side2 = np.sqrt((closest_point[0] - dot2[0])**2 + (closest_point[1] - dot2[1])**2) 

        # we choose the smaller line because it is parallel to the width
        if len_side1 < len_side2:
            p2 = closest_point
            p3 = dot1 
        else:
            p2 = closest_point
            p3 = dot2 
        
        return p2, p3

    def find_perpendicular_dots(self, x, y, dots):
        '''
        To find the two dots that create the approximately perpendicular lines with the given dot (x, y), you can use the concept of slope.
        Here's a step-by-step approach to solve this problem:
        Calculate the slope between the given dot (x, y) and each of the other three dots using the formula: slope = (y2 - y1) / (x2 - x1).
        Identify the two slopes that are approximately perpendicular. Perpendicular lines have slopes that are negative reciprocals of each other.
        So, look for two slopes that are close to each other and have a product close to -1.
        Once you have identified the two slopes, the corresponding dots will be the ones that have those slopes.
        '''
        min_diff = math.inf
        dot1 = None
        dot2 = None
        # print("dotsdotsdotsdotsdots", dots)
        slopes = []
        for dot in dots:
            slope = (dot[1] - y) / (dot[0] - x + 0.00001)
            slopes.append(slope)

        # print("closest", (x, y))
        # print("dots", dots)
        # print("slopes", slopes)
        # # if the rectangle is horizontal! 
        # if -math.inf in slopes:
        #     dot1 = dots[slopes.index(-math.inf)]
        #     dot2 = dots[slopes.index(0.0)]

        # else:
            # Find the two slopes that are approximately perpendicular
            # print("Hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
        min_angle_diff = float('inf')
        for i in range(len(slopes)):
            for j in range(i + 1, len(slopes)):
                angle_diff = np.abs(np.arctan2(slopes[j], 1) - np.arctan2(slopes[i], 1))
                if angle_diff < np.pi / 2:
                    angle_diff = np.pi - angle_diff  # Correct the angle difference for obtuse angles
                angle_diff = np.abs(angle_diff)  # Ensure the angle difference is non-negative

                if angle_diff < min_angle_diff:
                    min_angle_diff = angle_diff
                    dot1 = dots[i]
                    dot2 = dots[j]

        # print("dot1", dot1)
        # print("dot2", dot2)
        return dot1, dot2

    def draw_parallel_line(self, image, p1, p2, p3):
        '''
        get 3 points. 
        p1: center point
        p2, p3: start and end point of another line
        output: draw a line which cross the p1 and is parallel to the line of p2 and p3
        '''
        # Define the points
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)

        # Calculate the direction vector of the line connecting p1 and p2
        direction_vector = p3 - p2

        # # Calculate the perpendicular vector to the direction vector
        # perpendicular_vector = np.array([-direction_vector[1], direction_vector[0]])

        # Normalize the perpendicular vector
        # normalized_perpendicular_vector = perpendicular_vector / np.linalg.norm(perpendicular_vector)

        normalized_direction_vector = direction_vector / np.linalg.norm(direction_vector)

        # Calculate the start and end points of the parallel line
        start_point = p1 - normalized_direction_vector * self.width  # Adjust the length of the line as needed
        end_point = p1 + normalized_direction_vector * self.height

        # Draw the parallel line on an image
        cv2.line(image, tuple(start_point.astype(int)), tuple(end_point.astype(int)), 255, 1)  # Draw the line
        return image
    
    def find_distances_diameters_v3(self, object_box, reference_box, normalized_depth):
        '''
        based on the following link and the papers of proofs
        paper: A novel absolute localization estimation of a target with monocular vision
        '''
        # print("reference_box", reference_box)
        # print("object_box", object_box)

        U_obj1, V_obj1, U_obj2, V_obj2 = object_box
        U_ref1, V_ref1, U_ref2, V_ref2 = reference_box

        U_obj1 = int(U_obj1)
        V_obj1 = int(V_obj1)
        U_obj2 = int(U_obj2)
        V_obj2 = int(V_obj2)
        U_ref1 = int(U_ref1)
        V_ref1 = int(V_ref1)
        U_ref2 = int(U_ref2)
        V_ref2 = int(V_ref2)

        top_left_ref = (U_ref1, V_ref1)
        bottom_right_ref = (U_ref2, V_ref2)

        width_px_obj = (U_obj2 - U_obj1)
        hight_px_obj = (V_obj2 - V_obj1)
        S_px_ref = math.sqrt((U_ref2 - U_ref1)**2 + (V_ref2 - V_ref1)**2)
        
        # S_px_obj = width_px_obj * hight_px_obj

        U_Fref = (U_ref1 + U_ref2) / 2
        V_Fref = (V_ref1 + V_ref2) / 2
        U_Fobj = (U_obj1 + U_obj2) / 2
        V_Fobj = (V_obj1 + V_obj2) / 2
        m = ((U_Fref - self.args.c_x) / self.args.f_x)**2 + ((V_Fref - self.args.c_y) / self.args.f_y)**2
        n = ((U_Fobj - self.args.c_x) / self.args.f_x)**2 + ((V_Fobj - self.args.c_y) / self.args.f_y)**2

        d_ref = np.sqrt((self.args.S_real_ref * self.args.f_x * self.args.f_y) / S_px_ref)
        Z_ref = d_ref * np.sqrt(1 + m)

        Depth_obj = normalized_depth[int(V_Fobj) - 1, int(U_Fobj) - 1]
        Depth_ref = normalized_depth[int(V_Fref) - 1, int(U_Fref) - 1]
        Z_obj = Z_ref * (Depth_ref / Depth_obj)

        # angle_obj_ref = math.atan2(abs(V_Fref - V_Fobj), abs(U_Fref - U_Fobj))
        # print("math.sin(angle_obj_ref)", math.sin(angle_obj_ref))
        # Z_obj = Z_ref + (Z_obj - Z_ref) * math.sin(angle_obj_ref)

        H_real_obj = self.args.S_real_ref * (hight_px_obj / S_px_ref) * ((1 + m) / (1 + n)) * ((Z_obj / Z_ref) ** 2)
        W_real_obj = self.args.S_real_ref * (width_px_obj / S_px_ref) * ((1 + m) / (1 + n)) * ((Z_obj / Z_ref) ** 2)
        # S_real_obj = S_real_ref * (S_px_obj / S_px_ref) * ((1+n)/(1+m)) * ((Z_ref/Z_obj)**2)

        # real_W_obj, real_H_obj = self.get_H_W(S_real_obj, width_px_obj, hight_px_obj)     

        return W_real_obj, H_real_obj, Z_ref, Z_obj
    
    def get_H_W(self, S_real_obj, width_px_obj, hight_px_obj):

        ratio_px = width_px_obj / hight_px_obj
        real_H_obj = np.sqrt(S_real_obj / ratio_px)
        real_W_obj = ratio_px * real_H_obj
        return real_W_obj, real_H_obj

    
    def visualize_function(self, image_visualize2, img_visualize, biggestSI_segment_map, rect_SI, real_W_obj, real_H_obj, Z_ref, Z_obj, largestTumor, mask_vis_tumor):
        
        x1_tumor = int(largestTumor[0])
        x2_tumor = int(largestTumor[2])
        y1_tumor = int(largestTumor[1])
        y2_tumor = int(largestTumor[3])
        center_x_tumor = (x1_tumor + x2_tumor) / 2
        center_y_tumor = (y1_tumor + y2_tumor) / 2

        center_box_tumor = (int(center_x_tumor), int(center_y_tumor))

        ((c_x_SI, c_y_SI), (width_shape_SI, height_shape_SI), angle) = rect_SI

        mask_vis_tumor = cv2.cvtColor(np.uint8(mask_vis_tumor), cv2.COLOR_GRAY2BGR)
        img_visualize_1 = mask_vis_tumor + img_visualize

        # Set the pixels within the box to 1
        img_visualize_1[y1_tumor:y2_tumor, x1_tumor:x2_tumor] = 255
        
        # img_visualize_1 = cv2.putText(img_visualize_1, f'SI pixel:{int(self.diameter_SI_pixel)}', (int(c_x_SI), int(c_y_SI) + 10), cv2.FONT_HERSHEY_COMPLEX,  
        #            0.4, (255, 255, 0), 1, cv2.LINE_AA) 
        # img_visualize_1 = cv2.circle(img_visualize_1, (int(c_x_SI), int(c_y_SI)), radius=0, color=(0, 255, 0), thickness=3)

        #################### put text for tumor box #######################
        text_above_position = (x1_tumor - 100, y1_tumor - 10)
        text_below_position = (x1_tumor - 100, y1_tumor + 10) 
        text_belowbelow_position = (x1_tumor - 100, y1_tumor + 30) 
        img_visualize_2 = img_visualize_1.copy()
        # cv2.imwrite(os.path.join(self.path_save, 'visualize1.png'), img_visualize_1)

        img_visualize_2 = cv2.putText(img_visualize_2, f'Tumor Horizontal:{"{:.2f}".format(real_W_obj)}', text_above_position, cv2.FONT_HERSHEY_PLAIN,  
                   1.4, (0, 128, 255), 1, cv2.LINE_AA) 
        img_visualize_2 = cv2.putText(img_visualize_2, f'Tumor Vertical:{"{:.2f}".format(real_H_obj)}', text_below_position, cv2.FONT_HERSHEY_PLAIN,  
                   1.4, (0, 128, 255), 1, cv2.LINE_AA) 
        img_visualize_2 = cv2.putText(img_visualize_2, f'Tumor diagonal:{"{:.2f}".format(np.sqrt(real_W_obj**2 + real_H_obj**2))}', text_belowbelow_position, cv2.FONT_HERSHEY_PLAIN,  
                   1.4, (0, 128, 255), 1, cv2.LINE_AA) 
        
        # img_visualize_2 = cv2.putText(img_visualize_2, f'diameter_tumor_pixel:{int(diameter_box_tumor)}', text_below_position, cv2.FONT_HERSHEY_COMPLEX,  
        #            0.4, (0, 255, 0), 1, cv2.LINE_AA) 

        cv2.circle(img_visualize_2, center_box_tumor, radius=0, color=(0, 0, 255), thickness=3)
        ############################################################
        cv2.imwrite(os.path.join(self.path_save, 'visualize2.png'), img_visualize_2)


        # Convert the original image to grayscale
        gray_image = cv2.cvtColor(img_visualize_1, cv2.COLOR_BGR2GRAY)
        # Apply lower intensity to create a dimmed effect
        dim_factor = 0.5  # Adjust this value for the desired dimming effect
        dimmed_image = (dim_factor * gray_image).astype(np.uint8)
        # Blend the dimmed image with the original image
        dimmed = cv2.addWeighted(img_visualize_1, 0.5, cv2.cvtColor(dimmed_image, cv2.COLOR_GRAY2BGR), 0.5, 0)
        # cv2.imwrite(os.path.join(self.path_save, 'dimmed.png'), dimmed)

        # Convert the color mask to grayscale
        # gray_mask = cv2.cvtColor(img_visualize_1, cv2.COLOR_BGR2GRAY)
        # Blend the images based on the mask
        brightness_factor = 0.5  # Adjust this value for the desired brightness effect
        blended_image = cv2.addWeighted(dimmed, 1, img_visualize_1, brightness_factor, 0)
        # cv2.imwrite(os.path.join(self.path_save, 'blended_image.png'), blended_image)


        # img_visualize_2 = img_visualize_1 + dimmed_img0s
        # cv2.imwrite(os.path.join(self.path_save, 'visualize2.png'), img_visualize_2)

    def dimmed_mask(self, mask, original_image):
        '''
        lower the brighness of every pixel which is not in the mask
        '''

        # Set the brightness reduction factor for pixels outside the mask
        brightness_reduction_factor = 0.5  # You can adjust this value as needed

        # Create a copy of the original image to modify
        result_image = original_image.copy()

        # Apply the brightness reduction to pixels outside the mask
        result_image[mask == 0] = result_image[mask == 0] * brightness_reduction_factor
        return result_image

    def calculate_mae(self):
        list_predicted = []
        list_type_predictions = []
        list_gt = []
        list_sorted_dicts = []
        # self.list_results: [{'image_name': "AGRMNPRZ_P2_11635", "width": " ", "hight": " ", "diagonal": " "}, {}, ...]

        # Load the Excel file
        excel_data = pd.read_excel(self.args.ground_truth_path, header=None, engine='openpyxl')

        # Iterate through rows
        for index, row in excel_data.iterrows():
            # print(row.values)
            if index > 0: # to pass the first row
                if row.values[0]=='Mean size':
                    frame_numbers = row.values[5]
                    # print("frame_numbers", frame_numbers)
                    video_name = row.values[2]
                    video_name = video_name.replace(" ", "")
                    video_name = video_name.rstrip('\xa0')
                    size_gt = row.values[4]
                    type_measuring = row.values[6]
                    frame_numbers = str(frame_numbers).split(',')
                    for frame_number in frame_numbers:
                        frame_number = frame_number.replace(" ", "")
                        image_name_to_find = video_name + '_' + frame_number
                        
                        founded_dict = self.find_dictionary(image_name_to_find, self.list_results)
                        if founded_dict is None:
                            print("image_name_to_find", image_name_to_find)
                            continue
                        list_sorted_dicts.append(founded_dict)
                        list_gt.append(size_gt)
                        list_predicted.append(founded_dict[type_measuring])
                        list_type_predictions.append(type_measuring)

                        # print("image_name_to_find", image_name_to_find)
                        # print("list_gt", size_gt)
                        # print("list_predicted", founded_dict[type_measuring])

        MAE = self.mae(list_gt, list_predicted, list_type_predictions, list_sorted_dicts)
        return MAE

    def find_dictionary(self, image_name_to_find, dictionary_list):
        '''
        find a specific dictionary in a list of dictionaries using one of the keys
        '''
        for dictionary in dictionary_list:
            if dictionary.get('image_name') == image_name_to_find:
                return dictionary
        return None  # Return None if not found

    def mae(self, list1, list2, list_type_predictions, list_sorted_dicts):
        # Check if the input lists have the same length
        if len(list1) != len(list2):
            raise ValueError("Input lists must have the same length.")

        # Convert elements to floats (if they are strings)
        list1 = [float(item) if isinstance(item, str) else item for item in list1]
        list2 = [float(item) if isinstance(item, str) else item for item in list2]
        list2 = ["{:.2f}".format(number) for number in list2]
        list2 = [float(item) for item in list2]

        # print("list_sorted_dicts", list_sorted_dicts)
        print("list gt", list1)
        # print("list gt", sum(list1))
        print("list predictions", list2)
        # print("list_type_predictions", list_type_predictions)
        # Calculate the absolute differences between corresponding elements

        ######################## To find the big errors #############################
        big_errors = []
        absolute_errors = []
        for a, b, dict_ in zip(list1, list2, list_sorted_dicts):
            error = abs(a - b)
            absolute_errors.append(error)
            if error > 5:
                big_errors.append(dict_['image_name'])
        print("len(big_errors)", len(big_errors))
        print("big_errors", big_errors)
        #######################################################################################

        ##################################################################################
        small_errors_big_tumors = []
        # absolute_errors = []
        for a, b, dict_ in zip(list1, list2, list_sorted_dicts):
            error = abs(a - b)
            # absolute_errors.append(error)
            if a > 20:
                # print('here')
                if error < 5:
                    small_errors_big_tumors.append(dict_['image_name'])
        print("len(small_errors_big_tumors)", len(small_errors_big_tumors))
        print("small_errors_big_tumors", small_errors_big_tumors)
        #######################################################################################

        # Calculate the mean absolute error (MAE)
        mae = sum(absolute_errors) / len(list1)

        # # to visualize
        # for iii in range(len(list1)):
        #     print("sorted_dict", list_sorted_dicts[iii])
        #     print("gt", list1[iii])
        #     print("prediction", list2[iii])
        #     print("type_prediction", list_type_predictions[iii])

        return mae

    # def clahe(self, image):
    #     '''
    #     Contrast Limited Adaptive Histogram Equalization
    #     '''
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    #     clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(64,64))
    #     #0 to 'L' channel, 1 to 'a' channel, and 2 to 'b' channel
    #     image[:,:,0] = clahe.apply(image[:,:,0])

    #     image = cv2.cvtColor(image, cv2.COLOR_Lab2RGB)

    #     return image

if __name__ == '__main__':
    conf_obj = Config()
    args = conf_obj.get_args()
    obj_run = Run(args)
    obj_run.start_inferencing()
