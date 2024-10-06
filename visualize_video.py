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
import os, subprocess
from config import Config
# from run import *
from MiDaS.run_EMA_OpticalFlow import *
from midas.model_loader import default_models, load_model
import math
from segment_frames_mask2former import SegmentMMSegmentation
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from tracker.sort_tracker import Sort  # Import SORT tracker
from filterpy.kalman import KalmanFilter

author = "Seyed Amir Mousavi"
credits = ["Amir Mousavi"]
license = "Public Domain"
version = "1.0.0"
maintainer = "Seyed Amir Mousavi"
email = "seyedamir.mousavi@ghent.ac.kr"
status = "Research"

class Run:
    def __init__(self, args):
        super(Run, self).__init__()

        self.args = args
        self.segmentor = SegmentMMSegmentation(self.args)
        self.depther = DepthEstimator(self.args)
        self.tracker = Sort(max_age=25, min_hits=3, iou_threshold=0.5)  # Initialize SORT tracker
        self.tumor_measurements = {}  # Dictionary to store measurements for each tumor ID
        self.si_frame_count = 0
        self.both_SI_Tumor_appear = 0
        self.video_writer = None
        self.list_results = []
        self.kalman_filters = {}
        self.ema_values = {}
        self.ema_alpha = 0.1  # EMA alpha value

        self.stop_processing_seconds = 1
        self.both_SI_Tumor_appear = False
        self.show_and_stop_number = 1
        self.start_processing_atFrame = 1

    def initialize_video_writer(self, frame_width, frame_height, fps, output_path):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), True)

    def start_inferencing(self):
        if self.args.output_path is not None:
            os.makedirs(self.args.output_path, exist_ok=True)
        cap = cv2.VideoCapture(self.args.video_path)
        assert (cap.isOpened())
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.input_fps = cap.get(cv2.CAP_PROP_FPS)

        # Initialize output video writer
        output_file = os.path.join(self.args.output_file, self.args.video_path.split('/')[-1])
        self.initialize_video_writer(self.width * 2, self.height * 2, self.input_fps, output_file)

        frame_counter = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if frame_counter > self.start_processing_atFrame:
                if ret:
                    img_segmentation = frame.copy()
                    img_depth = frame.copy()
                    self.img_original = frame.copy()

                    visual_segmentation, segment_map = self.segmentor.segment(img_segmentation)
                    biggestSI_segment_map, exist_SI, tumor_boxes, exist_Tumor = self.create_segmentation_mask(segment_map[0])
                    prediction_depth, depth_map_visualize, raw_depth255, idepth = self.depther.run(img_depth, self.args.output_file)
                    
                    if exist_Tumor and not exist_SI:
                        for box in tumor_boxes:
                            x1, y1, x2, y2 = box
                            frame = self.draw_box(frame, x1, y1, x2, y2)

                    if exist_SI and exist_Tumor:
                        self.si_frame_count += 1
                        detections = self.prepare_detections(tumor_boxes)
                        tracked_objects = self.tracker.update(detections)

                        normalized_depth = self.normalize_depth(prediction_depth)
                        self.diameter_SI_pixel, rect_SI, img_visualize_SI = self.minAreaRect_SI(biggestSI_segment_map)
                        Ps1, Ps2 = self.minAreaRect_SI2(biggestSI_segment_map)
                        SI_box = (*Ps1, *Ps2)
                        for track in tracked_objects:
                            tumor_id = int(track[4])
                            x1, y1, x2, y2 = map(int, track[:4])
                            tumor_box = (x1, y1, x2, y2)

                            # Smooth the bounding box coordinates
                            if self.args.use_kalman_bbox:
                                tumor_box = self.smooth_bounding_box_kf(tumor_id, tumor_box)

                            frame = self.draw_box(frame, tumor_box[0], tumor_box[1], tumor_box[2], tumor_box[3])

                            mask_visualize_tumor, mask_cal_tumor, center_tumor_box = self.create_detection_mask(tumor_box)
                            avg_depth_SI, avg_depth_tumor = self.depth_tumor_SI(mask_cal_tumor, rect_SI, normalized_depth, depth_map_visualize)
                            real_W_obj, real_H_obj, Z_ref, Z_obj = self.find_distances_diameters_v3(tumor_box, SI_box, normalized_depth)
                            diameter = np.sqrt(real_W_obj**2 + real_H_obj**2)
                            
                            if tumor_id not in self.tumor_measurements:
                                self.tumor_measurements[tumor_id] = []

                            self.tumor_measurements[tumor_id].append({
                                "horizontal length": real_W_obj,
                                "vertical length": real_H_obj,
                                "diagonal": diameter
                            })

                        # Blend img_visualize_SI with frame to brighten the SI
                        calculation_frame = self.blend_images(frame, img_visualize_SI)

                        if self.si_frame_count == self.show_and_stop_number:
                            if self.args.average_method:
                                self.calculate_and_store_average_measurements()

                            if self.args.mae_method:
                                self.smooth_measurements_mae()

                            # Process to show intensified tumor colors and decreased brightness of other pixels
                            frame = self.intensify_tumor_and_darken_others(frame, tracked_objects)

                            # Pause processing and show results for a few seconds
                            self.display_results_for_few_seconds(self.img_original, visual_segmentation, calculation_frame, depth_map_visualize, tracked_objects)

                        combined_frame = self.create_combined_frame(self.img_original, visual_segmentation, calculation_frame, depth_map_visualize)
                        self.video_writer.write(combined_frame)

                    else:
                        # Show segmentation map if no SI and Tumor
                        combined_frame = self.create_combined_frame(self.img_original, visual_segmentation, frame, depth_map_visualize)
                        self.video_writer.write(combined_frame)
            frame_counter += 1
        cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()

    def prepare_detections(self, tumor_boxes):
        detections = []
        for box in tumor_boxes:
            x1, y1, x2, y2 = box
            score = 1.0  # Assume perfect detection score
            detections.append([x1, y1, x2, y2, score])
        return np.array(detections)

    def initialize_kalman_filter(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
        kf.P *= 1000.0
        kf.R *= 10.0
        kf.Q = np.eye(4)
        return kf

    def smooth_bounding_box_kf(self, tumor_id, current_box):
        if tumor_id not in self.kalman_filters:
            self.kalman_filters[tumor_id] = self.initialize_kalman_filter()
            self.kalman_filters[tumor_id].x[:2] = np.array(current_box[:2]).reshape(2, 1)

        kf = self.kalman_filters[tumor_id]
        kf.predict()
        kf.update(np.array(current_box[:2]).reshape(2, 1))

        smoothed_box = kf.x[:2].flatten()
        smoothed_box = np.concatenate((smoothed_box, current_box[2:4]))
        return [int(coord) for coord in smoothed_box]

    def smooth_bounding_box_ema(self, tumor_id, current_box):
        if tumor_id not in self.ema_values:
            self.ema_values[tumor_id] = np.array(current_box, dtype=float)
        else:
            self.ema_values[tumor_id] = self.ema_alpha * np.array(current_box) + (1 - self.ema_alpha) * self.ema_values[tumor_id]
        return [int(coord) for coord in self.ema_values[tumor_id]]

    def draw_box(self, frame, x1, y1, x2, y2, color=(0, 255, 0)):
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        return frame

    def calculate_and_store_average_measurements(self):
        avg_measurements = {}
        for tumor_id, measurements in self.tumor_measurements.items():
            avg_measurements[tumor_id] = {
                "horizontal length": np.mean([m["horizontal length"] for m in measurements]),
                "vertical length": np.mean([m["vertical length"] for m in measurements]),
                "diagonal": np.mean([m["diagonal"] for m in measurements])
            }
        self.list_results.append(avg_measurements)
        self.tumor_measurements = {}
        self.si_frame_count = 0

    def smooth_measurements_mae(self):
        smoothed_measurements = {}
        for tumor_id, measurements in self.tumor_measurements.items():
            if tumor_id not in self.ema_values:
                self.ema_values[tumor_id] = {
                    "horizontal length": measurements[0]["horizontal length"],
                    "vertical length": measurements[0]["vertical length"],
                    "diagonal": measurements[0]["diagonal"]
                }
            for measurement in measurements:
                self.ema_values[tumor_id]["horizontal length"] = self.ema_alpha * measurement["horizontal length"] + (1 - self.ema_alpha) * self.ema_values[tumor_id]["horizontal length"]
                self.ema_values[tumor_id]["vertical length"] = self.ema_alpha * measurement["vertical length"] + (1 - self.ema_alpha) * self.ema_values[tumor_id]["vertical length"]
                self.ema_values[tumor_id]["diagonal"] = self.ema_alpha * measurement["diagonal"] + (1 - self.ema_alpha) * self.ema_values[tumor_id]["diagonal"]

            smoothed_measurements[tumor_id] = {
                "horizontal length": self.ema_values[tumor_id]["horizontal length"],
                "vertical length": self.ema_values[tumor_id]["vertical length"],
                "diagonal": self.ema_values[tumor_id]["diagonal"]
            }
        self.list_results.append(smoothed_measurements)
        self.tumor_measurements = {}
        self.si_frame_count = 0

    def intensify_tumor_and_darken_others(self, frame, tracked_objects):
        mask = np.zeros_like(frame)
        for track in tracked_objects:
            x1, y1, x2, y2 = map(int, track[:4])
            cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
        mask_inv = cv2.bitwise_not(mask)
        darkened_frame = cv2.bitwise_and(frame, mask_inv)
        brightened_tumors = cv2.bitwise_and(frame, mask)
        brightened_tumors = cv2.convertScaleAbs(brightened_tumors, alpha=1.5, beta=0)
        combined_frame = cv2.add(darkened_frame, brightened_tumors)
        return combined_frame

    def blend_images(self, frame, img_visualize_SI):
        blended_frame = cv2.addWeighted(frame, 1.0, img_visualize_SI, 0.5, 0)
        return blended_frame

    def display_results_for_few_seconds(self, img_original, visual_segmentation, calculation_frame, depth_map_visualize, tracked_objects):
        avg_measurements = self.list_results[-1]
        
        horizontal_color = (255, 0, 0)
        vertical_color = (0, 255, 0)
        diagonal_color = (0, 0, 255)
        
        for track in tracked_objects:
            tumor_id = int(track[4])
            x1, y1, x2, y2 = map(int, track[:4])
            measurements = avg_measurements[tumor_id]
            
            horizontal_length = measurements["horizontal length"]
            vertical_length = measurements["vertical length"]
            diagonal_length = measurements["diagonal"]
            
            mid_y = (y1 + y2) // 2
            cv2.line(calculation_frame, (x1, mid_y), (x2, mid_y), horizontal_color, 1)
            
            mid_x = (x1 + x2) // 2
            cv2.line(calculation_frame, (mid_x, y1), (mid_x, y2), vertical_color, 1)
            
            cv2.line(calculation_frame, (x1, y1), (x2, y2), diagonal_color, 1)
            
            text_x = x2 + 10 if x2 < self.width - 100 else x1 - 100
            text_y = y1
            
            cv2.putText(calculation_frame, f'{horizontal_length:.2f} mm', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, horizontal_color, 1, cv2.LINE_AA)
            cv2.putText(calculation_frame, f'{vertical_length:.2f} mm', (text_x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, vertical_color, 1, cv2.LINE_AA)
            cv2.putText(calculation_frame, f'{diagonal_length:.2f} mm', (text_x, text_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, diagonal_color, 1, cv2.LINE_AA)

        combined_frame = self.create_combined_frame(img_original, visual_segmentation, calculation_frame, depth_map_visualize)
        self.video_writer.write(combined_frame)

    def create_combined_frame(self, original_frame, segmented_frame, calculation_frame, depth_frame):
        top_row = np.hstack((original_frame, segmented_frame))
        bottom_row = np.hstack((calculation_frame, depth_frame))
        combined_frame = np.vstack((top_row, bottom_row))
        return combined_frame

    def find_Tumors_boxes(self, mask_segmentation_tumors):
        contours, _ = cv2.findContours(mask_segmentation_tumors, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tumor_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            tumor_boxes.append((x, y, x + w, y + h))
        return tumor_boxes
    
    def create_segmentation_mask(self, segment_map):
        class_number_SI = self.segmentor.CLASSES.index("SI")
        mask_segmentation_SI = np.where(segment_map != int(class_number_SI), 0, 255)
        class_number_tumor = self.segmentor.CLASSES.index("Tumor")
        mask_segmentation_tumors = np.where(segment_map != int(class_number_tumor), 0, 255).astype(np.uint8)
        if np.mean(mask_segmentation_tumors) == 0:
            exist_Tumor = False
            tumor_boxes = None
        else:
            exist_Tumor = True
            tumor_boxes = self.find_Tumors_boxes(mask_segmentation_tumors)

        if np.mean(mask_segmentation_SI) == 0:
            exist_SI = False
            mask_segmentation_SI = None
        else:
            exist_SI = True
            mask_segmentation_SI = self.find_biggest_SI(mask_segmentation_SI)
        return mask_segmentation_SI, exist_SI, tumor_boxes, exist_Tumor
    
    def find_biggest_SI(self, mask_SI):
        mask_SI_rgb = cv2.cvtColor(np.uint8(mask_SI), cv2.COLOR_GRAY2BGR)
        mask_SI = cv2.cvtColor(mask_SI_rgb, cv2.COLOR_BGR2GRAY)
        contours,_ = cv2.findContours(mask_SI, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.cnt = max(contours, key=cv2.contourArea)
        mask_SI_rgb_copy = np.zeros_like(mask_SI)
        mask_SI_rgb_copy = cv2.cvtColor(np.uint8(mask_SI_rgb_copy), cv2.COLOR_GRAY2BGR)
        img_with_biggest_SI = cv2.drawContours(mask_SI_rgb_copy, [self.cnt], 0, (255, 255, 255), 10)
        return img_with_biggest_SI

    def create_detection_mask(self, box):
        x1 = int(box[0])
        x2 = int(box[2])
        y1 = int(box[1])
        y2 = int(box[3])
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center_box = (int(center_x), int(center_y))
        mask_visualize = np.zeros((self.height, self.width))
        mask_cal = np.zeros((self.height, self.width))
        mask_visualize[y1:y2, x1:x2] = 255
        mask_cal[y1:y2, x1:x2] = 1
        return mask_visualize, mask_cal, center_box

    def normalize_depth(self, depth_image):
        max_pixel_value = np.max(depth_image)
        normalized_image = depth_image / max_pixel_value
        normalized_image = np.exp(normalized_image - 1)
        return normalized_image

    def depth_tumor_SI(self, mask_tumor, rect, idepth, depth_map_visualize):
        mask_tumor = np.uint8(mask_tumor)
        tumor_depth = cv2.bitwise_and(idepth, idepth, mask=mask_tumor)
        avg_depth_tumor = self.calculate_average_image_oneChannel(tumor_depth)
        ((c_x, c_y), (width_shape, height_shape), angle) = rect
        SI_img = np.zeros_like(idepth)
        SI_img_visualize = np.zeros_like(depth_map_visualize)
        top_left = (int(c_x) - self.args.depth_avg_area, int(c_y) - self.args.depth_avg_area)
        bottom_right = (int(c_x) + self.args.depth_avg_area, int(c_y) + self.args.depth_avg_area)
        cv2.rectangle(SI_img, top_left, bottom_right, (255, 255, 255), -1)
        cv2.rectangle(SI_img_visualize, top_left, bottom_right, (255, 255, 255), -1)
        SI_img[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1] = idepth[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1]
        SI_img_visualize[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1] = depth_map_visualize[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1]
        avg_depth_SI = self.calculate_average_image_oneChannel(SI_img)
        return avg_depth_tumor, avg_depth_SI

    def calculate_average_image_oneChannel(self, img):
        non_zero_pixels = np.nonzero(img)
        average = np.mean(img[non_zero_pixels])
        return average

    def minAreaRect_SI(self, mask_biggestSI):
        mask_SI = cv2.cvtColor(mask_biggestSI, cv2.COLOR_BGR2GRAY)
        img = mask_biggestSI.copy()
        x,y,w,h = cv2.boundingRect(self.cnt)
        rect = cv2.minAreaRect(self.cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        ref = mask_SI.copy()
        cv2.drawContours(ref, self.cnt, 0, 255, 1)
        tmp = np.zeros_like(mask_SI)
        ((c_x, c_y), (width_shape, height_shape), angle) = rect
        tmp = self.find_vertical_dots(tmp, p=[c_x, c_y], rectangle=box)
        (row, col) = np.nonzero(np.logical_and(tmp, ref))
        tmp_and = np.logical_and(tmp, ref)
        tmp_rgb = cv2.cvtColor(np.uint8(tmp_and*255), cv2.COLOR_GRAY2BGR)
        out_visualize = img + tmp_rgb
        cv2.line(out_visualize, (col[-1],row[-1]), (col[0],row[0]), (0, 0, 255), 10)
        length = (np.sqrt((col[-1] - col[0]) ** 2 + (row[-1] - row[0]) ** 2))
        return length, rect, out_visualize
    
    def minAreaRect_SI2(self, mask_biggestSI):
        mask_SI = cv2.cvtColor(mask_biggestSI, cv2.COLOR_BGR2GRAY)
        rect = cv2.minAreaRect(self.cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        ref = mask_SI.copy()
        cv2.drawContours(ref, self.cnt, 0, 255, 1)
        tmp = np.zeros_like(mask_SI)
        ((c_x, c_y), (width_shape, height_shape), angle) = rect
        tmp = self.find_vertical_dots(tmp, p=[c_x, c_y], rectangle=box)
        (row, col) = np.nonzero(np.logical_and(tmp, ref))
        point1 = (col[0],row[0])
        point2 = (col[-1],row[-1])
        return point1, point2

    def find_vertical_dots(self, image, p, rectangle):
        center_x, center_y = self.width / 2, self.height / 2
        closest_point = None
        closest_distance = float('inf')
        for dot in rectangle:
            x, y = dot
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            if distance < closest_distance:
                closest_point = dot
                closest_distance = distance
        new_rectangle = [point for point in rectangle if not np.array_equal(point, closest_point)]
        closest_point = (x, y) = tuple(closest_point)
        dot1, dot2 = self.find_perpendicular_dots(x, y, new_rectangle)
        p2, p3 = self.find_smallest_line(image, closest_point, dot1, dot2)
        image = self.draw_parallel_line(image, p, p2, p3)
        return image

    def find_smallest_line(self, image, closest_point, dot1, dot2):
        len_side1 = np.sqrt((closest_point[0] - dot1[0])**2 + (closest_point[1] - dot1[1])**2)  
        len_side2 = np.sqrt((closest_point[0] - dot2[0])**2 + (closest_point[1] - dot2[1])**2)
        if (len_side1 < len_side2):
            p2 = closest_point
            p3 = dot1 
        else:
            p2 = closest_point
            p3 = dot2 
        return p2, p3

    def find_perpendicular_dots(self, x, y, dots):
        min_diff = math.inf
        dot1 = None
        dot2 = None
        slopes = []
        for dot in dots:
            slope = (dot[1] - y) / (dot[0] - x + 0.00001)
            slopes.append(slope)
        min_angle_diff = float('inf')
        for i in range(len(slopes)):
            for j in range(i + 1, len(slopes)):
                angle_diff = np.abs(np.arctan2(slopes[j], 1) - np.arctan2(slopes[i], 1))
                if angle_diff < np.pi / 2:
                    angle_diff = np.pi - angle_diff
                angle_diff = np.abs(angle_diff)
                if (angle_diff < min_angle_diff):
                    min_angle_diff = angle_diff
                    dot1 = dots[i]
                    dot2 = dots[j]
        return dot1, dot2

    def draw_parallel_line(self, image, p1, p2, p3):
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        direction_vector = p3 - p2
        normalized_direction_vector = direction_vector / np.linalg.norm(direction_vector)
        start_point = p1 - normalized_direction_vector * self.width
        end_point = p1 + normalized_direction_vector * self.height
        cv2.line(image, tuple(start_point.astype(int)), tuple(end_point.astype(int)), 255, 1)
        return image
    
    def find_distances_diameters_v3(self, object_box, reference_box, normalized_depth):
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
        H_real_obj = self.args.S_real_ref * (hight_px_obj / S_px_ref) * ((1 + m) / (1 + n)) * ((Z_obj / Z_ref) ** 2)
        W_real_obj = self.args.S_real_ref * (width_px_obj / S_px_ref) * ((1 + m) / (1 + n)) * ((Z_obj / Z_ref) ** 2)
        return W_real_obj, H_real_obj, Z_ref, Z_obj

if __name__ == '__main__':
    conf_obj = Config()
    args = conf_obj.get_args()
    obj_run = Run(args)
    obj_run.start_inferencing()