import numpy as np
import cv2

class SegmentationUtils:
    def __init__(self, args, segmentor_obj):
        self.args = args
        self.segmentor_obj = segmentor_obj

    def blend_images_v1(self, frame, img_visualize_SI, tumor_box):
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

    def find_biggestTumor_v1(self, boxes):
        # Initialize variables to store the largest diameter and corresponding box
        largest_diameter = 0
        largest_box = []
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

    def create_detection_mask_v1(self, box, height, width):

        x1 = int(box[0])
        x2 = int(box[2])
        y1 = int(box[1])
        y2 = int(box[3])
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        center_box = (int(center_x), int(center_y))
        # print("center_box", center_box)
        # Create a mask with zeros
        mask_visualize = np.zeros((height, width))
        mask_cal = np.zeros((height, width))

        # Set the pixels within the box to 1
        mask_visualize[y1:y2, x1:x2] = 255
        mask_cal[y1:y2, x1:x2] = 1

        return mask_visualize, mask_cal, center_box

    def bbox_overlap_area(self, bbox1, bbox2):
        # Calculate overlap area
        x1_max = max(bbox1[0], bbox2[0])
        y1_max = max(bbox1[1], bbox2[1])
        x2_min = min(bbox1[2], bbox2[2])
        y2_min = min(bbox1[3], bbox2[3])
        overlap_width = max(0, x2_min - x1_max)
        overlap_height = max(0, y2_min - y1_max)
        return overlap_width * overlap_height

    def create_segmentation_mask_v1(self, segment_map):
        '''
        create a mask for surgical instrument
        '''

        # color_SI = self.segmentor_obj.PALETTE[self.segmentor_obj.CLASSES.index("SI")]
        class_number_SI = self.segmentor_obj.CLASSES.index("SI")

        mask_segmentation = np.where(segment_map!=int(class_number_SI),0,255)
        # dimmed_mask_SI = self.dimmed_mask(mask_segmentation, self.img_original)
        # cv2.imwrite(os.path.join(self.path_save, 'dimmed_mask_SI.png'), dimmed_mask_SI)
        if np.mean(mask_segmentation)==0:
            exist_SI = False
            mask_segmentation = None
        else:
            exist_SI = True
            mask_segmentation, cnt = self.find_biggest_SI(mask_segmentation)

        return mask_segmentation, exist_SI, cnt
    
    def create_segmentation_mask_v2(self, segment_map):
        '''
        create a mask for surgical instrument
        '''

        # color_SI = self.segmentor_obj.PALETTE[self.segmentor_obj.CLASSES.index("SI")]
        class_number_SI = self.segmentor_obj.CLASSES.index("SI")
        si_mask = (segment_map == class_number_SI).astype(np.uint8)

        if np.mean(si_mask)==0:
            exist_SI = False
            si_mask = None
        else:
            exist_SI = True

        return si_mask, exist_SI

    def find_biggest_SI(self, mask_SI):

        mask_SI_rgb = cv2.cvtColor(np.uint8(mask_SI), cv2.COLOR_GRAY2BGR)
        
        # cv2.imwrite(os.path.join(self.path_save, 'mask_SI_rgb.png'), mask_SI_rgb)

        mask_SI = cv2.cvtColor(mask_SI_rgb, cv2.COLOR_BGR2GRAY)

        # cv2.imwrite(os.path.join(self.path_save, 'mask_SI_rgb.png'), mask_SI)

        # ret,thresh = cv2.threshold(mask_SI,127,255,0)
        contours,_ = cv2.findContours(mask_SI, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        cnt = max(contours, key=cv2.contourArea)

        mask_SI_rgb_copy = np.zeros_like(mask_SI)
        mask_SI_rgb_copy = cv2.cvtColor(np.uint8(mask_SI_rgb_copy), cv2.COLOR_GRAY2BGR)  # Convert to RGB
        img_with_biggest_SI = cv2.drawContours(mask_SI_rgb_copy, [cnt], 0, (255, 255, 255), 10)
        # cv2.imwrite('img_with_biggest_SI.png', img_with_biggest_SI)
        return img_with_biggest_SI, cnt

    def create_mask_SI_tumor_v1(self, segment_map, intended_tumor):
        '''
        Create a mask for surgical instrument and tumor.
        Find the tumor that overlaps with the intended tumor box and return the new bounding box coordinates.
        '''
        x1 = int(intended_tumor[0])
        x2 = int(intended_tumor[2])
        y1 = int(intended_tumor[1])
        y2 = int(intended_tumor[3])
        
        # Retrieve class indices for SI and Tumor
        class_number_SI = self.segmentor_obj.CLASSES.index("SI")
        class_number_Tumor = self.segmentor_obj.CLASSES.index("Tumor")

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

        # Iterate over all detected tumor contours
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            tumor_bbox = (x, y, x + w, y + h)
            # print("intended_bbox",intended_bbox)
            # print("tumor_bbox",tumor_bbox)
            overlap_area = self.bbox_overlap_area(intended_bbox, tumor_bbox)
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