
from .common import calculate_average_image_oneChannel_v1
import numpy as np
import cv2

class depthUtils:
    def __init__(self, args):
        self.args = args

    def normalize_depth_midas(self, depth_image):
        '''
        # https://github.com/isl-org/MiDaS/issues/4
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

    def idepth_to_depth_v2(self, idepth):
        """Converts inverse depth to depth (Z). Handles zero/inf values."""
        if idepth is None:
            print("Debug (_idepth_to_depth): Input inverse depth is None.")
            return None
        idepth = idepth.astype(np.float32)
        if np.any(idepth < -1e-9): # Check for unexpected negative values
                print(f"Debug (_idepth_to_depth): Warning: Found negative inverse depth values (min={np.min(idepth):.4f}). Clamping to zero.")
                idepth[idepth < 0] = 0.0

        depth = np.zeros_like(idepth, dtype=np.float32)
        non_zero_mask = idepth > 1e-9 # Avoid division by zero
        depth[non_zero_mask] = 1.0 / idepth[non_zero_mask]
        # Handle infinities that might arise from very small idepth values
        if np.any(np.isinf(depth)):
            print("Debug (_idepth_to_depth): Warning: Inf values created during 1/idepth. Replacing with 0.")
        depth[np.isinf(depth)] = 0 # Or a max depth value, or np.nan

        return depth

    def depth_tumor_SI_v1(self, path_save, mask_tumor, rect, idepth, depth_map_visualize):
        '''
        this function returns two value
        depth_value_SI: the depth value of the region that we find the biggest tumor
        depth_value_Tumor: the depth value of the region that we calculate the SI diameter based on its center in rect

        the depth value could be the average of depth values in that region.

        (center(x, y), (width, height), angle of rotation) = rect
        '''
        
        # cv2.imwrite(os.path.join(path_save, 'here2.png'), idepth)

        #################################### Depth mask Tumor ####################################
        # mask_tumor = cv2.cvtColor(np.uint8(mask_tumor), cv2.COLOR_GRAY2BGR)
        # mask_tumor = cv2.cvtColor(mask_tumor, cv2.COLOR_BGR2GRAY)
        mask_tumor = np.uint8(mask_tumor)
        # cv2.imwrite(os.path.join(path_save, 'here.png'), mask_tumor)

        # Apply the mask to the RGB image
        tumor_depth = cv2.bitwise_and(idepth, idepth, mask=mask_tumor)

        # cv2.imwrite(os.path.join(path_save, 'tumor_depth.png'), tumor_depth)

        avg_depth_tumor = calculate_average_image_oneChannel_v1(tumor_depth)

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
        # cv2.imwrite(os.path.join(path_save, 'SI_depth.png'), SI_img)

        # Copy the original image's square of size 10x10 centered into the new image
        SI_img[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1] = idepth[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1]
        # cv2.imwrite(os.path.join(path_save, 'here3.png'), SI_img)
        SI_img_visualize[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1] = depth_map_visualize[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1]
        # Save the result
        # cv2.imwrite(os.path.join(path_save, 'SI_depth.png'), SI_img_visualize)

        avg_depth_SI = calculate_average_image_oneChannel_v1(SI_img)

        return avg_depth_tumor, avg_depth_SI