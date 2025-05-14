import numpy as np
import cv2
import os

def create_segmentation_SI_Tumor_v1(img_original, visual, combined_mask):
    # Initialize a copy of the original image to modify
    merged_image = img_original.copy()

    darkening_factor = 0.5  # This reduces the brightness to 50%
    merged_image = (merged_image * darkening_factor).astype(img_original.dtype)
    # Create a boolean mask where the combined mask is 255 (white)
    mask_boolean = combined_mask == 255

    # Apply the mask to copy SI and Tumor segments from 'visual' to 'merged_image'
    merged_image[mask_boolean] = visual[mask_boolean]

    return merged_image

def visualize_function(path_save, image_visualize2, img_visualize, biggestSI_segment_map, rect_SI, real_W_obj, real_H_obj, Z_ref, Z_obj, largestTumor, mask_vis_tumor):
        
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
    
    #################### put text for tumor box #######################
    text_above_position = (x1_tumor - 100, y1_tumor - 10)
    text_below_position = (x1_tumor - 100, y1_tumor + 10) 
    text_belowbelow_position = (x1_tumor - 100, y1_tumor + 30) 
    img_visualize_2 = img_visualize_1.copy()
    # cv2.imwrite(os.path.join(path_save, 'visualize1.png'), img_visualize_1)

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
    cv2.imwrite(os.path.join(path_save, 'visualize2.png'), img_visualize_2)


    # Convert the original image to grayscale
    gray_image = cv2.cvtColor(img_visualize_1, cv2.COLOR_BGR2GRAY)
    # Apply lower intensity to create a dimmed effect
    dim_factor = 0.5  # Adjust this value for the desired dimming effect
    dimmed_image = (dim_factor * gray_image).astype(np.uint8)
    # Blend the dimmed image with the original image
    dimmed = cv2.addWeighted(img_visualize_1, 0.5, cv2.cvtColor(dimmed_image, cv2.COLOR_GRAY2BGR), 0.5, 0)
    # cv2.imwrite(os.path.join(path_save, 'dimmed.png'), dimmed)

    # Convert the color mask to grayscale
    # gray_mask = cv2.cvtColor(img_visualize_1, cv2.COLOR_BGR2GRAY)
    # Blend the images based on the mask
    brightness_factor = 0.5  # Adjust this value for the desired brightness effect
    blended_image = cv2.addWeighted(dimmed, 1, img_visualize_1, brightness_factor, 0)
    # cv2.imwrite(os.path.join(path_save, 'blended_image.png'), blended_image)

    # img_visualize_2 = img_visualize_1 + dimmed_img0s
    # cv2.imwrite(os.path.join(path_save, 'visualize2.png'), img_visualize_2)

def dimmed_mask(mask, original_image):
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

def write_depth_viz(depth_map, cmap='viridis'):
    """
    Creates a colormapped visualization of a depth map...
    """
    if depth_map is None:
        print("Debug (depth_viz): Input depth map is None.")
        return None

    viz_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)

    depth_min = viz_map.min()
    depth_max = viz_map.max()

    if depth_max - depth_min < np.finfo("float").eps:
        normalized_viz = np.zeros_like(viz_map, dtype=np.uint8)
    else:
        normalized_viz = 255 * (viz_map - depth_min) / (depth_max - depth_min)

    normalized_viz = normalized_viz.astype(np.uint8)

    mapper = plt.colormaps[cmap]
    # Use 'inferno' as a fallback matplotlib cmap if the requested one isn't found
    if not mapper: mapper = plt.colormaps['inferno']
    colored_viz = mapper(normalized_viz)
    colored_viz = (colored_viz[:, :, :3] * 255).astype(np.uint8)
    colored_viz = cv2.cvtColor(colored_viz, cv2.COLOR_RGB2BGR) # Convert to BGR for OpenCV
    return colored_viz

    # return cv2.applyColorMap(normalized_viz, cv2.COLORMAP_INFERNO) # Fallback cv2 colormap

