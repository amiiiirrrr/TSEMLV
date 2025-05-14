
import numpy as np

def calculate_average_image_oneChannel_v1(img):
    # Calculate the sum of the pixel values for each color channel
    non_zero_pixels = np.nonzero(img)
    average = np.mean(img[non_zero_pixels])
    return average

def calculate_average_image_oneChannel_v2(img, mask=None):
    """Calculates the average of non-zero (or masked) pixels in a single channel image."""
    if img is None:
            print("Debug (_calc_avg): Input image is None.")
            return 0

    if mask is not None:
        if img.shape[:2] != mask.shape[:2]:
                print(f"Debug (_calc_avg): Image shape {img.shape[:2]} does not match mask shape {mask.shape[:2]}. Cannot apply mask.")
                pixels_to_average = img[img != 0]
        else:
                pixels_to_average = img[mask > 0]
    else:
        pixels_to_average = img[img != 0] # Average non-zero pixels

    if pixels_to_average.size == 0:
        # print("Debug (_calc_avg): No non-zero or masked pixels found to average.")
        return 0
    # Filter out non-finite values before averaging
    pixels_to_average = pixels_to_average[np.isfinite(pixels_to_average)]
    if pixels_to_average.size == 0:
        print("Debug (_calc_avg): No finite pixels found after filtering.")
        return 0

    return np.mean(pixels_to_average)

def calculate_average_image_RGB(img):
    # Calculate the sum of the pixel values for each color channel
    sum_blue = np.sum(img[:, :, 0])
    sum_green = np.sum(img[:, :, 1])
    sum_red = np.sum(img[:, :, 2])

    # Calculate the total number of pixels
    total_pixels = img.shape[0] * img.shape[1]

    # Calculate the average color
    average_blue = int(sum_blue / total_pixels)
    average_green = int(sum_green / total_pixels)
    average_red = int(sum_red / total_pixels)

    return (average_blue, average_green, average_red)