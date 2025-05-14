import numpy as np
import cv2

class MeasureTumor:
    def __init__(self, args):
        self.args = args

    def pixel_to_3d(self, K_inv, u, v, Z):
        """
        Back-project a single pixel (u, v) with depth Z into 3D camera coordinates.
        
        Args:
            K_inv (np.ndarray): Inverse of the 3×3 camera intrinsic matrix.
            u (float): Pixel x-coordinate.
            v (float): Pixel y-coordinate.
            Z (float): Depth value at (u, v) in the absolute depth map (same units as K).
            
        Returns:
            np.ndarray: 3D point [X, Y, Z] in camera coordinates.
        """
        uv1 = np.array([u, v, 1.0])
        # Multiply by Z and K_inv
        return Z * (K_inv @ uv1)

    def measure_box_dimensions(self, K, D_abs, tumor_box):
        """
        Given a quadrilateral defined by 4 pixel corners, compute its real-world
        horizontal, vertical, and diagonal lengths.
        
        Args:
            K (np.ndarray): 3×3 camera intrinsic matrix.
            D_abs (np.ndarray): H×W absolute depth map.
            corners (list of tuple): Four (u, v) pixel coordinates in the order
                                    [top-left, top-right, bottom-right, bottom-left].
        
        Returns:
            dict: {
                'horizontal': float,
                'vertical': float,
                'diagonal1': float,  # TL → BR
                'diagonal2': float   # TR → BL
            }
        """
        x1, y1, x2, y2 = tumor_box
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        K_inv = np.linalg.inv(K)
        # Back-project each corner into 3D
        points_3d = []
        H, W = D_abs.shape
        for (u, v) in corners:
            ui, vi = int(round(u)), int(round(v))
            # Clamp to valid range
            ui = np.clip(ui, 0, W-1)
            vi = np.clip(vi, 0, H-1)
            Z = D_abs[vi, ui]
            points_3d.append(self.pixel_to_3d(K_inv, u, v, Z))
        
        # Unpack
        TL, TR, BR, BL = points_3d
        
        # Compute pairwise lengths
        h1 = np.linalg.norm(TR - TL)
        h2 = np.linalg.norm(BR - BL)
        v1 = np.linalg.norm(BL - TL)
        v2 = np.linalg.norm(BR - TR)
        d1 = np.linalg.norm(BR - TL)
        d2 = np.linalg.norm(BL - TR)

        # Average each category
        horizontal = (h1 + h2) / 2.0
        vertical   = (v1 + v2) / 2.0
        diagonal   = (d1 + d2) / 2.0
        
        return {
        'horizontal': horizontal,
        'vertical': vertical,
        'diagonal': diagonal
        }

# Example usage:
# K = np.array([[fx, 0, cx],
#               [0, fy, cy],
#               [0,  0,  1]])
# corners = [(u_tl, v_tl), (u_tr, v_tr), (u_br, v_br), (u_bl, v_bl)]
# dimensions = measure_box_dimensions(K, D_abs, corners)
# print(dimensions)