
import numpy as np
import cv2

class CameraCallibration:
    def __init__(self, args):
        self.args = args

        self.K_matrix = np.array([[self.args.f_x, 0,  self.args.c_x],
                            [0,  self.args.f_y, self.args.c_y],
                            [0,  0,  1]], dtype=np.float64)

        # self.distortions = np.array(self.args.coeffdistortions)
        self.distortions = np.array([self.args.k1, self.args.k2, 0.0, 0.0, 0.0], dtype=float)

    def match_matrix(self, w, h):
        # Ensure K matrix matches image dimensions if cx, cy were calculated from center
        # This check assumes cx/cy *should* be image_width/2, image_height/2 if not explicitly set
        if np.isclose(self.K_matrix[0,2], w/2.0) and np.isclose(self.K_matrix[1,2], h/2.0):
             if not (np.isclose(self.K_matrix[0,2], w/2.0) and np.isclose(self.K_matrix[1,2], h/2.0)):
                 print(f"Debug: K matrix principal point ({self.K_matrix[0,2]:.1f}, {self.K_matrix[1,2]:.1f}) updated to image center ({w/2.0:.1f}, {h/2.0:.1f}) based on image size {w}x{h}.")
                 self.K_matrix[0,2] = w/2.0
                 self.K_matrix[1,2] = h/2.0
        elif not (np.isclose(self.K_matrix[0,2], w/2.0) and np.isclose(self.K_matrix[1,2], h/2.0)):
             print(f"Debug: Warning: K matrix principal point ({self.K_matrix[0,2]:.1f}, {self.K_matrix[1,2]:.1f}) does not match image center ({w/2.0:.1f}, {h/2.0:.1f}). Ensure K is correct for this image and resolution.")


    def undistort_image(self, img: np.ndarray,
                        alpha: float = 1.0
                    ) -> np.ndarray:

        dist = self.distortions.flatten()

        # check image shape
        if img.ndim == 2:
            # grayscale, OK
            pass
        elif img.ndim == 3 and img.shape[2] == 3:
            # BGR color, OK
            pass
        else:
            raise ValueError(f"Unsupported image shape {img.shape}. Expected HxW or HxWx3.")

        h, w = img.shape[:2]

        # compute optimal new camera matrix
        new_cam_mtx, roi = cv2.getOptimalNewCameraMatrix(
            self.K_matrix, dist, (w, h), alpha, (w, h)
        )

        # undistort
        dst = cv2.undistort(img, self.K_matrix, dist, None, new_cam_mtx)

        # crop to valid ROI
        x, y, w_roi, h_roi = roi
        dst = dst[y : y + h_roi, x : x + w_roi]

        # if return_extra:
        #     return dst, new_cam_mtx, roi
        return dst

        