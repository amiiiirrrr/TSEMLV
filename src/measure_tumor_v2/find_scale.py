import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class ScaleDepth:
    def __init__(self, args):
        self.args = args

    def generate_axis_samples(self, P_c, v_L, T=80.0, N=100):
        """
        Generate N 3D points along the instrument axis.
        """
        ts = np.linspace(-T, T, N)
        return np.array([P_c + t * v_L for t in ts])
    
    def translate_axis2shaft(self, P_c, r=2.4):
        """
        Translate a point on the instrument axis to a corresponding point on the shaft axis, translated towards the camera/world origin.
        Returns point on the shaft axis P_s.
        P_c (3D point): point on instrument axis, where OP_c is perpendicular to v_L.
        r (float): surgical instrument radius (in mm).
        """
        v_OP_c = P_c
        v_OP_c_normalized = v_OP_c / np.linalg.norm(v_OP_c)

        P_s = P_c - r * v_OP_c_normalized
        return P_s

    def project_point(self, K, point3d):
        """
        Project a single 3D point into image coordinates using camera intrinsics.

        Args:
            K (np.ndarray): 3×3 camera intrinsic matrix.
            point3d (array-like): 3D point in camera coordinates [X, Y, Z].

        Returns:
            (u, v): Pixel coordinates in the image.
        """
        X, Y, Z = point3d
        if Z <= 0:
            raise ValueError("Point is behind the camera (Z <= 0).")

        # Convert to homogeneous image coordinates
        p_h = K @ np.array([X, Y, Z])
        u = p_h[0] / p_h[2]
        v = p_h[1] / p_h[2]
        return u, v

    def project_points(self, K, points_3d):
        """
        Project 3D points into the image using intrinsics K.
        Returns list of (u, v, Z_true).
        """
        projected = []
        for P in points_3d:
            p_h = K.dot(P)
            Z_true = p_h[2]
            if Z_true <= 0:
                continue  # behind camera
            u = p_h[0] / Z_true
            v = p_h[1] / Z_true
            projected.append((u, v, Z_true))
        return projected

    def filter_points_by_mask(self, projected_pts, mask, vis):
        """
        Keep only points whose projection falls inside the mask.
        Returns list of (u, v, Z_true).
        """
        H, W = mask.shape
        valid = []
        for u, v, Z in projected_pts:
            ui, vi = int(round(u)), int(round(v))
            cv2.circle(vis, (ui, vi), 3, (0, 0, 255), -1)
            if 0 <= ui < W and 0 <= vi < H and mask[vi, ui]:
                valid.append((u, v, Z))
        return valid, vis

    def sample_relative_depth(self, D_rel, valid_projected):
        """
        Sample the relative depth map at projected points.
        Returns list of (Z_rel, Z_true).
        """
        samples = []
        H, W = D_rel.shape
        # print('H, W', (H, W))
        for u, v, Z_true in valid_projected:
            ui, vi = int(round(u)), int(round(v))
            # print('ui, vi', (ui, vi))
            if 0 <= ui < W and 0 <= vi < H:
                Z_rel = D_rel[vi, ui]
                # print('Z_rel', Z_rel)
                if Z_rel > 0:
                    samples.append((Z_rel, Z_true))
        return samples

    def fit_scale_and_bias(self, samples):
        """
        Fit Z_true = s * Z_rel + b by least squares.
        Returns (s, b).
        """
        # print('samples', samples)
        Zrel = np.array([zr for zr, _ in samples])
        Ztrue = np.array([zt for _, zt in samples])
        # print('Zrel', Zrel)
        # print('Ztrue', Ztrue)
        
        zr_mean = Zrel.mean()
        zt_mean = Ztrue.mean()
        
        cov = ((Zrel - zr_mean) * (Ztrue - zt_mean)).sum()
        var = ((Zrel - zr_mean) ** 2).sum()
        
        s = cov / var
        b = zt_mean - s * zr_mean
        return s, b
    
    # def fit_inverse_model(self, samples):
    #     """
    #     Fit Z_true = a*(1/Z_rel) + b by least squares.
    #     samples: list of (Z_rel, Z_true) where Z_rel is disparity-like.
    #     """
    #     # Build X = 1/Z_rel, Y = Z_true
    #     Zrel = np.array([zr for zr, _ in samples])
    #     Ztrue = np.array([zt for _, zt in samples])
        
    #     X = 1.0 / Zrel
    #     Y = Ztrue
        
    #     # compute means
    #     Xm, Ym = X.mean(), Y.mean()
    #     # slope & intercept
    #     a = ((X - Xm) * (Y - Ym)).sum() / ((X - Xm)**2).sum()
    #     b = Ym - a * Xm
    #     return a, b

    def compute_absolute_depth_map(self, D_rel, s, b):
        """
        Convert a relative depth map to absolute using s and b.
        Returns absolute depth map.
        """
        return s * D_rel + b
    
    # def compute_absolute_depth_map_fromDisparity(self, D_rel, s, b):
    #     """
    #     Convert a relative depth map to absolute using s and b.
    #     Returns absolute depth map.
    #     """
    #     return s * (1/D_rel) + b

    def estimate_scale_and_depth_map(self, K, P_c, v_L, mask, D_rel, path_save, viz_copy, T=80.0, N=100):
        """
        End-to-end: estimate scale from axis, then compute absolute depth map.
        Also saves:
        - D_rel.png
        - projected_points_overlay.png
        - D_abs.png
        - D_abs_colorbar.png
        
        Returns:
        s, b, D_abs.
        """

        valid_points, viz_copy = self.find_valid_points(K, P_c, v_L, T, N, mask, viz_copy, path_save)
        # print('valid_points', valid_points)
        if len(valid_points)==0:
            valid_points, viz_copy = self.find_valid_points(K, -P_c, v_L, T, N, mask, viz_copy, path_save)
        # Sample depths and fit
        samples = self.sample_relative_depth(D_rel, valid_points)
        if len(samples)==0:
            print("No valid samples for scale estimation")
            return None, None, None, None
        
        s, b = self.fit_scale_and_bias(samples)
        # s, b = self.fit_inverse_model(samples)
        # s, b = self.fit_inverse_depth_model(samples)
        
        # Compute and save absolute depth
        D_abs = self.compute_absolute_depth_map(D_rel, s, b)
        # D_abs = self.compute_absolute_depth_map_fromDisparity(D_rel, s, b)
        # D_abs = self.predict_true_depth_inverse(D_rel, s, b)

        abs_vis = (D_abs / (D_abs.max() + 1e-8) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(path_save, 'D_abs.png'), abs_vis)
        
        # Colorbar
        vmin, vmax = D_abs.min(), D_abs.max()
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(D_abs, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.axis('off')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Depth (mm)')
        fig.savefig(os.path.join(path_save, 'D_abs_colorbar.png'), bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
        return s, b, D_abs, viz_copy
    
    def find_valid_points(self, K, P_c, v_L, T, N, mask, vis, path_save):
        
        # Sample & project
        pts_3d = self.generate_axis_samples(P_c, v_L, T, N)
        # print('pts_3d', pts_3d)
        proj = self.project_points(K, pts_3d)
        # print('proj', proj)
        
        # Overlay projected points on mask
        vis = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        valid, vis = self.filter_points_by_mask(proj, mask, vis)
        cv2.imwrite(os.path.join(path_save, 'projected_points_overlay.png'), vis)
        # print('valid', valid)

        return valid, vis



    # def fit_inverse_depth_model(self, samples):
    #     """
    #     Fit Z_true = 1/(a*Z_rel + b) via the linearization
    #     1/Z_true = a*Z_rel + b.

    #     Args:
    #         samples: list of (Z_rel, Z_true) pairs.

    #     Returns:
    #         a, b floats
    #     """
    #     Zrel = np.array([zr for zr, _ in samples], dtype=np.float64)
    #     Ztrue = np.array([zt for _, zt in samples], dtype=np.float64)
        
    #     # Discard any zeros for safety
    #     valid = Ztrue > 1e-8
    #     Zrel, Ztrue = Zrel[valid], Ztrue[valid]
        
    #     # Build design matrix and target
    #     X = Zrel.reshape(-1,1)          # shape (M,1)
    #     Y = (1.0 / Ztrue).reshape(-1,1) # shape (M,1)
        
    #     # Append ones for the intercept b
    #     A = np.hstack([X, np.ones_like(X)])  # shape (M,2)
        
    #     # Solve A [a; b] = Y in least squares
    #     ab, *_ = np.linalg.lstsq(A, Y, rcond=None)
    #     a, b = ab.flatten()
    #     return float(a), float(b)

    # def predict_true_depth_inverse(self, Z_rel_map, a, b):
    #     """
    #     Convert a relative-depth map to true depth via:
    #     Z_true = 1 / (a*Z_rel + b).

    #     Args:
    #         Z_rel_map: 2D array of relative-depth values.
    #         a, b: fitted parameters.

    #     Returns:
    #         2D array of true depth.
    #     """
    #     return 1.0 / (a * Z_rel_map + b)