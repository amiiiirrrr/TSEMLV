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

    def filter_points_by_mask(self, projected_pts, mask):
        """
        Keep only points whose projection falls inside the mask.
        Returns list of (u, v, Z_true).
        """
        H, W = mask.shape
        valid = []
        for u, v, Z in projected_pts:
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < W and 0 <= vi < H and mask[vi, ui]:
                valid.append((u, v, Z))
        return valid

    def sample_relative_depth(self, D_rel, valid_projected):
        """
        Sample the relative depth map at projected points.
        Returns list of (Z_rel, Z_true).
        """
        samples = []
        H, W = D_rel.shape
        for u, v, Z_true in valid_projected:
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < W and 0 <= vi < H:
                Z_rel = D_rel[vi, ui]
                if Z_rel > 0:
                    samples.append((Z_rel, Z_true))
        return samples

    def fit_scale_and_bias(self, samples):
        """
        Fit Z_true = s * Z_rel + b by least squares.
        Returns (s, b).
        """
        Zrel = np.array([zr for zr, _ in samples])
        Ztrue = np.array([zt for _, zt in samples])
        
        zr_mean = Zrel.mean()
        zt_mean = Ztrue.mean()
        
        cov = ((Zrel - zr_mean) * (Ztrue - zt_mean)).sum()
        var = ((Zrel - zr_mean) ** 2).sum()
        
        s = cov / var
        b = zt_mean - s * zr_mean
        return s, b

    def compute_absolute_depth_map(self, D_rel, s, b):
        """
        Convert a relative depth map to absolute using s and b.
        Returns absolute depth map.
        """
        return s * D_rel + b

    def estimate_scale_and_depth_map(self, K, P_c, v_L, mask, D_rel, path_save, T=80.0, N=100):
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

        # Sample & project
        print('P_c', P_c)
        print('v_L', v_L)
        pts_3d = self.generate_axis_samples(P_c, v_L, T, N)
        print('pts_3d', pts_3d)
        proj = self.project_points(K, pts_3d)
        print('proj', proj)
        valid = self.filter_points_by_mask(proj, mask)
        print('valid', valid)
        
        # Overlay projected points on mask
        vis = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for u, v, _ in valid:
            ui, vi = int(round(u)), int(round(v))
            cv2.circle(vis, (ui, vi), 3, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(path_save, 'projected_points_overlay.png'), vis)
        
        # Sample depths and fit
        samples = self.sample_relative_depth(D_rel, valid)
        if not samples:
            raise ValueError("No valid samples for scale estimation")
        s, b = self.fit_scale_and_bias(samples)
        
        # Compute and save absolute depth
        D_abs = self.compute_absolute_depth_map(D_rel, s, b)
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
        
        return s, b, D_abs

