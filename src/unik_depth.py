from unik3d.models import UniK3D
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class UnikDepther:
    def __init__(self, args):
        self.args = args

        # Move to CUDA, if any
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UniK3D.from_pretrained("lpiccinelli/unik3d-vitl") # vitl for ViT-L backbone
        self.model = model.to(device)
    
    def unik_depther(self, image_path):

        # Load the RGB image and the normalization will be taken care of by the model
        rgb = torch.from_numpy(np.array(Image.open(image_path))).permute(2, 0, 1) # C, H, W

        predictions = self.model.infer(rgb)

        # # Point Cloud in Camera Coordinate
        # xyz = predictions["points"]

        # # Unprojected rays
        # rays = predictions["rays"]

        # Metric Depth Estimation
        depth = predictions["depth"]

        return depth

    def vis_unik_depther(self, depth, save_path):
        # Get depth as numpy array: [1, 1, H, W] → [H, W]
        depth_np = depth.squeeze().cpu().numpy()

        # Normalize depth: closer = 255, farther = 0
        depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
        depth_norm = 1.0 - depth_norm  # reverse it: closer = white

        # Save normalized depth as .npy
        np.save(f"{save_path}/predicted_depth_unik3d_normalized.npy", depth_norm)

        # Apply a matplotlib colormap (e.g., 'plasma', 'jet', 'inferno')
        cmap = plt.get_cmap('plasma')  # or 'jet', 'magma', etc.
        depth_colored = cmap(depth_norm)[:, :, :3]  # Drop alpha channel → shape [H, W, 3]

        # Convert to uint8 RGB and save
        depth_rgb_uint8 = (depth_colored * 255).astype(np.uint8)
        Image.fromarray(depth_rgb_uint8).save(save_path + "/predicted_depth_unik3d_colored.png")