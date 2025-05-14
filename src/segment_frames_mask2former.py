# Copyright (c) OpenMMLab. All rights reserved.
import cv2
# import sys
# sys.path.append('mmsegmentation/')
from argparse import ArgumentParser
import numpy as np
from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model, show_result_pyplot

class SegmentMMSegmentation:
    def __init__(self, args):
        """
        Initialize AI modules
        """
        super(SegmentMMSegmentation, self).__init__()

        self.args = args

        self.initialize()
    
    def initialize(self):

        # build the model from a config file and a checkpoint file
        self.model = init_model(self.args.config, self.args.checkpoint, device=self.args.device_segmentation)
        self.PALETTE = PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], 
               [192, 128, 128], [0, 64, 0], [128, 64, 0], [128, 192, 0], [0, 64, 128], 
               [128, 64, 128], [0, 192, 128], [128, 192, 128], [64, 64, 0], [192, 64, 0], [64, 192, 0], 
               [192, 192, 0], [0, 192, 0],
                # [64, 64, 128], [192, 64, 128], [64, 192, 128], [192, 192, 128], 
               ]
        self.CLASSES = ["_background_", "GO","RL","RHD","GB","LL","FL","LHD","ST","DC","LPG",
                        "LPS","OV","BD","RPS","AC","RPG","UJ"
                        ,"SI","AP","CC","SC","TC","UT",
                        "SPL","SPR", "AW", "Tumor"
        ]
        
    def segment(self, img):

        result = inference_model(self.model, img)
        # show the results
        # blend raw image and prediction
        # draw_img = self.model.show_result(
        #     img,
        #     result,
        #     # palette=get_palette(args.palette),
        #     palette=self.PALETTE,
        #     show=False,
        #     opacity=self.args.opacity)
        result_array = np.array(result.pred_sem_seg.data.cpu())

        # result_array = result_array[0]

        # result_array = np.array(result_array)

        # out_file_ = 'output_image/'  + image

        draw_img = show_result_pyplot(
                self.model,
                img,
                result,
                title=self.args.title,
                opacity=self.args.opacity,
                with_labels=self.args.with_labels,
                draw_gt=False,
                show=False,
                out_file=self.args.out_file)

        return draw_img, result_array
        