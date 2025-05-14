# Copyright (c) OpenMMLab. All rights reserved.
import cv2
# import sys
# sys.path.append('mmsegmentation/')
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette

class SegmentMMSegmentation:
    def __init__(self, args):
        """
        Initialize AI modules
        """
        super(SegmentMMSegmentation, self).__init__()

        self.args = args

        self.initialize()
    
    def initialize(self):

        # Initialize
        # build the model from a config file and a checkpoint file
        self.model = init_segmentor(self.args.config, self.args.checkpoint, device=self.args.device_segmentation)
        
        self.PALETTE = PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], 
               [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128], 
               [128, 64, 128], [0, 192, 128], [128, 192, 128], [64, 64, 0], [192, 64, 0], [64, 192, 0], 
               [192, 192, 0], [64, 64, 128], [192, 64, 128], [64, 192, 128], [192, 192, 128], 
               ]
        
        self.CLASSES = ["_background_", "GO","RL","RHD","GB","LL","LO","FL","LHD","ST","DC","LPG",
                        "LPS","OV","BD","RPS","AC","RPG","UJ","LJ",
                        "UI","LI","SI","AP","CC","SC","TC","UT",
                        "SPL","SPR", "AW","Tumor"
        ]

    def segment(self, img):

        result = inference_segmentor(self.model, img)
        # show the results
        # blend raw image and prediction
        draw_img = self.model.show_result(
            img,
            result,
            # palette=get_palette(args.palette),
            palette=self.PALETTE,
            show=False,
            opacity=self.args.opacity)

        return draw_img, result
        