"""
config.py 
"""

import argparse

__authors__ = "Amir Mousavi"
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Amir Mousavi"
__email__ = "seyedamir.mousavi@ghent.ac.kr"
__status__ = "Production"


class Config:
    """
    This class set static paths and other configs.
    Args:
    argparse :
    The keys that users assign such as sentence, tagging_model and other statictics paths.
    Returns:
    The configuration dict specify text, statics paths and controller flags.
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.run()

    def run(self):
        '''
        run method to start definition of configurations
        '''
        
        self.parser.add_argument('--ground_truth_path', default='ground_truth/Tumor_Size_by_Experts_forCODE_miccai_workshop.xlsx', type=str)
        # self.parser.add_argument('--ground_truth_path', default='ground_truth/All sizes tumors_Biopsy study_REF.xlsx', type=str)
        self.parser.add_argument('--S_real_ref', type=int, default=4.8)
        self.parser.add_argument('--f_x', type=int, default=489)
        self.parser.add_argument('--f_y', type=int, default=529)
        self.parser.add_argument('--c_x', type=int, default=369)
        self.parser.add_argument('--c_y', type=int, default=277)
        self.parser.add_argument('--depth_avg_area', type=int, default=1)

        self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/MSDDK_P1_PCI29.mp4', type=str, help='')


        # self.parser.add_argument('--video_path', default='/media/data2/amir/Biopsy_Study/dataset_v1/videos/AFDL_P1.mp4', type=str, help='')

        self.parser.add_argument('--output_file', default='output_video/', type=str, help='')
        self.parser.add_argument('--pause_duration', default=1, type=int, help='')

        self.parser.add_argument('--use_kalman_bbox', default=True, type=str, help='')
        self.parser.add_argument('--mae_method', default=True, type=str, help='')
        self.parser.add_argument('--average_method', default=False, type=str, help='')

        ################ Depth Estimation Config ###################################
        self.parser.add_argument('--model_type',
                        default='dpt_beit_large_512',
                        help='Model type: '
                             'dpt_beit_large_512, dpt_beit_large_384, dpt_beit_base_384, dpt_swin2_large_384, '
                             'dpt_swin2_base_384, dpt_swin2_tiny_256, dpt_swin_large_384, dpt_next_vit_large_384, '
                             'dpt_levit_224, dpt_large_384, dpt_hybrid_384, midas_v21_384, midas_v21_small_256 or '
                             'openvino_midas_v21_small_256'
                        )
        
        self.parser.add_argument('--video_input', default=False, type=str, help='')
        self.parser.add_argument('--output_path', default='output_image/', type=str, help='')
        self.parser.add_argument('--model_weights',
                        default='MiDaS/weights/dpt_beit_large_512.pt',
                        help='Path to the trained weights of model'
                        )

        self.parser.add_argument('--optimize', dest='optimize', action='store_true', help='Use half-float optimization')
        self.parser.set_defaults(optimize=False)

        self.parser.add_argument('--side',
                        action='store_true',
                        help='Output images contain RGB and depth images side by side'
                        )

        self.parser.add_argument('--height',
                        type=int, default=None,
                        help='Preferred height of images feed into the encoder during inference. Note that the '
                             'preferred height may differ from the actual height, because an alignment to multiples of '
                             '32 takes place. Many models support only the height chosen during training, which is '
                             'used automatically if this parameter is not set.'
                        )
        self.parser.add_argument('--square',
                        action='store_true',
                        help='Option to resize images to a square resolution by changing their widths when images are '
                             'fed into the encoder during inference. If this parameter is not set, the aspect ratio of '
                             'images is tried to be preserved if supported by the model.'
                        )
        self.parser.add_argument('--grayscale',
                        action='store_true',
                        help='Use a grayscale colormap instead of the inferno one. Although the inferno colormap, '
                             'which is used by default, is better for visibility, it does not allow storing 16-bit '
                             'depth values in PNGs but only 8-bit ones due to the precision limitation of this '
                             'colormap.'
                        )
        #########################################################################################################
        ################### Yolo detection Config ###################################
        self.parser.add_argument('--weights', nargs='+', type=str, default='yolov7/runs/train/yolov72/weights/best.pt', help='model.pt path(s)')
        self.parser.add_argument('--source', type=str, default='surgeon_images_miccai_workshop', help='source')  # file/folder, 0 for webcam
        # self.parser.add_argument('--source', type=str, default='surgeon_images_v2', help='source') 
        self.parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        self.parser.add_argument('--conf-thres', type=float, default=0.19, help='object confidence threshold')
        self.parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        self.parser.add_argument('--device-det', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        self.parser.add_argument('--view-img', action='store_true', help='display results')
        self.parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        self.parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        self.parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        self.parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        self.parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        self.parser.add_argument('--augment', action='store_true', help='augmented inference')
        self.parser.add_argument('--update', action='store_true', help='update all models')
        self.parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        self.parser.add_argument('--name', default='exp', help='save results to project/name')
        self.parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        self.parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

        self.parser.add_argument('--surgeons_evaluation', default=True, type=str, help='')
        #########################################################################################################
        ################### Segmentation Config ###################################
        # self.parser.add_argument('img', help='Image file')
        # self.parser.add_argument('--config', default='mmsegmentation/configs/segformer/segformer_mit-b5_8x1_1024x1024_160k_pipac.py')
        # self.parser.add_argument('--checkpoint', default='mmsegmentation/work_dirs/iter_103000.pth')
        # self.parser.add_argument(
        #     '--device-segmentation', default='cuda:0', help='Device used for inference')
        # self.parser.add_argument(
        # '--opacity',
        # type=float,
        # default=0.5,
        # help='Opacity of painted segmentation map. In (0, 1] range.')

        self.parser.add_argument('--config', default='mmsegmentation_mask2former/configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py')
        self.parser.add_argument('--checkpoint', default='mmsegmentation_mask2former/work_dirs/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640/best_mIoU_iter_28000.pth')
        self.parser.add_argument(
        '--device-segmentation', default='cuda:0', help='Device used for inference')
        self.parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
        self.parser.add_argument(
        '--with-labels',
        action='store_true',
        default=False,
        help='Whether to display the class labels.')
        self.parser.add_argument(
            '--title', default='result', help='The image identifier.')
        self.parser.add_argument('--out-file', default=None, help='Path to output file')

    def get_args(self):
        '''
        get_args method to return defined configurations
        '''
        return self.parser.parse_args()