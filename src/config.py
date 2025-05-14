"""
config.py 
"""

import argparse

__authors__ = "Amir Mousavi"
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Amir Mousavi"
__email__ = "azmusavi19@gmail.com"
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
        
        self.parser.add_argument('--ground_truth_path', default='data/ground_truth/Tumor_Size_by_Experts_forCODE_miccai_workshop.xlsx', type=str)
        # self.parser.add_argument('--ground_truth_path', default='data/ground_truth/All sizes tumors_Biopsy study_REF.xlsx', type=str)
        self.parser.add_argument('--S_real_ref', type=int, default=4.8)
        self.parser.add_argument('--instrument_radius_mm', type=float, default=2.4, help='Known radius of the surgical instrument in mm (e.g., 5mm diameter -> 2.5mm radius)')
        self.parser.add_argument('--f_x', type=int, default=489)
        self.parser.add_argument('--f_y', type=int, default=529)
        self.parser.add_argument('--c_x', type=int, default=369)
        self.parser.add_argument('--c_y', type=int, default=277)
        self.parser.add_argument('--coeffdistortions', type=int, default=[[-0.01001264, -1.03390557, -0.00322796,  0.00245444,  1.57661359]])
        self.parser.add_argument('--k1', type=float, default=-0.01001264, help='1st radial distortion coefficient')
        self.parser.add_argument('--k2', type=float, default=-1.03390557, help='2nd radial distortion coefficient')
        self.parser.add_argument('--p1', type=float, default=-0.00322796, help='1st tangential distortion coefficient')
        self.parser.add_argument('--p2', type=float, default=0.00245444, help='2nd tangential distortion coefficient')
        self.parser.add_argument('--k3', type=float, default=1.57661359, help='3rd radial distortion coefficient')
        self.parser.add_argument('--depth_avg_area', type=int, default=1)

        # self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/MSDDK_P1_PCI29.mp4', type=str, help='')
        # self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/WVDRSNDN_P1_PCI12.mp4', type=str, help='')
        # self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/LDNS_P1_PCI7.mp4', type=str, help='')
        # self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/HPK_P1_PCI6.mp4', type=str, help='')
        # self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/EV_P0_PCI5.mp4', type=str, help='')
        # self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/BFRDSS_P1_PCI11.mp4', type=str, help='')

        # self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/MMNDBH_P1_PCI26.mp4', type=str, help='')
        # self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/AHRPLRT_P1_PCI9.mp4', type=str, help='')
        # self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/ANLLN_P1_PCI10.mp4', type=str, help='')
        # self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/AVDVNN_P5_PCI12.mp4', type=str, help='')
        # self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/AVLLBRGT_P1_PCI0.mp4', type=str, help='')
        # self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/AVRBG_P0_PCI0.mp4', type=str, help='')
        # self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/CZHNGH_P0_PCI11.mp4', type=str, help='')
        # self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/DPPP_P1_PCI14.mp4', type=str, help='')
        # self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/DVRMTN_P2_PCI6.mp4', type=str, help='')
        # self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/JDRDDR_P1_PCI0.mp4', type=str, help='')
        # self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/JPNKN_P2_PCI5.mp4', type=str, help='')
        # self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/LNWS_P1.mp4', type=str, help='')
        # self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/MGLS_P1_PCI5.mp4', type=str, help='')
        # self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/MH_P0_PCI7.mp4', type=str, help='')
        # self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/MPTRS_P4_PCI6.mp4', type=str, help='')
        # self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/SM_P0_PCI19.mp4', type=str, help='')

        # self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/GVLL_P0_PCI25.mp4', type=str, help='')
        # self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/MDGYTR_P1_PCI14.mp4', type=str, help='')
        # self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/RRB_P0_PCI0.mp4', type=str, help='')
        # self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/DNWNG_P1_PCI2.mp4', type=str, help='')
        # self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/CVCKR_P1_PCI11.mp4', type=str, help='')
        # self.parser.add_argument('--video_path', default='/media/data2/amir/PIPAC_DATA/videos_uploadedOnEncord_crf18or23/EWSSDRP_P1_PCI9.mp4', type=str, help='')
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
                        )
        
        self.parser.add_argument('--video_input', default=False, type=str, help='')
        self.parser.add_argument('--output_path', default='output_image/', type=str, help='')
        self.parser.add_argument('--model_weights',
                        default='models/DepthModels/MiDaS/weights/dpt_beit_large_512.pt',
                        )

        self.parser.add_argument('--optimize', dest='optimize', action='store_true', help='Use half-float optimization')
        self.parser.set_defaults(optimize=False)

        self.parser.add_argument('--side',
                        action='store_true',
                        )
        self.parser.add_argument('--height',
                        type=int, default=None,
                        )
        self.parser.add_argument('--square',
                        action='store_true',
                        )
        self.parser.add_argument('--grayscale',
                        action='store_true',
                        )
        #########################################################################################################
        ################### Yolo detection Config ###################################
        self.parser.add_argument('--weights', nargs='+', type=str, default='../../../detection/yolov7/pretrained/yolov7-e6e_training.pt', help='model.pt path(s)')
        self.parser.add_argument('--source', type=str, default='data/surgeon_images_miccai_workshop', help='source')  # file/folder, 0 for webcam
        # self.parser.add_argument('--source', type=str, default='data/surgeon_images_v2', help='source') 
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
        ################### Segmentation Config ##################################

        self.parser.add_argument('--config', default='models/SegmentationModels/mmsegmentation_mask2former/configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py')
        self.parser.add_argument('--checkpoint', default='models/SegmentationModels/mmsegmentation_mask2former/work_dirs/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640/best_mIoU_iter_28000.pth')
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