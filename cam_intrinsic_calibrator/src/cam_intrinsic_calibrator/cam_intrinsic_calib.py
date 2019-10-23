import os
import sys
import yaml
import cv_bridge
import numpy as np
import middleware as mw

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped

from calibrator import Calibrator
from img_extracter import ImgExtracter

class CamInstrinsicCalib(ImgExtracter):
    def __init__(self, cfg_path):
        if cfg_path is None:
            cfg_path = os.path.join(os.path.dirname(__file__),'../../config/config.yaml')
        
        super(CamInstrinsicCalib, self).__init__(cfg_path)

        # get params from config
        cfg = yaml.safe_load(open(cfg_path, 'r'))
        base_cfg = cfg.get('base')
        self.sum_images_need = base_cfg.get('sum_images_need')


    
        self.img_extracter = ImgExtracter(None)
        self.intri_calibrator = Calibrator(None)

    def setup(self):
            # set publisher
        self.pub = mw.Publisher('/camera{}/img_extact/compressed'.format(
            self.cam_id),
            CompressedImage,
            monitored=False)
        # sub node
        self._setup_subscriber()

        # start
        mw.run_handler_async(self.inputs,
                             self._img_exracter_callback,
                             fetch_option=dict(max_age=0.03),
                             monitored=True)

    def _setup_subscriber(self):
        img_sub = mw.Subscriber(self.img_topic, 
                                CompressedImage,
                                monitored=False)
        
        sub_lst = [img_sub]

        self.inputs = mw.SmartMixer(*sub_lst)
    
    def _pub_img_show(self, img_show, cam_data):
        br = cv_bridge.CvBridge()
        img_pub = br.cv2_to_compressed_imgmsg(img_show,dst_format='jpeg')
        img_pub.header.stamp = cam_data.header.stamp
        self.pub.publish(img_pub)

    def _img_exracter_callback(self, data):
        img_data = data.get(self.img_topic)
        if cam_data is None:
            mw.logger.warn("no camera data input")
    
        img = cv.imdecode(np.fromstring(cam_data, dtype=np.uint8), cv.IMREAD_COLOR)
        self.img_shape = (img.shape[1], img.shape[0])

        img_show = np.zeros(img.shape, np.uint8)
        img_show = img.copy()
        input_kwargs = {
            'last_params': self._last_params,
            'last_corners': self._last_corners,
            'params_list': self._params_list,
            'corners_list': self._corners_list,
            'cur_img_block_row': self._cur_img_block_row,
            'cur_img_block_col': self._cur_img_block_col,
            'cur_img_block_count': self._img_block_count,
            'img_names': self._img_names
        }

        (self._last_params, self._last_corners, 
        self._params_list, self._corners_list,  
        self._cur_img_block_row, self._cur_img_block_col,
        self._img_block_count, self._img_names, img_show) = self.img_extracter.img_extract_from_topic(img,
                                                    img_show,
                                                    input_kwargs,
                                                    self.img_shape)
        
        self._pub_img_show(img_show)
        
        if (len(self._corners_list) >= self.sum_images_need):
            self._corners_list = np.array(self._corners_list)
            self._img_names = np.array(self._img_names)
            self.intri_calibrator.calibrate(self._corners_list, self._img_names, self.img_shape)

    def img_extract(self):
        if self.input_method == 'dataset':
            corners_list, img_names, img_shape = \
                self._extract_img_from_ds()
            res = self.intri_calibrator.calibrate(corners_list, img_names, img_shape)
            print (res)
            sys.exit()

        elif self.input_method == 'topic':
            self.setup()
            mw.start_heartbeat()
            mw.spin()
            mw.shutdown()
        else:
            mw.logger.warn("no input method!")


## 
if __name__ ==  '__main__':
    cam_intrinsic_calibrator = CamInstrinsicCalib(None)
    cam_intrinsic_calibrator.img_extract()