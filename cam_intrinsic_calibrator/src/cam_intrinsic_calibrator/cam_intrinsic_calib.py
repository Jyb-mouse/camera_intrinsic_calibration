import os
import sys
import yaml
import time
import cv_bridge
import numpy as np
import middleware as mw
import cv2 as cv

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped

from .calibrator import Calibrator
from .img_extracter import ImgExtracter

class CamInstrinsicCalib(ImgExtracter):
    def __init__(self, cfg_path):
        # if cfg_path is None:
        #     cfg_path = os.path.join(os.path.dirname(__file__),'../../config/config.yaml')
        
        super(CamInstrinsicCalib, self).__init__(cfg_path)

        # get params from config
        cfg = yaml.safe_load(open(os.path.join(cfg_path,'config.yaml'), 'r'))
        base_cfg = cfg.get('base')
        self.sum_images_need = base_cfg.get('sum_images_need')

        self.br = cv_bridge.CvBridge()
        self.calib_status = False

        self.intri_calibrator = Calibrator(cfg_path)

    def setup(self):
            # set publisher
        self.pub = mw.Publisher('/camera{}/intrinsic_calib/compressed'.format(
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
        img_pub = self.br.cv2_to_compressed_imgmsg(img_show,dst_format='jpeg')
        img_pub.header.stamp = cam_data.header.stamp
        self.pub.publish(img_pub)

    def _img_exracter_callback(self, data):
        tt = time.time()
        cam_data = data.get(self.img_topic)
        if cam_data is None:
            mw.logger.warn("no camera data input!")
            return
        if self.calib_status == False:

            img = self.br.compressed_imgmsg_to_cv2(cam_data)
            # if self.cam_id in [4, 17]:
            #     if self.is_cam390:
            #         img = cv.resize(img, (960, 540)) 
            #     else:
            #         img = cv.resize(img, (1024, 576))            
            #print("t1 = ", time.time() - tt)
            self.img_shape = (img.shape[1], img.shape[0])

            if img.shape[1] < img.shape[0]:
                self.img_shape = (img.shape[0], img.shape[1])
                img = np.rot90(img)

            img_show = np.zeros(img.shape, np.uint8)
            img_show = img.copy()

            (ret,
            self._corners_list,
            self._img_names) = super(CamInstrinsicCalib, self).img_extract_from_topic(img,
                                                                                    img_show,
                                                                                    self.img_shape)
            
            self._pub_img_show(img_show, cam_data)
        else:
            img_show = np.zeros((576, 1024, 3), np.uint8)
            img_show.fill(255)
            text = 'camera{} intrinsic calibration is done! Thank you!'.format(self.cam_id)
            cv.putText(img_show, text, (15, 280), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            self._pub_img_show(img_show, cam_data)
        

        
        if (len(self._corners_list) >= 35) and self.calib_status == False:
            self._corners_list = np.array(self._corners_list[1:]) # drop the first one
            self._img_names = np.array(self._img_names[1:])
            self.intri_calibrator.calibrate(self._corners_list, self._img_names, self.img_shape)
            self.calib_status = True

    def img_extract(self):
        if self.input_method == 'dataset':
            corners_list, img_names, img_shape = \
                super(CamInstrinsicCalib, self)._extract_img_from_ds()
            res = self.intri_calibrator.calibrate(corners_list, img_names, img_shape)
            print (res)
            sys.exit(1)

        elif self.input_method == 'topic':
            mw.init_node('cam_intrinsic_calibrator')
            self.setup()
            mw.start_heartbeat()
            mw.spin()
            mw.shutdown()
        else:
            mw.logger.warn("no input method!")


## 
# if __name__ ==  '__main__':
#     cam_intrinsic_calibrator = CamInstrinsicCalib(None)
#     cam_intrinsic_calibrator.img_extract()