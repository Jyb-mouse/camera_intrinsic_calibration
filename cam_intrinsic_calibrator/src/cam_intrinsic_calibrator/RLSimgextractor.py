import os
import math

import yaml
import numpy as np

import middleware as mw 

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped

from img_extracter import ImgExtracter

class RLSImgExtractor(ImgExtracter):

    def __init__(self, cfg):
        super(RLSImgExtractor, self).__init__(cfg)
        
        ## init params from mw
        #self.cam_id = mw.get_param('~cam_id', 1)

    def setup(self):

        # sub node
        self._setup_subscriber()

        # start
        mw.run_handler_async(self.inputs,
                             ImgExtracter.img_extract_from_topic,
                             fetch_option=dict(max_age=0.03),
                             monitored=True)

    def _setup_subscriber(self):
        img_sub = mw.Subscriber(self.img_topic, 
                                CompressedImage,
                                monitored=False)
        
        sub_lst = [img_sub]

        self.inputs = mw.SmartMixer(*sub_lst)

    def img_extract(self):
        if self.using_rosbag:
            corners_list, img_names, img_shape = \
                super(RLSImgExtractor, self).extract_img_from_rosbag()
        else:
            self.setup()
            mw.start_heartbeat()
            mw.spin()
            mw.shutdown()
            corners_list = np.array(self._corners_list)
            img_names = np.array(self._img_names)
            img_shape = self.img_shape
        return corners_list, img_names, img_shape