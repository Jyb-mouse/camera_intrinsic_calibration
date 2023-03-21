import os
import sys

from image_detector import ImageDetector
from calibrator import Calibrator

class Interface(object):
    def __init__(self, img_path):
        self.img_path = img_path
        self.config_path = os.path.join(os.getcwd().replace('\\','/'), 'cam_intrinsic_calibrator', 'configs')

        self.img_detector = ImageDetector(self.img_path, self.config_path)
        self.calibrator = Calibrator(self.config_path)

    def process(self):
        corners, img_names, img_shape = self.img_detector.detect()

        self.calibrator.calibrate(corners, img_names, img_shape)

# for test:
if __name__ == '__main__':
    img_path = '/home/lingbo/Documents/intrinsic/new_gs4/senyun_x8b_2'
    interface = Interface(img_path) 
    interface.process()
 
  



        