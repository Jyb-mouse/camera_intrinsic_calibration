import os
import cv2 as cv
import numpy as np

from patternInfo import PatternInfo


class ImageDetector(object):
    def __init__(self, image_path, cfg_path):
        self.image_path = image_path
        self.cfg_path = cfg_path
        self.img_list = []
        self.img_corners_list = []
        self.img_name_list = []
        self.img_shape = None

        self.get_image_list(self.image_path, 'png')
        self.img_list.sort()
        if len(self.img_list) < 10:
            raise Exception("images not enough!")
        
        try:
            self.pattern_info = PatternInfo(self.cfg_path)
        except Exception as e:
            raise Exception(e)

    def get_image_list(self, dir_path, ext=None):
        newDir = dir_path
        if os.path.isfile(dir_path):
            if ext is None:
                self.img_list.append(dir_path)
            else:
                if ext in dir_path[-3:]:
                    self.img_list.append(dir_path)
        elif os.path.isdir(dir_path):
            for s in os.listdir(dir_path):
                newDir=os.path.join(dir_path,s)
                self.get_image_list(newDir, ext)
    
    def _draw_pattern_axis(self, find, corners, img, image_name):
        """
        draw the axis of pattern coordinate system on the image
        """
        shape = self.pattern_info.pattern_shape
        if find:
            corner = tuple(corners[0].ravel())
            img = cv.line(img,
                          corner,
                          tuple(corners[1].ravel()),
                          (255, 0, 0), # x axis is blue
                          4)
            img = cv.line(img,
                          corner,
                          tuple(corners[shape[0]].ravel()),
                          (0, 255, 0), # y axis is green
                          4)
            img = cv.circle(img,
                            corner,
                            4,
                            (0, 0, 255),
                            -1)
        cv.drawChessboardCorners(img, shape, corners, find)
        cv.imshow(image_name, cv.resize(img, (960, 510)))
        cv.waitKey(0)
        return img
    
    def detect(self):
        for image_name in self.img_list:
            img = cv.imread(image_name)
            img_shape = (img.shape[1], img.shape[0])
            img_show = np.zeros(img.shape, np.uint8)
            img_show = img.copy()
            find, corners, params = self.pattern_info.get_pattern_info(img)
            self._draw_pattern_axis(find, corners, img_show, image_name)
            if find:
                self.img_corners_list.append(corners)
                self.img_name_list.append(image_name)
            cv.destroyAllWindows()
        return np.array(self.img_corners_list), np.array(self.img_name_list), img_shape

# # for test:
# if __name__ == '__main__':
#     img_path = '/home/lingbo/Documents/intrinsic/to_ylb'
#     cfg_path = '/home/lingbo/calib_ws/cn-cam-intrinsic-calibrator/cam_intrinsic_calibrator/configs'
#     img_dector = ImageDetector(img_path, cfg_path)
#     img_dector.detect()
