import os
import time
import yaml
import cv2 as cv
import numpy as np
import math
from io import BytesIO

from board import ChessBoard, RingBoard



class PatternInfo:
    #findChessboardCornersSB()
    corner_flags_cv4 = cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_EXHAUSTIVE + cv.CALIB_CB_ACCURACY
    #findChessboardCorners()
    corner_flags_cv2 = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE

    def __init__(self, cfg_path):
        if cfg_path is None:
            cfg_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
        cfg = yaml.safe_load(open(cfg_path, 'r'))

        base_cfg = cfg.get('base')
        self.is_using_cv4 = base_cfg.get('is_using_cv4', False)

        camera_cfg = cfg.get('camera')
        self.img_shape = camera_cfg.get('img_shape')

        pattern_cfg = cfg.get('pattern')
        self.is_ring = pattern_cfg.get('is_ring', False)
        self.board = ChessBoard(eval(pattern_cfg.get('pattern_shape')),
                                float(pattern_cfg.get('corner_distance')))

        self.refine = True

    @staticmethod
    def _angle(a, b, c):
        """
        a, b, c are three points.
        return angle between lines ab, bc.
        """
        ab = a - b
        ab = a - b
        cb = c - b
        angle = math.acos(np.dot(ab,cb) / (np.linalg.norm(ab) * np.linalg.norm(cb)))
        return angle

    @staticmethod
    def _pdist(p1, p2):
        """
        Distance bwt two points. p1 = (x, y), p2 = (x, y).
        """
        return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))

    def _get_vertex_corners(self, corners):
        pattern_shape = self.board.cb_shape
        xdim = pattern_shape[0]
        ydim = pattern_shape[1]
        if corners.shape[0]*corners.shape[1] != xdim*ydim:
            raise Exception("Invalid number of corners! %d corners. X: %d, Y: %d" 
                                % (corners.shape[1] * corners.shape[0],xdim, ydim))
        
        if self.is_using_cv4:
            up_left = corners[-xdim, 0]
            up_right = corners[0, 0]
            down_left = corners[-1, 0]
            down_right = corners[xdim-1, 0]
        else:
            up_left = corners[xdim-1, 0]
            up_right = corners[-1, 0]
            down_left = corners[0, 0]
            down_right = corners[-xdim,0]

        return (up_left, up_right, down_left, down_right)

    def _get_pattern_area(self, corners):
        """
        return pattern area
        refer to http://mathworld.wolfram.com/Quadrilateral.html
        """
        (up_left, up_right, down_left, down_right) = self._get_vertex_corners(corners)
        a = up_right - up_left
        b = down_right - up_right
        c = down_left - down_right
        p = b + c
        q = a + b
        return abs(p[0]*q[1] - p[1]*q[0]) / 2.

    def _get_pattrern_skew(self, corners):
        (up_left, up_right, _, down_right) = self._get_vertex_corners(corners)
        skew = min(1.0, 2. * abs((math.pi / 2.) - self._angle(up_left, up_right, down_right)))
        return skew
    
    def _get_pattern_rotate(self, corners):
        """
        return pattern rotation
        1.27323 is the scale changed from pi/4 to 1
        """
        (up_left, up_right, down_left, down_right) = self._get_vertex_corners(corners)
        center = np.sum((up_right, down_right, up_left, down_left), axis=0) / 4
        center_forward = center + [20, 0]
        rotate = min(1.0, 1.27323 * abs((math.pi / 4.) - self._angle(up_right, center, center_forward)))
        return rotate

    def _get_pattern_loc_lable(self, corners, img_shape, block_shape):
        """
        return pattern location lable in the image
        otherwise, return 0
        for example, when block_shape is (2,3): 
        #########################
        #   1   #   2   #   3   #
        #########################
        #   4   #   5   #   6   #
        #########################
        """
        if img_shape[0] < img_shape[1]:
            img_shape[0], img_shape[1] = img_shape[1], img_shape[0]

        center = np.sum(self._get_vertex_corners(corners), axis=0) / 4
        lable = int(center[1] / img_shape[1] * block_shape[0]) * block_shape[1]  + int(center[0] / img_shape[0] * block_shape[1] + 1)
        if lable < 1 or lable > (block_shape[0] * block_shape[1]):
            return 0
        return lable

    def _get_img_roi(self, img, corners):
        """
        return the croped img contain the pattern
        """
        mask = np.zeros(img.shape[:2], np.uint8)
        (up_left, up_right, down_left, down_right) = self._get_vertex_corners(corners)
        points = []
        points.append(up_left.astype(int))
        points.append(up_right.astype(int)) 
        points.append(down_right.astype(int)) 
        points.append(down_left.astype(int))
        counter = np.array(points)

        # Extend 10 pixels to the counter
        left =  np.min(counter[:, 0]) - 10 # +1
        right = np.max(counter[:, 0]) + 10 # -1
        top  =  np.min(counter[:, 1]) - 10 # +1
        bottom= np.max(counter[:, 1]) + 10 # -1

        _img_roi = img[top:bottom, left:right, :]
        return _img_roi

    def _get_pattren_sharpness(self, img, corners):
        """
        return the sharpness of the pattern area in the image 
        """
        img_roi = self._get_img_roi(img, corners)
        if len(img_roi.shape) == 3 and img_roi.shape[2] == 3:
            gray_roi = cv.cvtColor(img_roi, cv.COLOR_BGR2GRAY)
        else:
            gray_roi = img_roi
        sharpness = cv.Laplacian(gray_roi, cv.CV_64F).var()
        return sharpness

    def _get_corners(self, img):
        """
        Get corners for a particular pattern for an image
        """
        h = img.shape[0]
        w = img.shape[1]
        pattern_shape = self.board.cb_shape
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            gray = img
        if self.is_using_cv4:
            find, corners = cv.findChessboardCornersSB(gray, pattern_shape, flags=self.corner_flags_cv4)
        else:
            find, corners = cv.findChessboardCorners(gray, pattern_shape, flags=self.corner_flags_cv2)
        if not find:
            return find, corners
        
        # If any corners are within BORDER pixels of the screen edge, reject the detection by setting ok to false
        # NOTE: This may cause problems with very low-resolution cameras, where 8 pixels is a non-negligible fraction
        BORDER = 8
        if not all([(BORDER < corners[i, 0, 0] < (w - BORDER)) and (BORDER < corners[i, 0, 1] < (h - BORDER)) for i in range(corners.shape[0])]):
            find = False

        # TODO: how to make sure the pattern origin is the corners[0]?
        # make sure all corner arrays are going from top to bottom
        # when use findchessboardcorners(),there is no need to filp the corners
        if corners[0,0,1] > corners[-1,0,1] and self.is_using_cv4:
            corners = np.copy(np.flipud(corners))

        if self.refine and find:
            distance = float("inf")
            for row in range(pattern_shape[1] - 1):
                for col in range(pattern_shape[0] - 1):
                    index = row*pattern_shape[1] + col
                    distance = min(distance, self._pdist(corners[index, 0], corners[index + 1, 0]))
            for row in range(pattern_shape[1] - 2):
                for col in range(pattern_shape[0]):
                    index = row*pattern_shape[1] + col
                    distance = min(distance, self._pdist(corners[index, 0], corners[index + pattern_shape[0], 0]))
            radius = int(math.ceil(distance * 0.5))
            cv.cornerSubPix(gray, corners, (radius,radius), (-1,-1),
                                        ( cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.000001 ))
        return find, corners

    def get_pattern_info(self, img, block_shape = (2, 3), refine = True, usingCV4 = False):
        """
        return params of pattern corners
        params list [X, Y, skew, rotate, scale, sharpness, lable] of the pattern view
        X and Y described the pattern's location in the image
        scale: pattern'area / image'area
        skew describe the change in posture of the pattern
        """
        if img is None:
            raise Exception("no image found!")
        
        params = []
        find, corners = self._get_corners(img)

        if not find:
            return find, corners, params
        
        list_x = corners[:,:,0]
        list_y = corners[:,:,1]

        (width, height) = (img.shape[0], img.shape[1])

        area = self._get_pattern_area(corners)
        border = math.sqrt(area)

        # get params
        loc_X = min(1.0, max(0.0, (np.mean(list_x) - border / 2) / (width  - border)))
        loc_Y = min(1.0, max(0.0, (np.mean(list_y) - border / 2) / (height - border)))
        area_scale = math.sqrt(area / (width * height))
        skew = self._get_pattrern_skew(corners)
        rotate = self._get_pattern_rotate(corners)
        sharpness = self._get_pattren_sharpness(img, corners) / area_scale
        lable = self._get_pattern_loc_lable(corners, [width, height], block_shape)
        params = [loc_X, loc_Y, skew, rotate, area_scale, sharpness, lable]

        np.set_printoptions(precision=15)
        params = np.array(params)
        return find, corners, params

# # for test:
# if __name__ == '__main__':
#     from patternInfo import PatternInfo
#     img = cv.imread("./imgs/41.png",cv.IMREAD_COLOR)
#     # cv.imshow("pic", img)
#     # cv.waitKey(0)
#     pattern_info = PatternInfo(None)
#     find, corners, params = pattern_info.get_pattern_info(img, (2,3))

#     print (type(params))
#     print (find)
#     print (corners)
#     print (params)     