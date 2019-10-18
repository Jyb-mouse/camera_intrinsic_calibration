import os
import yaml
import math
import shutil
import rosbag
import cv2 as cv
import numpy as np
import middleware as mw 
from dataset_store import Dataset
from queue import Queue

from patternInfo import PatternInfo
from board import ChessBoard

class ImgExtracter(object):
    camera_topic_390 = '/camera{}/image_color/compressed'
    camera_topic_pg = '/camera{}/image_color/compressed'

    def __init__(self, cfg_path):

        #init param from chonfig
        if cfg_path is None:
            cfg_path = os.path.join(os.path.dirname(__file__), '../../config/config.yaml')
        cfg = yaml.safe_load(open(cfg_path, 'r'))

        threshold_cfg = cfg.get('threshold')
        self.min_pin_difference = threshold_cfg.get('min_pin_difference', 0.1)
        self.min_area_scale = threshold_cfg.get('min_area_scale', 0.2)
        self.min_area_difference = threshold_cfg.get('min_area_difference', 0.005)
        self.min_rotation_difference = threshold_cfg.get('min_rotation_difference', 0.4)
        self.min_pattern_sharpness = threshold_cfg.get('min_pattern_sharpness', 100)
        self.max_pattern_moving_speed = threshold_cfg.get('max_pattern_moving_speed')
        
        base_cfg = cfg.get('base')
        self.is_using_cv4 = base_cfg.get('is_using_cv4', False)
        self.sum_images_need = base_cfg.get('sum_images_need', 120)
        self.img_block_shape = eval(base_cfg.get('img_block_shape'))

        self.data_cfg = cfg.get('data')
        self.input_method = self.data_cfg.get('input_method')
        self.dataset_name = self.data_cfg.get('dataset_name')
        self.using_rosbag = self.data_cfg.get('using_rosbag', False)
        self.rosbag_path = self.data_cfg.get('rosbag_name')
        self.dir_output = self.data_cfg.get('output_dir')

        camera_cfg = cfg.get('camera')
        self.is_cam390 = camera_cfg.get('is_cam390', False)
        self.cam_id = camera_cfg.get('id')
        self.img_shape = camera_cfg.get('img_shape')

        pattern_cfg = cfg.get('pattern')
        self.is_ring = pattern_cfg.get('is_ring', False)
        self.board = ChessBoard(eval(pattern_cfg.get('pattern_shape')),
                                float(pattern_cfg.get('corner_distance')))

        self.refine = True

        self._num_img = 0
        self.img_path = None
        self._bag_name = None
        self._bag_path = None
        self.img_format = 'png'
        
        if self.dir_output is None:
            print 'Please enter the path of output at config/data/output_dir'
            sys.exit()
        self.output_dir = None
        if self.input_method == 'rosbag':
            self._bag_name = os.path.basename(self.rosbag_path).split('.')[0]
        else:
            raise Exception('get no input method!')

        self._pattern_info = PatternInfo(cfg_path)

        self.output_dir = os.path.join(self.dir_output, self._bag_name, 'cam_{}'.format(self.cam_id))
        self.to_reset = self._clearup_folder(self.output_dir)
        self.img_path = os.path.join(self.output_dir, 'imgs/')

        self.img_topic = self.camera_topic_390.format(self.cam_id) if self.is_cam390 else self.camera_topic_pg.format(
                                                     self.cam_id)

        ## init param from mw
        #self.cam_id = mw.get_param('~cam_id', 1)

        self._extract_img_params_init()

    def _extract_img_params_init(self):
        self.block_sum = self._get_img_block_sum(self.img_block_shape)
        self.each_block_img_sum = int(self.sum_images_need / self.block_sum)

        # (1,2) to (2,2) to (2,3)
        self.img_block_row = 1.5
        self.img_block_col = 2.0
        self.imgs_block_count = np.zeros(int(self.img_block_row) * \
                                         int(self.img_block_col)).astype(int)

        # check path
        self.img_path = os.path.join(self.output_dir, 'imgs/')
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
        
        # new list 
        self._corners_list = []
        self._params_list = []
        self._img_names = []
        self._last_corners = []
        self._last_params = []

    @staticmethod
    def _clearup_folder(output_dir):
        to_reset = True
        if output_dir is not None:
            if os.path.exists(output_dir):
                if len(os.listdir(output_dir)) > 0:
                    print 'The folder {} is NOT empty\n' \
                          'There is going to delete the data'.format(output_dir)
                    for p in os.listdir(output_dir):
                        path = os.path.join(output_dir, p)
                        if os.path.isdir(path):
                            shutil.rmtree(path)
                        elif os.path.isfile(path):
                            os.remove(path)
                        else:
                            print 'Unknown type: {}'.format(path)
                # else:
                #     to_reset = False
            else:
                os.makedirs(output_dir)
        return to_reset
    
    def parse_imgs(self, max_num_imgs):
        input_method = self.input_method
        rosbag_path = self.rosbag_name
        self.img_path = os.path.join(self.dir_output, 'imgs/')

        if not self.to_reset:
            self._num_img = len(os.listdir(self.img_path))
            return self.fetch_imgs()
        else:
            if self.input_method == 'rosbag':
                self._parse_imgs_from_rosbag(rosbag_path, self.cam_id)
                return self.fetch_imgs()
            else:
                raise Exception('get no input method!')
    
    def fetch_imgs(self):
        imgs_file_names = sorted(os.listdir(os.path.join(self.img_path)))
        for im_name in imgs_file_names:
            yield im_name, cv.imread(os.path.join(self.img_path, im_name))


    def _parse_imgs_from_rosbag(self, bag_path, cam_id):
        print '\tDump imgs...'
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
        num_img_count = 0
        bag = rosbag.Bag(bag_path)
        for topic, msg, ts in bag.read_messages(topics=[self.img_topic]):
            img = cv.imdecode(np.fromstring(msg.data, dtype=np.uint8), cv.IMREAD_COLOR)
            cv.imwrite(os.path.join(self.img_path, '{}.{}'.format(ts, self.img_format)), img)

            num_img_count += 1
            print (num_img_count)
        bag.close()
        self._num_img = num_img_count


    def _judge_little_moving(self, corners, last_corners):
        """
        Returns true if the motion of the pattern is sufficiently low between
        this and the previous frame.
        """
        if len(last_corners) == 0:
            return False
        corners_difference = (corners - last_corners).reshape(corners.shape[0] * corners.shape[1], 2)
        diff_average = np.average(np.linalg.norm(corners_difference, axis=1))
        return diff_average <= self.max_pattern_moving_speed
    
    def _judge_nice_pattern_view(self, params, corners, last_params, last_corners, img_block_sum, saved_params_list):
        """
        return True if the pattern described by params is nice
        """
        out_str = " "
        ret = True
        if not saved_params_list:
            return ret, out_str
        def param_distance(p1, p2):
            return sum([abs(a-b) for (a,b) in zip(p1, p2)])
        
        # TODO: What's a good threshold here? Should it be configurable?
        if len(last_params) != 0:
            # check rotate change
            r = max([param_distance(params[2:4], last_params[2:4])])
            # check translation change
            t = min([param_distance(params[0:2], p[0:2]) for p in saved_params_list])
            if t <= self.min_pin_difference:
                out_str = "Please move the pattren by panning!"
                if r <= self.min_rotation_difference:
                    out_str = "Please move the pattren by rotating!"
                    ret = False
                    return ret, out_str

        # check sharpness
        if params[5] < self.min_pattern_sharpness:
            out_str = "Please increase the sharpness!"
            ret = False
            return ret, out_str

        ## check pattern moving speed
        # if self.max_pattern_moving_speed > 0:
        #     if not self._judge_little_moving(corners, last_corners):
        #         return False

        # check pattern area proportion
        if (params[4] * img_block_sum) < self.min_area_scale or (params[4] > (1. / img_block_sum * 1.4)):
            out_str = "Please change the distance from pattern to camera!"
            ret = False
            return ret, out_str
        # all check passed,return True
        out_str = "This is a nice pattern view!"
        return ret, out_str
    
    @staticmethod
    def _get_block_vertices(lable, img_block_shape, img_shape):
        """
        return two vertices of input image block
        """
        pt0 = np.ones(2)
        pt1 = np.ones(2)
        if lable > img_block_shape[1]:
            pt1[1] = img_shape[1]
            pt1[0] = int(img_shape[0] / img_block_shape[1] * (lable - img_block_shape[1]))
        else:
            pt1[1] = int(img_shape[1] / img_block_shape[0])
            pt1[0] = int(img_shape[0] / img_block_shape[1] * lable)
        pt0[0] = pt1[0] - int(img_shape[0] / img_block_shape[1])
        pt0[1] = pt1[1] - int(img_shape[1] / img_block_shape[0])
        return pt0.astype(int), pt1.astype(int)
    
    def _get_img_block_sum(self, img_block_shape):
        """
        return image block sum
        """
        sum = 0
        row = img_block_shape[0]
        col = img_block_shape[1]

        while row >= 1 and col >= 2:
            sum = sum + row*col
            if (row < col):
                col -= 1
            else:
                row -= 1
        return sum

    def _draw_pattern_axis(self, corners, img):
        """
        draw the axis of pattern coordinate system on the image
        """
        corner = tuple(corners[0].ravel())
        img = cv.line(img, 
                      corner,
                      tuple(corners[1].ravel()),
                      (255,0,0), # x axis is blue
                      4) 
        img = cv.line(img,
                      corner,
                      tuple(corners[self.board.cb_shape[0]].ravel()),
                      (0,255,0), # y axis is green
                      4) 
        img = cv.circle(img,
                        corner,
                        4,
                        (0,0,255),
                        -1)
        return img

    
    def extract_img_from_rosbag(self):
        print '\tDump imgs from rosbag...'

        bag = rosbag.Bag(self.rosbag_path)
        for topic, msg, ts in bag.read_messages(topics=[self.img_topic]):
            img = cv.imdecode(np.fromstring(msg.data, 
                                            dtype=np.uint8),
                                            cv.IMREAD_COLOR)
            self.img_shape = (img.shape[1], img.shape[0])
            img_show = np.zeros(img.shape, np.uint8)
            img_show = img.copy()
            block_img_fill_success = True
            if int(self.img_block_row) <= self.img_block_shape[0]:
                find, corners, params = self._pattern_info.get_pattern_info(img, 
                                             (int(self.img_block_row), int(self.img_block_col)))
                if not find:
                    print 'no corners! skipped'
                else:
                    ret, out_str = self._judge_nice_pattern_view(params, corners, self._last_params, self._last_corners, 
                                         int(self.img_block_row) * int(self.img_block_col), self._params_list)
                    print (out_str)
                    img_show = self._draw_pattern_axis(corners, img_show)
                    if ret:
                        for i in xrange(len(self.imgs_block_count)):
                            if params[6] == i + 1 and self.imgs_block_count[i] < self.each_block_img_sum:
                                print (params)
                                self._params_list.append(params)
                                self._corners_list.append(corners)
                                self._last_corners = corners
                                self._last_params = params
                                self.imgs_block_count[i] += 1
                                img_name = str('{}.{}'.format(ts, self.img_format))
                                self._img_names.append(img_name)
                                cv.imwrite(os.path.join(self.img_path, img_name), img)
                                print 'image saved!'
                # draw on the img_show
                for i in xrange(len(self.imgs_block_count)):
                    pt0, pt1 = self._get_block_vertices(i+1, (int(self.img_block_row),int(self.img_block_col)), self.img_shape)
                    if self.imgs_block_count[i] == self.each_block_img_sum:
                        cv.rectangle(img_show, tuple(pt0), tuple(pt1), (0,0,255), 2)  
                        cv.line(img_show, tuple(pt0), tuple(pt1), (0,0,255), 2)
                        cv.line(img_show, (pt0[0], pt0[1]+int(self.img_shape[1]/int(self.img_block_row))), 
                                            (pt1[0], pt1[1]-int(self.img_shape[1]/int(self.img_block_row))), (0,0,255), 2)     
                    else: 
                        text = "img_num: {}/{}".format(int(self.imgs_block_count[i]), self.each_block_img_sum)
                        cv.putText(img_show, text, (pt0[0]+15, pt0[1]+30), cv.FONT_HERSHEY_PLAIN, 1.4, (0,255,0), 2)
                        block_img_fill_success = False
            if block_img_fill_success:
                self.img_block_col += 0.5
                self.img_block_row += 0.5
                self.imgs_block_count = np.zeros(int(self.img_block_row) * int(self.img_block_col)).astype(int)            

            if len(self._corners_list) == self.each_block_img_sum * self.block_sum:
                print '------images extraction done!------'
                break
            cv.imshow("img", img_show)
            cv.waitKey(10)
        corners_list = np.array(self._corners_list)
        img_names = np.array(self._img_names)
        return corners_list, img_names, self.img_shape
    
    def img_extract_from_topic(self, data):
        print '\tDump imgs from camera topic...'

        cam_data = data.get(self.img_topic)
        if cam_data is None:
            mw.logger.warn("no camera data input")

        img = cv.imdecode(np.fromstring(cam_data, dtype=np.uint8), cv.IMREAD_COLOR)
        self.img_shape = (img.shape[1], img.shape[0])
        img_show = np.zeros(img.shape, np.uint8)
        img_show = img.copy()

        block_img_fill_success = True
        if int(self.img_block_row) <= self.img_block_shape[0]:
            find, corners, params = self._pattern_info.get_pattern_info(img, 
                                         (int(self.img_block_row), int(self.img_block_col)))
            if not find:
                print 'no corners! skipped'
            else:
                ret, out_str = self._judge_nice_pattern_view(params, corners, self._last_params, self._last_corners, 
                                     int(self.img_block_row) * int(self.img_block_col), self._params_list)
                print (out_str)
                img_show = self._draw_pattern_axis(corners, img_show)
                if ret:
                    for i in xrange(len(self.imgs_block_count)):
                        if params[6] == i + 1 and self.imgs_block_count[i] < self.each_block_img_sum:
                            print (params)
                            self._params_list.append(params)
                            self._corners_list.append(corners)
                            self._last_corners = corners
                            self._last_params = params
                            self.imgs_block_count[i] += 1
                            img_name = str('{}.{}'.format(ts, self.img_format))
                            self._img_names.append(img_name)
                            cv.imwrite(os.path.join(self.img_path, img_name), img)
                            print 'image saved!'
            # draw on the img_show
            for i in xrange(len(self.imgs_block_count)):
                pt0, pt1 = self._get_block_vertices(i+1, (int(self.img_block_row),int(self.img_block_col)), img_shape)
                if self.imgs_block_count[i] == self.each_block_img_sum:
                    cv.rectangle(img_show, tuple(pt0), tuple(pt1), (0,0,255), 2)  
                    cv.line(img_show, tuple(pt0), tuple(pt1), (0,0,255), 2)
                    cv.line(img_show, (pt0[0], pt0[1]+int(img_shape[1]/int(self.img_block_row))), 
                                        (pt1[0], pt1[1]-int(img_shape[1]/int(self.img_block_row))), (0,0,255), 2)     
                else: 
                    text = "img_num: {}/{}".format(int(self.imgs_block_count[i]), self.each_block_img_sum)
                    cv.putText(img_show, text, (pt0[0]+15, pt0[1]+30), cv.FONT_HERSHEY_PLAIN, 1.4, (0,255,0), 2)
                    block_img_fill_success = False
        if block_img_fill_success:
            self.img_block_col += 0.5
            self.img_block_row += 0.5
            self.imgs_block_count = np.zeros(int(self.img_block_row) * int(self.img_block_col)).astype(int)            
        if len(self._corners_list) == self.each_block_img_sum * self.block_sum:
            print '------images extraction done!------'
            # TODO how to make the img extract done
            mw.shutdown()
            
        cv.imshow("img", img_show)
        cv.waitKey(10)


# # for test:
# if __name__ == '__main__':
#     from img_extracter import ImgExtracter
#     imgextracter = ImgExtracter(None)
#     corners_list, img_names, img_shape = imgextracter.img_extract()