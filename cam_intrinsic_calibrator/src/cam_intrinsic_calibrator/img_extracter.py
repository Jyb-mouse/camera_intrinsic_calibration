import os
import yaml
import math
import copy
import sys
import time
import shutil
import cv2 as cv
import numpy as np
import middleware as mw
from PIL import Image, ImageDraw, ImageFont
from dataset_store import Dataset


from .patternInfo import PatternInfo
from .board import ChessBoard
from .util import time2secs

class ImgExtracter(object):
    camera_topic_390 = '/camera{}/image_color/compressed'
    camera_topic_pg = '/camera{}/image_color/compressed'

    def __init__(self, cfg_path):

        ## init param from mw
        self.cam_id = mw.get_param('~cam_id', 1)
        self.is_cam390 = mw.get_param('~is_cam390', False)
        self.vehicle_name = mw.get_param('~vehicle_name')

        #init param from config
        cfg = yaml.safe_load(open(os.path.join(cfg_path,'config.yaml'), 'r'))
        
        threshold_cfg = cfg.get('threshold')
        self.min_pin_difference = threshold_cfg.get('min_pin_difference', 0.1)
        self.min_area_scale = threshold_cfg.get('min_area_scale', 0.2)
        self.min_area_difference = threshold_cfg.get('min_area_difference', 0.005)
        self.min_rotation_difference = threshold_cfg.get('min_rotation_difference', 0.4)
        self.min_pattern_sharpness = threshold_cfg.get('min_pattern_sharpness', 100)
        self.max_pattern_moving_speed = threshold_cfg.get('max_pattern_moving_speed')
        self.max_skew_limit = threshold_cfg.get('max_skew_limit')
        
        base_cfg = cfg.get('base')
        self.is_using_cv4 = base_cfg.get('is_using_cv4', False)
        self.sum_images_need = base_cfg.get('sum_images_need', 120)
        self.img_block_shape = eval(base_cfg.get('img_block_shape'))

        data_cfg = cfg.get('data')
        self.input_method = data_cfg.get('input_method')
        self.dataset_name = data_cfg.get('dataset_name')
        self.dir_output = data_cfg.get('output_dir')

        camera_cfg = cfg.get('camera')
        if self.input_method == 'dataset':
            self.is_cam390 = camera_cfg.get('is_cam390', False)
            self.cam_id = camera_cfg.get('id')

        pattern_cfg = cfg.get('pattern')
        self.is_ring = pattern_cfg.get('is_ring', False)
        self.board = ChessBoard(eval(pattern_cfg.get('pattern_shape')),
                                float(pattern_cfg.get('corner_distance')))

        self.refine = True
        self.filled_str = "This image block has been filled!"

        self._num_img = 0
        self.img_path = None
        self._bag_name = None
        self._bag_path = None
        self.img_format = 'png'
        
        if self.dir_output is None:
            print ('Please enter the path of output at config/data/output_dir')
            sys.exit(0)
        self.output_dir = None
        if self.input_method == 'dataset':
            self._bag_name = self.dataset_name
        elif self.input_method == 'topic':
            self._bag_name = self.vehicle_name
        else:
            raise Exception('get no input method!')

        self._pattern_info = PatternInfo(cfg_path)
        self.font_path = os.environ["TSPKG_PREFIX_PATH"] + '/share/cam_intrinsic_calibrator/font/font.ttf'
        self.output_dir = os.path.join(self.dir_output, self._bag_name, 'cam_{}'.format(self.cam_id))
        self.to_reset = self._clearup_folder(self.output_dir)
        self.img_path = os.path.join(self.output_dir, 'imgs/')

        self.img_topic = self.camera_topic_390.format(self.cam_id) \
                        if self.is_cam390 else self.camera_topic_pg.format(self.cam_id)

        # set constant
        self.TEXT_CHINESE_OUTPUT = {
            'TEXT_ONE' : '此标定板图像符合要求,已保存!',
            'TEXT_TWO' : '请让图像中标定板更清晰!',
            'TEXT_THREE' : '请让标定板在图像面积与当前图像区域大小基本相同!',
            'TEXT_FOUR' : '请减小标定板偏斜角度!',
            'TEXT_FIVE' : '请平移标定板!',
            'TEXT_SIX' : '请从多个方向旋转标定板!',
            'TEXT_SEVEN' : '未检测到完整标定板角点!',
            'TEXT_EIGHT' : '此图像区域已经采集完毕!'
        }
        self.TEXT_ENGLISH_OUTPUT = {
            'TEXT_ONE' : 'This is a nice pattern view! saved!',
            'TEXT_TWO' : 'Please make the pattern clearer!',
            'TEXT_THREE' : 'Please change the distance from pattern to camera!',
            'TEXT_FOUR' : 'The skew is too large!',
            'TEXT_FIVE' : 'Please move the pattren by panning!',
            'TEXT_SIX' : 'Please move the pattren by rotating!',
            'TEXT_SEVEN' : 'no corners!skipped',
            'TEXT_EIGHT' : 'This image block has been filled!'
        }

        self._extract_img_params_init()

    def _extract_img_params_init(self):
        self.block_sum = self._get_img_block_sum(self.img_block_shape)
        self.each_block_img_sum = int(self.sum_images_need / self.block_sum)

        # check path
        self.img_path = os.path.join(self.output_dir, 'imgs/')
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
        
        # set global params
        self._last_params = []
        self._last_corners = []
        self._params_list = []
        self._corners_list = []
        self._cur_img_block_row = 1.5 #  1 to 2 to 2
        self._cur_img_block_col = 2.0 #  2 to 2 to 3
        self._img_names = []
        self._img_block_count = np.zeros(int(self._cur_img_block_row) * int(self._cur_img_block_col))
        self.img_shape = np.zeros(2)
        self._corners_list_kawrgs = {i + 1 : list() for i in range(self.img_block_shape[0]*self.img_block_shape[1])}
        self._params_list_kawrgs = {i + 1 : list() for i in range(self.img_block_shape[0]*self.img_block_shape[1])}

    @staticmethod
    def _clearup_folder(output_dir):
        to_reset = True
        if output_dir is not None:
            if os.path.exists(output_dir):
                if len(os.listdir(output_dir)) > 0:
                    print('The folder {} is NOT empty\n' 
                          'There is going to delete the data'.format(output_dir))
                    for p in os.listdir(output_dir):
                        path = os.path.join(output_dir, p)
                        if os.path.isdir(path):
                            shutil.rmtree(path)
                        elif os.path.isfile(path):
                            os.remove(path)
                        else:
                            print ('Unknown type: {}'.format(path))
                    print('data deleted!')
                # else:
                #     to_reset = False
            else:
                os.makedirs(output_dir)
        return to_reset

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
        out_chinese_str = self.TEXT_CHINESE_OUTPUT['TEXT_ONE']
        out_english_str = self.TEXT_ENGLISH_OUTPUT['TEXT_ONE']
        ret = True
        def param_distance(p1, p2):
            return sum([abs(a-b) for (a,b) in zip(p1, p2)])

        # check sharpness
        if params[5] < self.min_pattern_sharpness:
            out_chinese_str = self.TEXT_CHINESE_OUTPUT['TEXT_TWO']
            out_english_str = self.TEXT_ENGLISH_OUTPUT['TEXT_TWO']
            ret = False
            return ret, out_chinese_str, out_english_str

        ## check pattern moving speed
        # if self.max_pattern_moving_speed > 0:
        #     if not self._judge_little_moving(corners, last_corners):
        #         return False

        # check pattern area proportion,*1.4 for hasing got the lager area of the pattern
        if (params[4] * img_block_sum) < self.min_area_scale or params[4] > ((1. / img_block_sum) * 1.4):
            out_chinese_str = self.TEXT_CHINESE_OUTPUT['TEXT_THREE']
            out_english_str = self.TEXT_ENGLISH_OUTPUT['TEXT_THREE']
            ret = False
            return ret, out_chinese_str, out_english_str

        # check skew
        if params[2] >= self.max_skew_limit:
            out_chinese_str = self.TEXT_CHINESE_OUTPUT['TEXT_FOUR']
            out_english_str = self.TEXT_ENGLISH_OUTPUT['TEXT_FOUR']
            ret = False
            return ret, out_chinese_str, out_english_str
    
        if not saved_params_list:
            return ret, out_chinese_str, out_english_str
    
        # TODO: What's a good threshold here? Should it be configurable?
        if len(last_params) != 0:
            #r = max([param_distance(params[2:4], last_params[2:4])])
            r = max([param_distance(params[2:4], p[2:4]) for p in saved_params_list])
            t = min([param_distance(params[0:2], last_params[0:2])])
            # check translation change
            if t <= self.min_pin_difference:
                out_chinese_str = self.TEXT_CHINESE_OUTPUT['TEXT_FIVE']
                out_english_str = self.TEXT_ENGLISH_OUTPUT['TEXT_FIVE']
                ret = False
                # check rotate change
                if r <= self.min_rotation_difference:
                    out_chinese_str = self.TEXT_CHINESE_OUTPUT['TEXT_SIX']
                    out_english_str = self.TEXT_ENGLISH_OUTPUT['TEXT_SIX']
                    ret = False
                    return ret, out_chinese_str, out_english_str
                else:
                    # rotation check passed
                    ret = True

        # all check passed,return True
        return ret, out_chinese_str, out_english_str
    
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
            if row < col:
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
                      (255, 0, 0), # x axis is blue
                      4)
        img = cv.line(img,
                      corner,
                      tuple(corners[self.board.cb_shape[0]].ravel()),
                      (0, 255, 0), # y axis is green
                      4)
        img = cv.circle(img,
                        corner,
                        4,
                        (0, 0, 255),
                        -1)
        return img

    @staticmethod
    def _draw_satisfied_img_block(img, img_shape, img_block_row, pt0, pt1, put_str):
        p0 = tuple(pt0)
        p1 = tuple(pt1)
        cv.rectangle(img, p0, p1, (0, 0, 255), 2)
        cv.line(img, p0, p1, (0, 0, 255), 2)
        cv.line(img,
                (pt0[0], pt0[1]+int(img_shape[1]/int(img_block_row))),
                (pt1[0], pt1[1]-int(img_shape[1]/int(img_block_row))),
                (0, 0, 255),
                2)
        cv.putText(img,
                   put_str,
                   (pt0[0]+20, pt0[1]+20),
                   cv.FONT_HERSHEY_PLAIN,
                   1,
                   (0, 255, 0),
                   2)
        return img
    
    def cv_img_add_text(self, img, text, left, top, textColor=(255, 0, 0), textSize=20):
        if (isinstance(img, np.ndarray)):  # opencv type
            img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        fontText = ImageFont.truetype(
            self.font_path, textSize, encoding="utf-8")
        draw.text((left, top), text, textColor, font=fontText)
        return cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)

    def img_extract_from_topic(self, img, img_show, img_shape):
        t1 = time.time()
        mw.logger.info('\tDump imgs from camera topic...')
        block_row = int(self._cur_img_block_row)
        block_col = int(self._cur_img_block_col)

        block_img_fill_success = True
        ret = False
        find, corners, params = self._pattern_info.get_pattern_info(img, (block_row, block_col))
        #print("t2 =", time.time() - t1)
        if not find:
            out_chinese_str = self.TEXT_CHINESE_OUTPUT['TEXT_SEVEN']
            out_english_str = self.TEXT_ENGLISH_OUTPUT['TEXT_SEVEN']
        else:
            ret, out_chinese_str, out_english_str = self._judge_nice_pattern_view(params,
                                                        corners,
                                                        self._last_params,
                                                        self._last_corners,
                                                        block_row * block_col,
                                                        self._params_list_kawrgs[params[6]])
            img_show = self._draw_pattern_axis(corners, img_show)
            #print("t3 = ", time.time() - t1)
            if ret:
                for i in range(len(self._img_block_count)):
                    if params[6] == i + 1 and self._img_block_count[i] < self.each_block_img_sum:
                        self._params_list_kawrgs[params[6]].append(params)
                        self._corners_list.append(corners)
                        self._last_corners = corners
                        self._img_block_count[i] += 1
                        ts = time.time()
                        img_name = str('{}.{}'.format(ts, self.img_format))
                        self._img_names.append(img_name)
                        print("Got one frame : ", params)
                        ret = True
                        cv.imwrite(os.path.join(self.img_path, img_name), img)
                    elif params[6] == i + 1 and self._img_block_count[i] == self.each_block_img_sum:
                        out_chinese_str = self.TEXT_CHINESE_OUTPUT['TEXT_EIGHT']
                        out_english_str = self.TEXT_ENGLISH_OUTPUT['TEXT_EIGHT']
            self._last_params = params
            #print("t4 = ", time.time() - t1)
        # draw on the img_show
        for i in range(len(self._img_block_count)):
            pt0, pt1 = self._get_block_vertices(i+1, (block_row, block_col), img_shape)
            if self._img_block_count[i] == self.each_block_img_sum:
                img_show = self._draw_satisfied_img_block(img_show, img_shape, block_row, pt0, pt1, self.filled_str)
            else: 
                text = "img_num: {}/{}".format(int(self._img_block_count[i]), self.each_block_img_sum)
                cv.putText(img_show, text, (pt0[0]+15, pt0[1]+30), cv.FONT_HERSHEY_PLAIN, 1.4, (0, 255, 0), 2)
                block_img_fill_success = False
        cv.putText(img_show, out_english_str, (20, img_shape[1]-20), cv.FONT_HERSHEY_PLAIN, 1.8, (0, 0, 255), 2)
        if not isinstance(out_chinese_str, unicode):
            out_chinese_str = out_chinese_str.decode('utf-8')
        img_pub = self.cv_img_add_text(img_show, out_chinese_str, 20, img_shape[1] - 110, (255, 0, 0), 40)

        # judge img blocks filled each time
        #print("t5 = ", time.time() - t1)
        if block_img_fill_success:
            self._cur_img_block_row += 0.5
            self._cur_img_block_col += 0.5
            self._img_block_count = np.zeros(int(self._cur_img_block_row) * int(self._cur_img_block_col)).astype(int)
            self._params_list_kawrgs = {i + 1 : list() for i in range(self.img_block_shape[0]*self.img_block_shape[1])}
        return  (ret, img_pub, self._corners_list, self._img_names)

    def _extract_img_from_ds(self):
        print('\tDump imgs from camera dataset...')
        if self._bag_path is not None:
            ds = Dataset(self._bag_path)
        else:
            ds = Dataset.open(self._bag_name)

        print('\tFinding best time segments for calibration...')
        corner_flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE
        res_list = self._find_calibrate_time(ds, self.img_topic, self.board.cb_shape, corner_flags)

        print('\tEffective time segments chosen:')
        for rl in res_list:
            print ([str(time2secs(l, ds.meta['ts_begin'])) + 's' for l in rl])

        print('\tDump imgs... ')
        sample_rate = self._get_sample_rate(ds, res_list, self.img_topic, 2000)
        for res in res_list:
            for ts, data in ds.fetch(self.img_topic, ts_begin=res[0], ts_end=res[1]):
                if np.random.uniform(0, 1) > sample_rate:
                    img = cv.imdecode(np.fromstring(data.data, np.uint8), cv.IMREAD_COLOR)
                    self.img_shape = (img.shape[1], img.shape[0])
                    img_show = np.zeros(img.shape, np.uint8)
                    img_show = img.copy()
                    block_row = int(self._cur_img_block_row)
                    block_col = int(self._cur_img_block_col)
                    block_img_fill_success = True
                    if block_row <= self.img_block_shape[0]:
                        find, corners, params = self._pattern_info.get_pattern_info(img,
                                                     (block_row, block_col))
                        if not find:
                            out_chinese_str = self.TEXT_CHINESE_OUTPUT['TEXT_SEVEN']
                            out_english_str = self.TEXT_ENGLISH_OUTPUT['TEXT_SEVEN']
                        else:
                            ret, out_chinese_str, out_english_str = self._judge_nice_pattern_view(params, corners, self._last_params, self._last_corners,
                                                 block_row * block_col, self._params_list_kawrgs[params[6]])
                            img_show = self._draw_pattern_axis(corners, img_show)
                            if ret:
                                for i in range(len(self._img_block_count)):
                                    if params[6] == i + 1 and self._img_block_count[i] < self.each_block_img_sum:
                                        self._params_list_kawrgs[params[6]].append(params)
                                        self._corners_list.append(corners)
                                        self._last_corners = corners
                                        self._img_block_count[i] += 1
                                        img_name = str('{}.{}'.format(ts, self.img_format))
                                        self._img_names.append(img_name)
                                        print("Got one frame : ", params)
                                        cv.imwrite(os.path.join(self.img_path, img_name), img)
                                    elif params[6] == i + 1 and self._img_block_count[i] == self.each_block_img_sum:
                                        out_chinese_str = self.TEXT_CHINESE_OUTPUT['TEXT_EIGHT']
                                        out_english_str = self.TEXT_ENGLISH_OUTPUT['TEXT_EIGHT']
                            self._last_params = params                            
                        # draw on the img_show
                        for i in range(len(self._img_block_count)):
                            pt0, pt1 = self._get_block_vertices(i+1, 
                                                                (block_row, block_col), 
                                                                self.img_shape)
                            if self._img_block_count[i] == self.each_block_img_sum:
                                img_show = self._draw_satisfied_img_block(img_show, self.img_shape, block_row, pt0, pt1)     
                            else: 
                                text = "img_num: {}/{}".format(int(self._img_block_count[i]), self.each_block_img_sum)
                                cv.putText(img_show, text, (pt0[0]+15, pt0[1]+30), cv.FONT_HERSHEY_PLAIN, 1.4, (0,255,0), 2)
                                block_img_fill_success = False
                        cv.putText(img_show, out_english_str, (20, img_shape[1]-20), cv.FONT_HERSHEY_PLAIN, 1.8, (0, 0, 255), 2)
                        if not isinstance(out_chinese_str, unicode):
                            out_chinese_str = out_chinese_str.decode('utf-8')
                        img_pub = self.cv_img_add_text(img_show, out_chinese_str, 20, img_shape[1] - 110, (255, 0, 0), 40)
                    if block_img_fill_success:
                        self._cur_img_block_row += 0.5
                        self._cur_img_block_col += 0.5
                        self._img_block_count = np.zeros(int(self._cur_img_block_row) * int(self._cur_img_block_col)).astype(int)
                        self._params_list_kawrgs = {i + 1 : list() for i in range(self.img_block_shape[0]*self.img_block_shape[1])}            
            
                    if len(self._corners_list) == self.each_block_img_sum * self.block_sum:
                        print('------images extraction done!------')
                        ret = True
                        break
                corners_list = np.array(self._corners_list)
                img_names = np.array(self._img_names)
        return corners_list, img_names, self.img_shape

    @staticmethod
    def _get_sample_rate(ds, ts_lst, topic_name, max_num):
        topic_freq = int(np.round(ds.topics[topic_name].meta['stat']['fps']))
        total_time = 0
        for interval in ts_lst:
            ts_start, ts_end = interval
            total_time += ((ts_end - ts_start) / 1e9)
        sample_rate = (1 - max_num / float(topic_freq * total_time))
        return sample_rate
    
    @staticmethod
    def _find_calibrate_time(ds, cam_topic, checkerboard, corner_flags, time_step=0.2):
        """
        input: ds - dataset instance
               cam_topic - camera topic
        output: valid calibration time range, a list of [ts_begin, ts_end]
        """
        ts_begin, ts_end = ds.meta['ts_begin'], ds.meta['ts_end']
        time_step = time_step * 60.0 * 1e9  # convert from minutes to milliseconds
        ts_list = np.arange(ts_begin, ts_end, time_step)
        res_list, cur = [], []

        # make chessboard scan quickly
        img_default_shape = (1024, 576)

        for t in ts_list:
            for ts, data in ds.fetch_near(cam_topic, t, limit=1):
                im = cv.imdecode(np.fromstring(data.data, np.uint8), cv.IMREAD_COLOR)
                gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
                # make chessboard scan quickly
                if gray.shape != img_default_shape:
                    gray = cv.resize(gray, img_default_shape)
                found, _c = cv.findChessboardCorners(gray, checkerboard, flags=corner_flags)
                if found:
                    if len(cur) < 2:
                        cur.append(ts)
                    else:
                        cur[1] = ts
                else:
                    if len(cur) == 2:
                        res_list.append(copy.copy(cur))
                    cur = []
        # for the end
        if len(cur) == 2:
            res_list.append(copy.copy(cur))
        return res_list



# # for test:
# if __name__ == '__main__':
#     from img_extracter import ImgExtracter
#     imgextracter = ImgExtracter(None)
#     corners_list, img_names, img_shape = imgextracter.img_extract()