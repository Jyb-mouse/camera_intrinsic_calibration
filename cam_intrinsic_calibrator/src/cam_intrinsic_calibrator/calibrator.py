import os
import yaml
import cv2 as cv
import shutil
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import cpu_count
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial

from .util import save_params, outliers_iqr, outliers_norm_std
from .img_extracter import ImgExtracter

from .detector_util import *

class Calibrator(ImgExtracter):
    CV_TERM_CRITERIAS = (cv.TERM_CRITERIA_MAX_ITER, cv.TERM_CRITERIA_EPS)

    cali_flags = cv.CALIB_USE_INTRINSIC_GUESS + cv.CALIB_ZERO_TANGENT_DIST + \
                 cv.CALIB_FIX_K3 + cv.CALIB_FIX_K4 
    calib_crit = [100, 1e-5]
    criteria=(cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 0.000001)

    min_num_img_todo_sampling = 200

    def __init__(self, cfg_path):
        # if cfg_path is None:
        #     cfg_path = os.path.join(os.path.dirname(__file__), '../../config/config.yaml')
        #     cfg_path = 
        cfg = yaml.safe_load(open(os.path.join(cfg_path,'config.yaml'), 'r'))
        
        super(Calibrator ,self).__init__(cfg_path)
        self.cali_crit = self._build_term_crit(self.calib_crit)

        threshold_cfg = cfg.get('threshold')
        self.batch_size = threshold_cfg.get('batch_size')
        self.reg = float(threshold_cfg.get('lambda'))

        base_cfg = cfg.get('base')
        self.max_iter = base_cfg.get('max_iter')
        self.is_using_OR_calibrate = base_cfg.get('is_using_OR_calibrate')

        self.RLS_cfg = cfg.get('RLS')

        debug_cfg = cfg.get('debug')
        self.viz = debug_cfg.get('viz')
        self.save = debug_cfg.get('save')
        self.verbose = debug_cfg.get('verbose')

        camera_cfg = cfg.get('camera')
        self.output_img_shape = camera_cfg.get('output_img_shape')
        self.flip_input_img = camera_cfg.get('flip_input_img')
        self.flip_output_img = camera_cfg.get('flip_output_img')
        if self.is_cam390:
            self.output_img_shape = [960, 540]
        else:
            self.output_img_shape = [1024,576]
        if self.cam_id in [6, 7]:
            self.flip_input_img = True
            self.flip_output_img = True
        if self.cam_id in [1, 14, 15]:
            camera_cfg_name = "4mm_lens_config_390.yaml" if self.is_cam390 else \
                               "4mm_lens_config.yaml"
        elif self.cam_id in [3, 6, 7, 12, 13]:
            camera_cfg_name = "12mm_lens_config_390.yaml" if self.is_cam390 else \
                               "12mm_lens_config.yaml"
        elif self.cam_id in [4, 17]:
            camera_cfg_name = "25mm_lens_config_390.yaml" if self.is_cam390 else \
                               "25mm_lens_config.yaml"
        cam_cfg_path = os.path.join(os.path.dirname(cfg_path), camera_cfg_name)
        self.cam_config = yaml.safe_load(open(cam_cfg_path, 'r'))

        data_cfg = cfg.get('data')
        self.num_thread = data_cfg.get('num_thread')

        pattern_cfg = cfg.get('pattern')
        self.is_ring = pattern_cfg.get('is_ring', False)
        self.pattern_shape = eval(pattern_cfg.get('pattern_shape'))
        self.corner_distance = pattern_cfg.get('corner_distance')

        self.data_dir = self.output_dir
        self.progress_output_dir = os.path.join(self.data_dir, 'progress')
        if not os.path.exists(self.progress_output_dir):
            os.makedirs(self.progress_output_dir)


    @staticmethod
    def _build_term_crit(choices):
        _crits = 0
        _choices = []
        for crit, choice in zip(Calibrator.CV_TERM_CRITERIAS, choices):
            if choice is not None and choice >= 0:
                _crits += crit
                _choices.append(choice)
        return (_crits,) + tuple(_choices)

    @staticmethod
    def _compute_init_guess_intrinsic(camera_cfg, img_shape, img_flipped=False):
        """
            function to find the init guess of the intrinsic
            input:
                camera_cfg: configuration from yaml file
                img_shape: [H,W](flip because opencv)
            output:
                init_intrinsic: factory standard parameters
        """
        optical_focal_length = camera_cfg['focal_length']
        pixel_size = camera_cfg['pixel_size']
        film_shape = np.array(camera_cfg['film_shape'])
        if img_flipped:
            film_shape = film_shape[::-1]

        ratio = max(img_shape / film_shape.astype(float))

        focal = optical_focal_length / pixel_size * ratio
        cx, cy = np.array(img_shape) / 2.0

        params = np.array([focal, focal, cx, cy])
        mat = np.array([[focal, 0, cx],
                        [0, focal, cy],
                        [0, 0, 1]])

        return params, mat
    
    @staticmethod
    def cal_v(i, j, H):
        v = np.array([
            H[0, i] * H[0, j],
            H[0, i] * H[1, j] + H[1, i] * H[0, j],
            H[1, i] * H[1, j],
            H[2, i] * H[0, j] + H[0, i] * H[2, j],
            H[2, i] * H[1, j] + H[1, i] * H[2, j],
            H[2, i] * H[2, j]
        ])
        return v

    @staticmethod
    def _select_candidate_by_reprojection_err(k_list, d_list, ret_list, r_list, t_list,
                                             id_list, corners, verbose, candidate_num=5):

        # create top CANDIDATE_NUM reprojection error list out of the source,
        # which is candidate_perc * batch_num, e.g. 0.1 x 200 = 20
        k_list, d_list, ret_list = np.array(k_list), np.array(d_list), np.array(ret_list)
        r_list, t_list, id_list = np.array(r_list), np.array(t_list), np.array(id_list)
        candidate_num = min(candidate_num, len(ret_list))
        candidate_indices = ret_list.argsort()[:candidate_num]

        # apply the intrinsic results to all the batches and compute the mean of the reprojection error list
        # compare and give the optimal result - smallest error
        reproj_list = []
        for ci in candidate_indices:
            tmp_list = []
            for i in range(r_list.shape[0]):
                tmp, tot_pts = 0, 0
                for j in range(r_list.shape[1]):
                    proj_pts, _ = cv.projectPoints(corners[0][id_list[i, j]], r_list[i, j], t_list[i, j],
                                                   k_list[ci], d_list[ci])
                    proj_pts = proj_pts[:, 0, :]
                    tmp += np.sum(np.abs(proj_pts - corners[1][id_list[i, j], :, 0, :]) ** 2)
                    tot_pts += len(corners[1][id_list[i, j], :, 0, :])
                tmp_list.append(np.sqrt(tmp * 1. / tot_pts))
            reproj_list.append(tmp_list)

        # compute new matrix
        reproj_list = np.array(reproj_list)
        for i in range(reproj_list.shape[1]):
            # TODO: why use (x - mu) / std ?
            std = np.std(reproj_list[:, i])
            std = 1 if std == 0 else std
            reproj_list[:, i] = abs((reproj_list[:, i] - np.mean(reproj_list[:, i]))) / std
        for j in range(reproj_list.shape[0]):
            # TODO: eliminate magic number
            reproj_list[j, candidate_indices[j]] *= 3  # pay more attention to the current reprojection error
        reproj_list = np.sum(reproj_list, axis=1)

        # print results
        idx = candidate_indices[np.argmin(reproj_list)]
        if verbose:
            print ('reprojection list: {}'.format(reproj_list))
            print ('least reprojection idx: {}, new method idx: {}'.format(np.argmin(np.abs(ret_list)), idx))
            print ('corresponding reproj error: {}, {}'.format(ret_list[np.argmin(np.abs(ret_list))], ret_list[idx]))

        return candidate_indices, reproj_list

    @staticmethod
    def eval_intrinsic_params(data_dir, img_shape, k_lst, selected_params, iter_num, err):
        # draw distribution of fx, fy, cx, cy
        k_lst = np.array(k_lst)

        fxy = k_lst[:, :2, :2]
        uv = k_lst[:, :, 2:]

        k, distortion = selected_params
        d1, d2 = distortion[:2]
        dist = np.linalg.norm(np.array(img_shape) / 2.0)
        rx_max = dist / k[0, 0]
        ry_max = dist / k[1, 1]
        x = np.arange(0, rx_max, 0.01)
        y = np.arange(0, ry_max, 0.01)

        # draw distortion effects
        f = lambda r, d_coeff1, d_coeff2: (1 + d_coeff1 * r ** 2 + d_coeff2 * r ** 4) * r

        fig = plt.figure()
        plt.subplot(221)
        plt.scatter(uv[:, 0], uv[:, 1], c='b')
        plt.title("reproj_error: " + str(err))
        plt.subplot(222)
        plt.scatter(fxy[:, 0, 0], fxy[:, 1, 1], c='b')
        plt.subplot(223)
        plt.plot(x, f(x, d1, d2), 'o', x, x, 'k')
        plt.subplot(224)
        plt.plot(y, f(y, d1, d2), 'o', y, y, 'k')

        fig.savefig(os.path.join(data_dir, 'evaluation_iter_{}.png'.format(iter_num)))
        plt.close(fig)

        return True

    def _compute_init_algo_intrinsic(self, corners_cooders, corners_list, camera_cfg, img_shape, img_flipped=False):
        """
        return the initial alogi intrinsic
        """
        params, intrinsicMatrix = self._compute_init_guess_intrinsic(camera_cfg, img_shape, img_flipped)
        print ('img_shape: ',img_shape)
        ret1, matrix, distort1, rvecs1, tvecs1, newObPoints = \
        cv.calibrateCameraRO(corners_cooders, corners_list, img_shape, -1, intrinsicMatrix, None,
                            flags=self.cali_flags, criteria=self.cali_crit)
        return params, matrix    
    
    def _get_corner_coords(self, corners_list_len):
        """
        corner_coords(N,1,W*H,3): 3D coordinate the origin is on the top right corner
        """
        corner_coord = np.mgrid[:self.pattern_shape[0], :self.pattern_shape[1]].T.reshape(-1, 2)
        corner_coord = np.hstack((corner_coord, np.zeros((corner_coord.shape[0], 1))))
        corner_coord = np.expand_dims(corner_coord, 0)
        corner_coords = np.repeat(corner_coord[None, :], corners_list_len, axis=0).astype(np.float32)
        corner_coords *= self.corner_distance
        return corner_coords     

    def _optimize_param_multithr(self, candidate_idx, idx_list, intrinsic_results, corners, img_shape):
        num_worker = cpu_count()

        if self.num_thread > (num_worker - 2):
            thread_num = num_worker - 2
        else:
            thread_num = self.num_thread

        # MLE using opencv for all candidate K
        intrinsic_lst = [intrinsic_results[i] for i in candidate_idx]
        idx_batch = np.array(idx_list)[candidate_idx]
        corners_x_lst = [corners[0][i] for i in idx_batch]
        corners_y_lst = [corners[1][i] for i in idx_batch]
        chunksize = max(len(candidate_idx)/thread_num, 1)

        pool = ThreadPool(thread_num)
        res_lst = pool.map(partial(self._optimize_param, img_shape=img_shape),
                           zip(intrinsic_lst, corners_x_lst, corners_y_lst),
                           chunksize=chunksize)

        pool.close()
        pool.join()

        k_list, distort_list, ret_list, r_list, t_list = [], [], [], [], []
        for res in res_lst:
            ret_list += [res[0]]
            k_list += [res[1]]
            distort_list += [res[2][0]]
            r_list += [res[3]]
            t_list += [res[4]]
        id_list = idx_batch.tolist()

        candidate_indices, reproj_list = \
            self.select_candidate_by_reprojection_err(k_list, distort_list, ret_list, r_list, t_list,
                                                      id_list, corners, self.verbose, candidate_num=5)
        idx = candidate_indices[np.argmin(np.abs(reproj_list))]
        k_list = np.array(k_list)

        # # get the result base on the biggest second eigen value
        # self.select_candiate_by_2nd_smallest_eignvalue(candidate_idx, idx_list, k_list, r_list, t_list,
        #                                                distort_list, corners, ret_list)

        # get the result
        fx, fy, u, v = intrinsic_results[idx]
        rls_k_mat = np.array([[fx, 0, u],
                              [0, fy, v],
                              [0, 0, 1]])

        opt_k_mat = k_list[idx]
        opt_distortion = distort_list[idx]

        print ("-----------rls K------------")
        print (rls_k_mat)
        print ("-----------opt K------------")
        print (opt_k_mat)
        print ("-------------D--------------")
        print (opt_distortion)
        print ('reprojection error: {}'.format(ret_list[idx]))

        return rls_k_mat, opt_k_mat, opt_distortion, k_list, distort_list, \
               r_list, t_list, id_list, ret_list, ret_list[idx] 
    
    def _get_intrinsic(self, homographies, intrinsic_init, reg=1e-8):
        """
        return rls intrinsic params
        Don't use RLS for short and middle range camera
        """
        if not self.RLS_cfg.get(str(self.cam_config['focal_length']) + 'mm'):
            return [intrinsic_init[0], intrinsic_init[1], intrinsic_init[2], intrinsic_init[3]]
        V = []
        for h in homographies:
            curr = h.reshape((3, 3))
            V.append(self.cal_v(0, 1, curr))
            V.append(self.cal_v(0, 0, curr) - self.cal_v(1, 1, curr))
        V = np.array(V)

        x = np.array(intrinsic_init)
        c = np.zeros_like(x)
        c[0] = x[0] * x[0] / (x[1] * x[1])
        c[1] = -x[2]
        c[2] = - c[0] * x[3]
        c[3] = -x[2] + c[0] * x[3] * x[3] + x[0] * x[0]
        A = V[:, 2:]
        b = - V[:, 0]
        f = np.linalg.inv(A.T.dot(A) + reg * np.eye(4)).dot(A.T.dot(b) + reg * c.T)
        u = -f[1]
        v = - f[2] / f[0]
        fx = np.sqrt(f[3] - f[1] + f[2] * v)
        fy = np.sqrt(1 / f[0]) * fx
        return [fx, fy, u, v]

    def _optimize_param_multithr(self, candidate_idx, idx_list, intrinsic_results, corners, img_shape):
        num_worker = cpu_count()

        if self.num_thread > (num_worker - 2):
            thread_num = num_worker - 2
        else:
            thread_num = self.num_thread

        # MLE using opencv for all candidate K
        intrinsic_lst = [intrinsic_results[i] for i in candidate_idx]
        idx_batch = np.array(idx_list)[candidate_idx]
        corners_x_lst = [corners[0][i] for i in idx_batch]
        corners_y_lst = [corners[1][i] for i in idx_batch]
        chunksize = max(len(candidate_idx)/thread_num, 1)

        pool = ThreadPool(thread_num)
        res_lst = pool.map(partial(self._optimize_param, img_shape=img_shape),
                           zip(intrinsic_lst, corners_x_lst, corners_y_lst),
                           chunksize=chunksize)

        pool.close()
        pool.join()

        k_list, distort_list, ret_list, r_list, t_list = [], [], [], [], []
        for res in res_lst:
            ret_list += [res[0]]
            k_list += [res[1]]
            distort_list += [res[2][0]]
            r_list += [res[3]]
            t_list += [res[4]]
        id_list = idx_batch.tolist()

        candidate_indices, reproj_list = \
            self.select_candidate_by_reprojection_err(k_list, distort_list, ret_list, r_list, t_list,
                                                      id_list, corners, self.verbose, candidate_num=5)
        idx = candidate_indices[np.argmin(np.abs(reproj_list))]
        k_list = np.array(k_list)

        # # get the result base on the biggest second eigen value
        # self.select_candiate_by_2nd_smallest_eignvalue(candidate_idx, idx_list, k_list, r_list, t_list,
        #                                                distort_list, corners, ret_list)

        # get the result
        fx, fy, u, v = intrinsic_results[idx]
        rls_k_mat = np.array([[fx, 0, u],
                              [0, fy, v],
                              [0, 0, 1]])

        opt_k_mat = k_list[idx]
        opt_distortion = distort_list[idx]

        print ("-----------rls K------------")
        print (rls_k_mat)
        print ("-----------opt K------------")
        print (opt_k_mat)
        print ("-------------D--------------")
        print (opt_distortion)
        print ('reprojection error: {}'.format(ret_list[idx]))

        return rls_k_mat, opt_k_mat, opt_distortion, k_list, distort_list, \
               r_list, t_list, id_list, ret_list, ret_list[idx]

    def _save_cluster(self, kmeans_labels, img_names):
        print("Saving the clustered images")
        path = self.data_dir + '/cluster'
        if os.path.exists(path):  # delete the folder if already exists
            shutil.rmtree(path)
        os.mkdir(path)
        for i in range(int(np.max(kmeans_labels)) + 1):
            os.mkdir(os.path.join(path, str(i + 1)))
        for i in range(len(img_names)):
            image = cv.imread(str(self.img_path + img_names[i]), cv.IMREAD_COLOR)
            cv.imwrite(path + '/' + str(int(kmeans_labels[i]) + 1) + '/' + img_names[i], image)

    def _save_batch_imgs(self, idx_lst, img_names):
        print("Saving the selected images for each batch")
        path = self.data_dir + '/batch'
        if os.path.exists(path):  # delete the folder if already exists
            shutil.rmtree(path)
        os.mkdir(path)
        for i, idx in enumerate(idx_lst):
            sub_path = os.path.join(path, str(i))
            os.mkdir(sub_path)
            for im_name in img_names[idx]:
                image = cv.imread(self.img_path + im_name)
                cv.imwrite(os.path.join(sub_path, im_name), image)
    
    def _img_cluster(self, batch_size, homographies, init_k_params, reg, img_names, iter_num):
        """
        K Means to divide into different poses of boards
        """

        num_imgs = len(img_names)

        if num_imgs < Calibrator.min_num_img_todo_sampling:
            num_cluster = 1
            batch_num = 1
            num_each_cluster = num_imgs
            kmeans_labels = np.zeros(num_imgs)
            print ('with {} images, no sampling needed'.format(num_imgs))
        else:
            print ("please make sure the img num is less then min_num_img_todo_sampling")
            raise Exception('error')


        # assign the index of each batch here
        idx_list = []
        intrinsic_results = []
        label_index_map, label_size_map = [], []

        for i in range(num_cluster):
            label_index_map.append(np.where(kmeans_labels == i)[0].flatten())
            label_size_map.append(len(label_index_map[-1]))

        # np.random.seed(5)  # set the seed so same data will yield same results
        np.random.seed()
        for batch_iter in range(batch_num):
            idx = []
            for i in range(num_cluster):
                if num_each_cluster >= label_size_map[i]:
                    idx.extend(label_index_map[i])
                else:
                    idx.extend(label_index_map[i][np.random.choice(label_size_map[i], num_each_cluster, replace=False)])
            idx_list.append(idx)
            homos = homographies[idx]
            intrinsic = self._get_intrinsic(homos, init_k_params, reg)
            intrinsic_results.append(intrinsic)

        # get all the intrinsic result
        candidate_idx = range(len(intrinsic_results))

        # save clustered data here
        if self.save:
            self._save_cluster(kmeans_labels, img_names)
            self._save_batch_imgs(idx_list, img_names)

        return candidate_idx, idx_list, intrinsic_results

    def optimize_param_multithr(self, candidate_idx, idx_list, intrinsic_results, corners, img_shape):
        num_worker = cpu_count()

        if self.num_thread > (num_worker - 2):
            thread_num = num_worker - 2
        else:
            thread_num = self.num_thread

        # MLE using opencv for all candidate K
        intrinsic_lst = [intrinsic_results[i] for i in candidate_idx]
        idx_batch = np.array(idx_list)[candidate_idx]
        corners_x_lst = [corners[0][i] for i in idx_batch]
        corners_y_lst = [corners[1][i] for i in idx_batch]
        chunksize = max(len(candidate_idx)/thread_num, 1)

        pool = ThreadPool(thread_num)
        res_lst = pool.map(partial(self._optimize_param, img_shape=img_shape),
                           zip(intrinsic_lst, corners_x_lst, corners_y_lst),
                           chunksize=chunksize)

        pool.close()
        pool.join()

        k_list, distort_list, ret_list, r_list, t_list = [], [], [], [], []
        for res in res_lst:
            ret_list += [res[0]]
            k_list += [res[1]]
            distort_list += [res[2][0]]
            r_list += [res[3]]
            t_list += [res[4]]
        id_list = idx_batch.tolist()

        candidate_indices, reproj_list = \
            self._select_candidate_by_reprojection_err(k_list, distort_list, ret_list, r_list, t_list,
                                                      id_list, corners, self.verbose, candidate_num=5)
        idx = candidate_indices[np.argmin(np.abs(reproj_list))]
        k_list = np.array(k_list)

        # # get the result base on the biggest second eigen value
        # self.select_candiate_by_2nd_smallest_eignvalue(candidate_idx, idx_list, k_list, r_list, t_list,
        #                                                distort_list, corners, ret_list)

        # get the result
        fx, fy, u, v = intrinsic_results[idx]
        rls_k_mat = np.array([[fx, 0, u],
                              [0, fy, v],
                              [0, 0, 1]])

        opt_k_mat = k_list[idx]
        opt_distortion = distort_list[idx]

        print ("-----------rls K------------")
        print (rls_k_mat)
        print ("-----------opt K------------")
        print (opt_k_mat)
        print ("-------------D--------------")
        print (opt_distortion)
        print ('reprojection error: {}'.format(ret_list[idx]))

        return rls_k_mat, opt_k_mat, opt_distortion, k_list, distort_list, \
               r_list, t_list, id_list, ret_list, ret_list[idx]

    def _optimize_param(self, value, img_shape):
        k, corners_x, corners_y = value
        fx, fy, u, v = k
        init_k_mat = np.array([[fx, 0, u],
                               [0, fy, v],
                               [0, 0, 1]])
        if self.is_using_OR_calibrate:
            Calibrator.iFixedPoint = len(corners_x[0]) - 2
        newObPoints = None
        # print Calibrator.iFixedPoint int(len(corners_x) - 3)
        # NOTE:there is a bug in function calibrateCameraRO() of opencv4 when iFixedPoint>0
        # ret, k, distort, rvecs, tvecs, newObPoints = \
        #    cv.calibrateCameraRO(corners_x, corners_y, img_shape, -1, init_k_mat, None,
        #                       flags=self.cali_flags, criteria=self.cali_crit)
        ret, k, distort, rvecs, tvecs = \
            cv.calibrateCamera(corners_x, corners_y, img_shape, init_k_mat, None,
                               flags=self.cali_flags, criteria=self.cali_crit)
        return ret, k, distort, rvecs, tvecs, newObPoints

    def filter_by_reprojection_error(self, homographies, corners,
                                     img_names, k_list, d_list, r_list, t_list, id_list, iter_num):

        k_list, d_list = np.array(k_list), np.array(d_list)
        r_list, t_list, id_list = np.array(r_list), np.array(t_list), np.array(id_list)
        reprojection_error = np.zeros((corners[0].shape[0],))
        counter = np.zeros((corners[0].shape[0],))
        for i in range(k_list.shape[0]):
            for j in range(id_list.shape[1]):
                proj_pts, _ = cv.projectPoints(corners[0][id_list[i, j]], r_list[i, j], t_list[i, j],
                                               k_list[i], d_list[i])
                proj_pts = proj_pts[:, 0, :]
                reprojection_error[id_list[i, j]] += np.sum(np.abs(proj_pts - corners[1][id_list[i, j], :, 0, :]) ** 2)
                counter[id_list[i, j]] += len(corners[1][id_list[i, j], :, 0, :])
        for i in range(corners[0].shape[0]):
            if counter[i] != 0:
                reprojection_error[i] = np.sqrt(reprojection_error[i] / float(counter[i]))

        # plt.bar(range(reprojection_error.shape[0]), reprojection_error)
        idx_iqr = outliers_iqr(reprojection_error, lower=False)
        idx_norm_std = outliers_norm_std(reprojection_error, lower=False)
        invalid_idx = idx_iqr | idx_norm_std

        # Fix the id_list after filter
        current_id = 0
        id_list = np.asarray(id_list).astype(float)
        for i in range(corners[0].shape[0]):
            if not invalid_idx[i]:
                id_list[id_list == i] = current_id
                current_id += 1
            else:
                # distory the invaild index
                id_list[id_list == i] = np.inf
        if sum(invalid_idx) == 0:
            print ("no outliers are found by re-proj error")
            success = True
        else:
            success = False
            valid_idx = ~invalid_idx

            homographies = homographies[valid_idx]
            corners[1] = corners[1][valid_idx]
            corners[0] = corners[0][valid_idx]
            img_names = img_names[valid_idx]

            if self.viz:
                fig = plt.figure()
                plt.bar(range(len(reprojection_error)), reprojection_error)
                plt.bar(np.where(invalid_idx == 1)[0], reprojection_error[invalid_idx])
                plt.title('reprojection error')
                plt.savefig(os.path.join(self.progress_output_dir, 'reproj_filter_iter_{}.png'.format(iter_num)))
                plt.close(fig)

            if self.save:
                path_filtered = os.path.join(self.data_dir, 'Filtered_imgs_iter_{}'.format(iter_num))
                path_remained = os.path.join(self.data_dir, 'Remained_imgs_iter_{}'.format(iter_num))
                if not os.path.exists(path_filtered):
                    os.mkdir(path_filtered)
                if not os.path.exists(path_remained):
                    os.mkdir(path_remained)

                for im_name in img_names:
                    path_img = os.path.join(self.data_dir, 'imgs', im_name)
                    cv.imwrite(os.path.join(path_filtered, im_name), cv.imread(path_img))
                for im_name in img_names:
                    path_img = os.path.join(self.data_dir, 'imgs', im_name)
                    cv.imwrite(os.path.join(path_remained, im_name), cv.imread(path_img))

        return homographies, corners, img_names, id_list, success
    
    def calibrate(self, corners_list, img_names, img_shape):
  
        iter_img_sum = 0
        success= False

        corners_coords = self._get_corner_coords(len(corners_list))
        homographies = compute_homographies(corners_coords, corners_list)
        corners = [corners_coords, corners_list]

        print ("\nstart to calibrate camera intrinsic params...")
        print ("-----------initial  K------------")
        init_k_params, init_k_mat = self._compute_init_guess_intrinsic(self.cam_config, img_shape, self.flip_input_img)
        print (init_k_mat)

        iter_num = 1
        opt_k = np.eye(3)
        opt_distortion = np.zeros(5)
        err = 0
        success = False
        ret_list, k_list, distort_list, r_list, t_list, id_list = None, None, None, None, None, None

        while not success and iter_num <= self.max_iter:
            print ('\nstarting {} iteration'.format(iter_num))
            if iter_num != 1 and self.is_ring:
                # TODO: refine the corner detection for ring
                kwargs = {'corners': corners_list, 'img_names': img_names, 'ret_list': ret_list, 'k_list': k_list,
                          'distort_list': distort_list, 'r_list': r_list, 't_list': t_list, 'id_list': id_list}
                homographies, corners, _, _ = self.detector.detect(self.board, self.num_thread, iter_num, **kwargs)

            # divide into different poses of boards
            candidate_idx, idx_list, intrinsic_results = \
                self._img_cluster(self.batch_size, homographies, init_k_params, self.reg, img_names, iter_num)
            
            print ("parameter optimizing...")
            # rls_k, opt_k, opt_distortion, k_list, distort_list, r_list, t_list, id_list, ret_list, err = \
            #     self.optimize_param(candidate_idx, idx_list, intrinsic_results, corners, img_shape)
            rls_k, opt_k, opt_distortion, k_list, distort_list, r_list, t_list, id_list, ret_list, err = \
                self.optimize_param_multithr(candidate_idx, idx_list, intrinsic_results, corners, img_shape)

            if self.viz:
                self.eval_intrinsic_params(self.progress_output_dir, img_shape,
                                           k_list, [opt_k, opt_distortion], iter_num, err)  

            # Re run ###
            # filtering with reprojection error
            homographies, corners, img_names, id_list, success = \
                self.filter_by_reprojection_error(homographies, corners, img_names,
                                                  k_list, distort_list, r_list, t_list, id_list, iter_num)

            iter_num += 1
            print ("----------------------")

        print ("saving intrinsic parameters...")
        params_res = save_params(self.data_dir, self._bag_name, self.cam_id, opt_k, opt_distortion,
                    img_shape, self.output_img_shape, self.flip_input_img, self.flip_output_img, err, 
                    self.cam_config.get('focal_length'))

        return params_res


## for test
# if __name__ == '__main__':
#     from calibrator import Calibrator
#     cfg_path = os.path.join(os.path.dirname(__file__), '../../config/config.yaml')
#     rls_calib = Calibrator(cfg_path)
#     rls_calib.calibrate()