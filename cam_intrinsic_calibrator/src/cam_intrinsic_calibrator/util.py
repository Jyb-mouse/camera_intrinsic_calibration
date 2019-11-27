import os
import yaml
import numpy as np
from datetime import datetime


"""
    stats utils
"""


def outliers_iqr(ys, lower=True, upper=True):
    ratio = 1.5
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * ratio)
    upper_bound = quartile_3 + (iqr * ratio)

    idx = np.zeros(len(ys)).astype(bool)
    if lower:
        idx = (ys < lower_bound)
    if upper:
        idx |= (ys > upper_bound)

    return idx


def outliers_norm_std(data, lower=True, upper=True):
    ratio = 1.5
    mu, std = data.mean(), data.std()
    idx = np.zeros(len(data)).astype(bool)
    if lower:
        idx = (data < mu - ratio * std)
    if upper:
        idx |= (data > mu + ratio * std)

    return idx

def scale_cam_intrinsic(intrinsic, src_shape, dst_shape, override=False):
    """
    scale intrinsic due to image size changed
    :param intrinsic: source intrinsic
    :param src_shape: source image shape
    :param dst_shape: destination image shape
    :return:
    """
    src_w, src_h = src_shape
    dst_w, dst_h = dst_shape
    rx, ry = src_w * 1. / dst_w, src_h * 1. / dst_h
    mat_intr = intrinsic if override else intrinsic.copy()
    mat_intr[0, 0] /= rx
    mat_intr[0, 2] /= rx
    mat_intr[1, 1] /= ry
    mat_intr[1, 2] /= ry
    return mat_intr


def flip_intrinsic(intrinsic, override=False):
    mat_intr = intrinsic if override else intrinsic.copy()
    mat_intr[0, 0], mat_intr[1, 1] = mat_intr[1, 1], mat_intr[0, 0]
    mat_intr[0, -1], mat_intr[1, -1] = mat_intr[1, -1], mat_intr[0, -1]
    return mat_intr


def save_params(output_file_path, bag_name, cam_id, intrinsic, distortion,
                input_shape, output_shape, flip_input_img, flip_output_img,
                reproj_err, focal_length, vehicle_name):

    output_file_path = os.path.expanduser(output_file_path)
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    # flip default output image shape
    # if flip_input_img:
    #     output_shape = output_shape[::-1]

    # scale intrinsic, assert input_shape and output_shape has the same ratio
    scale_cam_intrinsic(intrinsic, input_shape, output_shape, override=True)

    # flip output image shape if needed
    if flip_input_img == flip_output_img == True:
        flip_intrinsic(intrinsic, override=True)

    meta_dict = {'dir_name': bag_name, 'calibrate_date': datetime.now().strftime('%Y-%m-%d'),
                 'vehicle': vehicle_name,'reproj_err': reproj_err, 'focal_length': focal_length}
    res_dict = {'distortion': [distortion.tolist()], 'intrinsic': intrinsic.tolist(),
                'width': int(output_shape[0]), 'height': int(output_shape[1]),
                'meta': meta_dict}
    with open(os.path.join(output_file_path, 'camera-{}.yaml'.format(cam_id)), 'w') as f:
        yaml.safe_dump(res_dict, f)
    return res_dict

def time2secs(time, ts_begin):
    """
    input: "AA:BB", AA -> min, BB -> sec
           or unix time, 15xxxxxxxxxxx
    output: number of seconds since ts_begin
    """
    assert type(time) in [int, str], "input format could only be int or string"
    if isinstance(time, int):
        return int((time - ts_begin) * 1. / 1e9)
    else:
        res = time.split(':')
        assert len(res) == 2, '{} format not correct, should be AA:BB'.format(time)
        _min, _sec = res
        return int(_min) * 60 + int(_sec)