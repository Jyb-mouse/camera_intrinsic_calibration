import numpy as np
import cv2 as cv
import os
import math
import matplotlib.pyplot as plt


def save_detected_corners(data_dir, iter_num, valid_img_names, corners):
    """
    
    """
    if not os.path.exists(data_dir + '/detected_corner_iter' + str(iter_num)):
        os.mkdir(data_dir + '/detected_corner_iter' + str(iter_num))
    for i in range(len(valid_img_names)):
        image = cv.imread(data_dir + '/imgs/' + valid_img_names[i])
        plt.imshow(image)
        plt.scatter(corners[1][i, :, 0, 0], corners[1][i, :, 0, 1], linewidths=0.1, marker='x', c='r')
        plt.savefig(data_dir + '/detected_corner_iter' + str(iter_num) + '/' + valid_img_names[i])
        plt.close()


def compute_homographies(corner_coords, corner_pts):
    """
        Lowest level function to find the homographies
        input:
            corner_coords(N,1,W*H,3): 3D coordinate the origin is on the top right corner
            corner_pts(N,W*H,1,2): 2D coordinate on the image
        output:
            Homographies (N,3,3): Rotation matrix for each image
    """
    homographies = []
    for i in range(len(corner_coords)):
        real = np.array(corner_coords[i, 0, :, :2])
        sensed = corner_pts[i, :, 0, :]
        H, J = cv.findHomography(real, sensed)
        homographies.append(H)
    return np.array(homographies)
