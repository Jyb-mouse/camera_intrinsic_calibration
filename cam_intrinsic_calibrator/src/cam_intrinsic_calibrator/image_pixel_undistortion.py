import os
import numpy as np
import cv2 as cv
from transformation_util2 import undistortPoint, Project3DPtsToImg

class ImagePixelUndistortion():
    def __init__(self, img_path, intrinsic, distortion):
        self.img_path = img_path
        self.intrinsic = intrinsic
        self.distortion = distortion
        try:
            self.img = cv.imread(self.img_path)
            self.image_height = self.img.shape[0]
            self.image_width = self.img.shape[1]
            print()
        except Exception as e:
            print("open image: {} fail!".format(self.img_path))
            raise Exception(e)
    
    def undistort_pixel(self, pxl):
        # pxl = (912, 472)
        # pxl = (1824, 944)
        un_pxl = undistortPoint(pxl, self.intrinsic, self.distortion, max_count=1000)
        print(un_pxl)
        pxl_x = un_pxl[0]*self.intrinsic[0][0]+self.intrinsic[0][2]
        pxl_y = un_pxl[1]*self.intrinsic[1][1]+self.intrinsic[1][2]
        return [pxl_x, pxl_y]

    def distort_pixel(self, pxl):
        direction_x = (pxl[0]-self.intrinsic[0][2]) / self.intrinsic[0][0]
        direction_y = (pxl[1]-self.intrinsic[1][2]) / self.intrinsic[1][1]
        direction = np.array([np.array([direction_x, direction_y, 1])])
        # print(point.reshape(-1, 1, 3))
        # img_pts, _ = cv.projectPoints(direction, np.zeros((3, 1)), np.zeros((3, 1)),
        #                               self.intrinsic, self.distortion)
        img_pts = Project3DPtsToImg(direction, self.intrinsic, self.distortion)
        # img_pts = img_pts[:, 0, :]
        return img_pts[0]

    def compute_map(self):
        pxl = (0,0)
        un_pxl = undistortPoint(pxl, self.intrinsic, self.distortion)
        # un_pxl = un_pxl[:,0,:]
        point = np.array([np.array([un_pxl[0],un_pxl[1],1])])
        # print(point.reshape(-1, 1, 3))
        img_pts, _ = cv.projectPoints(point, np.zeros((3, 1)), np.zeros((3, 1)),
                                      self.intrinsic, self.distortion)
        img_size = (self.image_width,self.image_height)
        # new_img_size = (1824*2, 944*2)
        new_img_size = (7200, 2400)
        newcameramatrix, _ = cv.getOptimalNewCameraMatrix(
                           self.intrinsic, self.distortion, img_size, 0, new_img_size, 1)
        mapx, mapy = cv.initUndistortRectifyMap(self.intrinsic, self.distortion, None, newcameramatrix, new_img_size, cv.CV_32FC1)
        # np.savetxt('intrinsic.txt', self.intrinsic)
        # np.savetxt('mapx.txt', mapx)
        # np.savetxt('mapy.txt', mapy)
        # np.savetxt('intrinsic.txt', newcameramatrix)
        # np.savetxt('distortion.txt', self.distortion)
        ud_img = cv.undistort(self.img, self.intrinsic, self.distortion)
        frame = cv.remap(self.img, mapx, mapy, cv.INTER_LINEAR)
        # frame = cv.undistort(self.img, self.intrinsic, self.distortion, newCameraMatrix=newcameramatrix)
        #cv.circle(frame, (int(un_pxl[0][0]),int(un_pxl[0][1])), 1, (0,0,255),2)
        print(frame.shape)
        cv.imwrite('./frame2.jpg', frame)
        # cv.imwrite('frame1.jpg', self.img)
        frame = cv.resize(frame, (int(7200/4), int(2400/4)))
        ud_img = cv.resize(ud_img, (int(3820/2), int(2160/2)))
        cv.imshow('img', frame)
        cv.waitKey(0)
        print()

    def process(self):
        self.compute_map()
        dist = 0
        for height in range(10, self.image_height, 10):
            for width in range(10, self.image_width, 10):
                pxl = (width, height)
                un_pxl = self.undistort_pixel(pxl)
                dis_pxl = self.distort_pixel(un_pxl)
                dis_pxl = (int(dis_pxl[0]),int(dis_pxl[1]))
                if un_pxl[0]>=0 and un_pxl[0]<self.image_width and un_pxl[1]>=0 and un_pxl[1]<=self.image_height:
                    dist1 = np.sqrt((dis_pxl[0]-pxl[0])**2 + (dis_pxl[1]-pxl[1])**2)
                    # if dist1>=dist and dist1<200:
                    #     dist = dist1
                    # #cv.circle(self.img, (int(un_pxl[0]), int(un_pxl[1])), 1, (0,255,0),1)
                    # cv.line(self.img, pxl, (int(un_pxl[0]), int(un_pxl[1])), (255,0,0), 1)
                    # cv.circle(self.img, pxl, 1, (0,0,255),1)
                cv.circle(self.img, pxl, 1, (0,0,255),1)
                cv.circle(self.img, dis_pxl, 1, (0,255,0),1)
                cv.line(self.img, pxl, dis_pxl, (255,0,0), 1)
        print(dist)
        img_show = cv.resize(self.img, (int(width/2), int(height/2)))
        cv.imwrite('frame1.jpg', self.img)
        cv.imshow('jpg', img_show)
        cv.waitKey(0)

if __name__ == "__main__":
    cam_intrinsic = np.array([
            [2426.1234612, 0.0, 1918.83591],
            [0.0, 2426.6058331, 1082.38481],
            [0.0, 0.0, 1.0]
        ])
    cam_distort = np.array(
            [5.480320950047, 2.3289343197, -0.00000513480953, -0.000014420611, 0.05047945984, 6.1479871719, 5.630043345, 0.62771997905]
        )
    img_path = '/media/lingbo/F8FCA47FFCA439B0/output/jpg/1678434323001.jpg'
    img_pxl_undistort = ImagePixelUndistortion(img_path, cam_intrinsic, cam_distort)
    img_pxl_undistort.process()
