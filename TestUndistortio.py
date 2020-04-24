import numpy as np 
import cv2
import glob
import os 

baseFilePath = 'D:/CalibrationTest'

mtx = np.load('D:/Calibration/CamDParams/Calibration_mtx.npy')
dist = np.load('D:/Calibration/CamDParams/Calibration_dist.npy')

img = cv2.imread(baseFilePath+'/CamD_Calibration/frame7.jpg')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite(baseFilePath+'/Undistorted.png',dst)
cv2.imshow('frame', dst)

