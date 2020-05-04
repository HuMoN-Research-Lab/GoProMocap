import numpy as np
import cv2
import glob
import os 
from config import cam_names
from create_project import calibrationFilePath, rawVideoFolder, checkerVideoFolder, cameraParamsFilePath, baseFilePath
import subprocess

def UndistortVideo():
    rawDatadir = [rawVideoFolder]
    '''
    for dir in rawDatadir:
        k = 0 
        
        for video in os.listdir(dir):
            mtx = np.load(cameraParamsFilePath+'/'+cam_names[k]+'/Calibration_mtx.npy')
            dist = np.load(cameraParamsFilePath+'/'+cam_names[k]+'/Calibration_dist.npy')
            vidcap = cv2.VideoCapture(rawVideoFolder+'/'+video)
            frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            #size = (int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            for ii in range(int(frame_count)):
                ret, frame =vidcap.read()
                if ret:       
                    h,  w = frame.shape[:2]
                    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

                    # undistort
                    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
                    # crop the image
                    x,y,w,h = roi
                    dst = dst[y:y+h, x:x+w]
                    #dst = cv2.resize(dst, (1280,960))
                    
                    height, width, layers = dst.shape
                    break

            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            fps = int(vidcap.get(cv2.CAP_PROP_FPS))
            frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            size = (width, height)

            print(size)

            writer = cv2.VideoWriter(baseFilePath+'/Intermediate/Undistorted/'+video, -1, fps, size)
            for ii in range(int(frame_count)):
                
                ret, frame =vidcap.read()
                if ret:
                    h,  w = frame.shape[:2]
                    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
  
                     # undistort
                    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
                    x,y,w,h = roi
                    dst = dst[y:y+h, x:x+w]
                    #dst = cv2.resize(dst, (1280,960))
                    #cv2.imshow('frame',dst)
                    #cv2.waitKey(5000)
                    writer.write(dst)
            vidcap.release()
            writer.release()
            print(dst.shape)
            k +=1
    '''
    
    for dir in rawDatadir:
        for video in os.listdir(dir):
            subprocess.call(['ffmpeg', '-i', rawVideoFolder+'/'+video, '-vf', "lenscorrection=cx=0.5:cy=0.5:k1=-.1432:k2=-0.042", baseFilePath+'/Intermediate/Undistorted/'+video])
    '''
    for dir in rawDatadir:
        for video in os.listdir(dir):
            subprocess.call(['ffmpeg', '-i', rawVideoFolder+'/'+video, '-vf', 'lensfun=make=Canon:model=' ,'Canon EOS 100D',':lens_model=','Canon EF-S 18-55mm f/3.5-5.6 IS STM', baseFilePath+'/Intermediate/Undistorted/'+video])
    '''
#UndistortVideo()

