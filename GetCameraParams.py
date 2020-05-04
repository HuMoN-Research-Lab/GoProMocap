import numpy as np
import cv2
import glob
import os 
from config import cam_names
from create_project import calibrationFilePath, rawVideoFolder, checkerVideoFolder, calibVideoFilepath, cameraParamsFilePath


def getCameraParams():
    amountOfCalImages = 7
    calibDatadir  = [calibVideoFilepath]
    for dir in calibDatadir:
        k = 0 
        for video in os.listdir(dir):
            
            vidcap = cv2.VideoCapture(calibVideoFilepath+'/'+video)
            frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            calImagesinVideo = frame_count/amountOfCalImages
            vidlength = range(int(frame_count)) 
            
            for ii in vidlength:

                success,image = vidcap.read()
                if success:
                    height , width , layers =  image.shape 
                    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    #single_video.append(image)   
                    if not os.path.exists(calibrationFilePath + '/'+cam_names[k]+'_CalibrationImages'):
                        os.mkdir(calibrationFilePath + '/'+cam_names[k]+'_CalibrationImages')                       
                    cv2.imwrite(calibrationFilePath+'/'+cam_names[k]+'_CalibrationImages/frame%d.jpg' %ii , image)     # save frame as JPEG file    
                else:
                    continue
            
            
            
            
            # termination criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            objp = np.zeros((9*6,3), np.float32)
            objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)

            # Arrays to store object points and image points from all the images.
            objpoints = [] # 3d point in real world space
            imgpoints = [] # 2d points in image plane.

            images = glob.glob(calibrationFilePath+'/'+cam_names[k]+'_CalibrationImages/*.jpg')
            p = 0
            for fname in images[::int(calImagesinVideo-1)]:
                img = cv2.imread(fname)
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                # Find the chess board corners
                ret, corners = cv2.findChessboardCorners(img, (6,9),None)
                # If found, add object points, image points (after refining them)
                if ret == True:
                    objpoints.append(objp)

                    corners2 = cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
                    imgpoints.append(corners2)
                
                    # Draw and display the corners
                    img = cv2.drawChessboardCorners(img, (6,9), corners2,ret)
                    cv2.imshow('img',img)
                    cv2.waitKey(500)
                p+=1

            print(p)
            cv2.destroyAllWindows()

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
            if not os.path.exists(cameraParamsFilePath + '/'+cam_names[k]):
                os.mkdir(cameraParamsFilePath + '/'+cam_names[k])   

            np.save(cameraParamsFilePath + '/'+cam_names[k]+'/Calibration_ret.npy',ret)
            np.save(cameraParamsFilePath + '/'+cam_names[k]+'/Calibration_mtx.npy',mtx)
            np.save(cameraParamsFilePath + '/'+cam_names[k]+'/Calibration_dist.npy',dist)
            np.save(cameraParamsFilePath + '/'+cam_names[k]+'/Calibration_rvec.npy',rvecs)
            np.save(cameraParamsFilePath + '/'+cam_names[k]+'/Calibration_tvecs.npy',tvecs)
            
            print(dist)
            tot_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                tot_error += error
            meanError = tot_error/len(objpoints)
            print(cam_names[k])
            print ("mean error: ", meanError)
            print(tot_error)
            k +=1
#getCameraParams()
