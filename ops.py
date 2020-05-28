import numpy as np
import sys
import cv2
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from config import num_of_cameras,useCheckerboardVid, cam_names
from create_project import baseFilePath, checkerVideoFolder, rawVideoFolder




class Exceptions(Exception):
    pass


def get_TransMat(base_H,H1,H2=None,H3=None):
    """
    calculate homogeneous transformation matrix between base Cam and the other cameras
    and calculate the projection matrix of all cameras

    """
    M_base = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
    H_base_1 = H1.dot(inverseH(base_H))

    T1 = H_base_1[:3,-1].reshape(3,1)
    R1 = H_base_1[:3,:3]
    M1 = np.hstack((R1, T1))
    
    if num_of_cameras == 2 :
        return M_base,M1
    elif num_of_cameras == 3:
        H_base_2 = H2.dot(inverseH(base_H))
        T2 = H_base_2[:3,-1].reshape(3,1)
        R2 = H_base_2[:3,:3]
        M2 = np.hstack((R2, T2))

        return M_base,M1,M2
    elif num_of_cameras == 4:
        H_base_2 = H2.dot(inverseH(base_H))
        T2 = H_base_2[:3,-1].reshape(3,1)
        R2 = H_base_2[:3,:3]
        M2 = np.hstack((R2, T2))

        H_base_3 = H3.dot(inverseH(base_H))
        T3 = H_base_3[:3,-1].reshape(3,1)
        R3 = H_base_3[:3,:3]
        M3 = np.hstack((R3, T3))

        return M_base,M1,M2,M3



#============export a csv file
def toCsv(coord,name):
    np.savetxt(name,coord,delimiter=",",fmt='%10.5f')



#============convert a vector to its skewed symmetric form
def vec2skewMat(vec):
    res = np.zeros((3,3))
    res[0,1],res[0,2],res[1,0],res[1,2],res[2,0],res[2,1] = -vec[2],vec[1],vec[2],-vec[0],-vec[1],vec[0]

    return res


#___________inverse a homogenies transformation matrix
def inverseH(H):
    trans_R = H[:3,:3].T
    trans_T = -trans_R.dot(H[:3,-1])
    inv_H = np.zeros((4,4))
    inv_H[:3,:3] = trans_R
    inv_H[:3,-1] = trans_T
    inv_H[-1,-1] = 1

    return inv_H


def R_t2H(R,T):
    ret = np.zeros((4,4))
    ret[:3,:3],ret[:3,-1] = R,T.reshape(3,)
    ret[-1,:] = np.array([0,0,0,1])

    return ret


def get_RT_mtx(path,Cam_indx,video_resolution):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(path)
    images.sort(key=lambda f: int(re.sub('\D', '', f)))
    count = 0
    for fname in images:
        img = cv2.imread(fname)
        img = cv2.resize(img,video_resolution)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners   
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv2.namedWindow("output", cv2.WINDOW_NORMAL)
            img = cv2.drawChessboardCorners(img, (6,9), corners2,ret)
            cv2.imshow('img',img)
            cv2.waitKey(500)
    
        count += 1
        print(count)

    cv2.destroyAllWindows()

    #=================store camera information
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    np.save('CameraINFO/Cam'+str(Cam_indx)+'_ret.npy',ret)
    np.save('CameraINFO/Cam'+str(Cam_indx)+'_mtx.npy',mtx)
    np.save('CameraINFO/Cam'+str(Cam_indx)+'_dist.npy',dist)
    np.save('CameraINFO/Cam'+str(Cam_indx)+'_rvec.npy',rvecs)
    np.save('CameraINFO/Cam'+str(Cam_indx)+'_tvecs.npy',tvecs)

    return ret,mtx,dist,rvecs,tvecs




def video_loader(fileName,Cam_Indx):
    """
    Cam_Index: CamA/CamB/CamC, depand on how many cameras are used during recording
    """

    if not os.path.exists(baseFilePath + '/Calibration'):
        os.mkdir(baseFilePath + '/Calibration')
    calibratefilepath = baseFilePath + '/Calibration'

    DATADIR_1 = SourceVideoFolder
    datadir =[DATADIR_1]
    video_array = []
    
    for dir in datadir:
        for video in os.listdir(dir):
            if video == fileName:
                
                vidcap = cv2.VideoCapture(os.path.join(dir,video))
                
                vidWidth  = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)  
                vidHeight = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
                video_resolution = (int(vidWidth),int(vidHeight))
                success,image = vidcap.read()
                count = 0
                #success = True

                while success:
                    success,image = vidcap.read()
                    if success:
                        height , width , layers =  image.shape
                        resize = cv2.resize(image, video_resolution) 
                        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        #single_video.append(image)
                        if count < 20:   
                            if not os.path.exists(baseFilePath + '/Calibration/'+Cam_Indx+'_Calibration'):
                                os.mkdir(baseFilePath + '/Calibration/'+Cam_Indx+'_Calibration')                       
                            cv2.imwrite(baseFilePath+'/Calibration/'+Cam_Indx+'_Calibration/frame%d.jpg' %count , image)     # save frame as JPEG file
                            #print(resize.shape)
                        else:
                            break

    
                        print('Read a new frame: ', success)
                        count += 1
                        print(count)
                    else:
                        break
    return video_resolution



#=======================triangulate points
class triangulate:
    """
    ImgPoints: a (frame,#_of_keypoints,#_of_views,3) matrix -> (x,y,prob)
    ProjectMat: a Mx3x4 matrix, M is number of views, each views has its 3x4 projection matrix
    """
    
    def __init__(self,ImgPoints,ProjectMat):
        
        self.ImgPoints = ImgPoints
        self.ProjectMat = ProjectMat

    def solveA(self): 

        if self.ImgPoints.shape[2] != len(self.ProjectMat):
            raise Exceptions('number of views must be equal to number of projection matrix')
        
        N_views = len(self.ProjectMat)
        A = np.zeros((N_views*2,4)) #prepare svd matrix A
        X = np.zeros((self.ImgPoints.shape[0],self.ImgPoints.shape[1],4))

        for i in range(self.ImgPoints.shape[0]): #for each point
            
            for k in range(self.ImgPoints.shape[1]):
                #if the lowest p-value of the obersvations is less than 0.9, skip the calculation
                if float(min(list(self.ImgPoints[i,k,:,-1]))) < -1:
                    continue
                for j in range(N_views): #for each view
                    
                    u,v = self.ImgPoints[i,k,j,0],self.ImgPoints[i,k,j,1] #initialize x,y points
                    
                    for col in range(4):
                        A[j*2+0,col] = u*self.ProjectMat[j,2,col] - self.ProjectMat[j,0,col]
                        A[j*2+1,col] = v*self.ProjectMat[j,2,col] - self.ProjectMat[j,1,col]
     

                U,s,V = np.linalg.svd(A)
                P = V[-1,:] / V[-1,-1]
                #X[i] = P[:-1]
                X[i,k] = P
        
        return X


#=====================pick best angle version
class triangulateTest:
    """
    ImgPoints: a (frame,#_of_keypoints,#_of_views,3) matrix -> (x,y,prob)
    ProjectMat: a Mx3x4 matrix, M is number of views, each views has its 3x4 projection matrix
    """
    
    def __init__(self,ImgPoints,ProjectMat,base_cam):
        
        self.ImgPoints = ImgPoints
        self.ProjectMat = ProjectMat
        self.base_cam = base_cam
    

    def solveA(self): 

        if self.ImgPoints.shape[2] != len(self.ProjectMat):
            raise Exceptions('number of views must be equal to number of projection matrix')
        
        N_views = len(self.ProjectMat)
        A = np.zeros((N_views*2,4)) #prepare svd matrix A
        X = np.zeros((self.ImgPoints.shape[0],self.ImgPoints.shape[1],4))
        #vis_list = {}
        vis_list = []
        
        for i in range(self.ImgPoints.shape[0]): #for each point
            
            for k in range(self.ImgPoints.shape[1]):
                
                v_indx,max_ = -1.0,-1.0
                for view_index in range(self.ImgPoints.shape[2]):
                    if view_index == self.base_cam:
                        continue
                    elif self.ImgPoints[i,k,view_index,2] > max_:
                        max_ = self.ImgPoints[i,k,view_index,-1]
                        v_indx = view_index
                
                #vis_list[(i,k)] = v_indx
                vis_list.append(v_indx)
                
                view_list = [self.base_cam,v_indx]
                
                for j in view_list: #for each view            
                    u,v = self.ImgPoints[i,k,j,0],self.ImgPoints[i,k,j,1] #initialize x,y points
                    
                    for col in range(4):
                        A[j*2+0,col] = u*self.ProjectMat[j,2,col] - self.ProjectMat[j,0,col]
                        A[j*2+1,col] = v*self.ProjectMat[j,2,col] - self.ProjectMat[j,1,col]
        
                A1 = np.zeros((4,4))
                A1[0,:] = A[view_list[0]*2,:]
                A1[1,:] = A[view_list[0]*2+1,:]
                A1[2,:] = A[view_list[1]*2,:]
                A1[3,:] = A[view_list[1]*2+1,:]
                A1 = np.array(A1)
                A1 = A1.reshape((4,4))
                U,s,V = np.linalg.svd(A)
                P = V[-1,:] / V[-1,-1]
                #X[i] = P[:-1]
                X[i,k] = P
        
        return X,vis_list



#=====================pick best angle version and discard
class triangulateFlex:
    """
    ImgPoints: a (frame,#_of_keypoints,#_of_views,3) matrix -> (x,y,prob)
    ProjectMat: a Mx3x4 matrix, M is number of views, each views has its 3x4 projection matrix
    """
    
    def __init__(self,ImgPoints,ProjectMat,base_cam):
        
        self.ImgPoints = ImgPoints
        self.ProjectMat = ProjectMat
        self.base_cam = base_cam
    

    def solveA(self): 

        if self.ImgPoints.shape[2] != len(self.ProjectMat):
            raise Exceptions('number of views must be equal to number of projection matrix')
        
        N_views = len(self.ProjectMat)
        A = np.zeros((N_views*2,4)) #prepare svd matrix A
        X = np.zeros((self.ImgPoints.shape[0],self.ImgPoints.shape[1],4))
        #vis_list = {}
        vis_list = []
        valid_point_list = []
        
        for i in range(self.ImgPoints.shape[0]): #for each point
            
            for k in range(self.ImgPoints.shape[1]):
                
                v_indx,max_ = -1.0,-1.0
                for view_index in range(self.ImgPoints.shape[2]):
                    if view_index == self.base_cam:
                        continue
                    elif self.ImgPoints[i,k,view_index,2] > max_:
                        max_ = self.ImgPoints[i,k,view_index,2]
                        v_indx = view_index
                
                #vis_list[(i,k)] = v_indx
                vis_list.append(v_indx)
                
                #========================remove point module
                if min(self.ImgPoints[i,k,self.base_cam,2],self.ImgPoints[i,k,v_indx,2]) < 0.4:
                    valid_point_list.append(0)
                else:
                    valid_point_list.append(1)
                
                view_list = [self.base_cam,v_indx] #for each frame
                
                for j in view_list: #for each view            
                    u,v = self.ImgPoints[i,k,j,0],self.ImgPoints[i,k,j,1] #initialize x,y points
                    
                    for col in range(4):
                        A[j*2+0,col] = u*self.ProjectMat[j,2,col] - self.ProjectMat[j,0,col]
                        A[j*2+1,col] = v*self.ProjectMat[j,2,col] - self.ProjectMat[j,1,col]
        
                A1 = np.zeros((4,4))
                A1[0,:] = A[view_list[0]*2,:]
                A1[1,:] = A[view_list[0]*2+1,:]
                A1[2,:] = A[view_list[1]*2,:]
                A1[3,:] = A[view_list[1]*2+1,:]
                U,s,V = np.linalg.svd(A1)
                P = V[-1,:] / V[-1,-1]
                #X[i] = P[:-1]
                X[i,k] = P
        
        return X,vis_list,valid_point_list









