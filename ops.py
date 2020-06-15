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
import cv2.aruco as aruco
from create_project import GetVariables
from itertools import combinations
from pykalman import KalmanFilter
import statistics

configVariables = GetVariables()

cam_names = configVariables[1]
useCheckerboardVid = configVariables[9]
baseProjectPath = configVariables[13]



num_of_cameras = len(cam_names)
if useCheckerboardVid == True:
    SourceVideoFolder = baseProjectPath + '/Intermediate/CheckerboardUndistorted'
else: 
    SourceVideoFolder = baseProjectPath + '/Intermediate/Trimmed'

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
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)*25

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
        ret, corners = cv2.findChessboardCorners(gray, (6,9),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv2.namedWindow("output", cv2.WINDOW_NORMAL)
            img = cv2.drawChessboardCorners(img, (6,9), corners2,ret)
            cv2.imshow('img',img)
            cv2.waitKey(50)
    
        count += 1
        print(count)

    cv2.destroyAllWindows()

    #=================store camera information
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    if not os.path.exists(baseProjectPath + '/Calibration/CameraINFO'):
        os.mkdir(baseProjectPath + '/Calibration/CameraINFO')
    np.save(baseProjectPath+'/Calibration/CameraINFO/'+str(Cam_indx)+'_ret.npy',ret)
    np.save(baseProjectPath+'/Calibration/CameraINFO/'+str(Cam_indx)+'_mtx.npy',mtx)
    np.save(baseProjectPath+'/Calibration/CameraINFO/'+str(Cam_indx)+'_dist.npy',dist)
    np.save(baseProjectPath+'/Calibration/CameraINFO/'+str(Cam_indx)+'_rvec.npy',rvecs)
    np.save(baseProjectPath+'/Calibration/CameraINFO/'+str(Cam_indx)+'_tvecs.npy',tvecs)

    return ret,mtx,dist,rvecs,tvecs




def video_loader(fileName,Cam_Indx):
    """
    Cam_Index: CamA/CamB/CamC, depand on how many cameras are used during recording
    """
    

    if not os.path.exists(baseProjectPath + '/Calibration'):
        os.mkdir(baseProjectPath + '/Calibration')
    calibratefilepath = baseProjectPath + '/Calibration'

    DATADIR_1 = SourceVideoFolder
    datadir =[DATADIR_1]
    video_array = []
    
    for dir in datadir:
        for video in os.listdir(dir):
            print(video)
            print(fileName)
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
                            if not os.path.exists(baseProjectPath + '/Calibration/'+Cam_Indx+'_Calibration'):
                                os.mkdir(baseProjectPath + '/Calibration/'+Cam_Indx+'_Calibration')                       
                            cv2.imwrite(baseProjectPath+'/Calibration/'+Cam_Indx+'_Calibration/frame%d.jpg' %count , image)     # save frame as JPEG file
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
        N_Combinations = len(list(combinations(range(N_views),2)))
        Q =[]
        c = np.zeros((N_views*2,4)) #prepare svd matrix A
        X = np.zeros((self.ImgPoints.shape[0],self.ImgPoints.shape[1],4))
        for i in range(self.ImgPoints.shape[0]): #for each point
            T=[]   
            for k in range(self.ImgPoints.shape[1]):
                #if the lowest p-value of the obersvations is less than 0.9, skip the calculation
                if float(min(list(self.ImgPoints[i,k,:,-1]))) < -1:
                    continue
                for j in range(N_views): #for each view
                    
                    u,v = self.ImgPoints[i,k,j,0],self.ImgPoints[i,k,j,1] #initialize x,y points
                    
                    for col in range(4):
                        c[j*2+0,col] = u*self.ProjectMat[j,2,col] - self.ProjectMat[j,0,col]
                        c[j*2+1,col] = v*self.ProjectMat[j,2,col] - self.ProjectMat[j,1,col]
                U,s,V = np.linalg.svd(c)
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


def aruco_detect(path,Cam_indx,video_resolution):

    

    images = glob.glob(path)
    count = 0
    for fname in images:
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
        img = cv2.imread(fname)
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # detector parameters can be set here (List of detection parameters[3])
        parameters = cv2.aruco.DetectorParameters_create()
        parameters.adaptiveThreshConstant = 10

        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
        rvec_d = {}
        tvec_d = {}
        font = cv2.FONT_HERSHEY_SIMPLEX
        if np.all(ids != None):

            # estimate pose of each marker and return the values
            # rvet and tvec-different from camera coefficients
            rvec, tvec,_ = cv2.aruco.estimatePoseSingleMarkers(corners,0.1, K,dist)
            if count == 0:
                for i in range(len(ids)):
                    #ret[ids[i]] = {'rvec':list(rvec[i].reshape((-1))),'tvec':list(tvec[i].reshape((-1)))}
                    rvec_d[int(ids[i])] = rvec[i].reshape((-1)).tolist()
                    tvec_d[int(ids[i])] = tvec[i].reshape((-1)).tolist()

                # np.save('ARUCO_POSE/CamA_t.npy',tvec)
                # np.save('ARUCO_POSE/CamA_r.npy',rvec)

            #(rvec-tvec).any() # get rid of that nasty numpy value array error

            for i in range(0, ids.size):
                # draw axis for the aruco markers
                cv2.aruco.drawAxis(img, K, dist, rvec[i], tvec[i], 0.1)

            # draw a square around the markers
            cv2.aruco.drawDetectedMarkers(img, corners)


            # code to show ids of the marker found
            strg = ''
            for i in range(0, ids.size):
                strg += str(ids[i][0])+', '

            cv2.putText(img, "Id: " + strg, (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
            count += 1

            cv2.imshow('frame',img)
            key = cv2.waitKey(3000)

            cv2.destroyAllWindows()



            return K,dist,rvec_d,tvec_d
    
        else:
            # code to show 'No Ids' when no markers are found
            cv2.putText(img, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)



def charuco_detect(path,Cam_index,video_resolution):
    
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    board = aruco.CharucoBoard_create(7, 5, 1, .8, aruco_dict)
    images = glob.glob(path)


    def read_chessboards(images):
        """
        Charuco base pose estimation.
        """
        print("POSE ESTIMATION STARTS:")
        allCorners = []
        allIds = []
        decimator = 0
        # SUB PIXEL CORNER DETECTION CRITERION
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

        for im in images:
            print("=> Processing image {0}".format(im))
            frame = cv2.imread(im)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

            if len(corners)>0:
                # SUB PIXEL DETECTION
                for corner in corners:
                    cv2.cornerSubPix(gray, corner,
                                    winSize = (3,3),
                                    zeroZone = (-1,-1),
                                    criteria = criteria)
                res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,gray,board)
                if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0:
                    allCorners.append(res2[1])
                    allIds.append(res2[2])

            decimator+=1

        imsize = gray.shape
        return allCorners,allIds,imsize
    
    def calibrate_camera(allCorners,allIds,imsize):
        """
        Calibrates the camera using the dected corners.
        """
        print("CAMERA CALIBRATION")

        cameraMatrixInit = np.array([[ 1000.,    0., imsize[0]/2.],
                                    [    0., 1000., imsize[1]/2.],
                                    [    0.,    0.,           1.]])

        distCoeffsInit = np.zeros((5,1))
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
        #flags = (cv2.CALIB_RATIONAL_MODEL)
        (ret, camera_matrix, distortion_coefficients0,
        rotation_vectors, translation_vectors,
        stdDeviationsIntrinsics, stdDeviationsExtrinsics,
        perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                        charucoCorners=allCorners,
                        charucoIds=allIds,
                        board=board,
                        imageSize=imsize,
                        cameraMatrix=cameraMatrixInit,
                        distCoeffs=distCoeffsInit,
                        flags=flags,
                        criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

        return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors

    allCorners,allIds,imsize=read_chessboards(images)
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(allCorners,allIds,imsize)
    tvecs = np.array(tvecs)
    mtx = np.array(mtx)
    rvecs = np.array(rvecs)


    return mtx,dist,rvecs,tvecs







