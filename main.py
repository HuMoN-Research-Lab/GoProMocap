import numpy as np
import matplotlib.pyplot as plt
import tkinter
import os
import cv2
import pickle
import csv
#from visulize_with_out_head import Vis
from visualize import Vis
from scipy.optimize import least_squares
import time
from scipy.sparse import lil_matrix
import subprocess
from create_project import create_project
#=========================Create Folders for project
configVariables = create_project()
from EditVideoRunOpandDLC import getCameraParams, concatVideos, undistortVideos,trimVideos, runDeepLabCut,runOpenPose, Parse_Openpose, checkerBoardUndistort
from Filters import smoothOpenPose, kalman, butterFilt
from ops import SBA,toCsv,vec2skewMat,inverseH,R_t2H,get_RT_mtx,video_loader,get_TransMat,triangulate,triangulateFlex,triangulateTest,aruco_detect,charuco_detect

subject = configVariables[0]
cam_names = configVariables[1]
base_Cam_Index = configVariables[2]
Len_of_frame = configVariables[3]
start_frame = configVariables[4]
include_DLC = configVariables[5]
include_OpenPoseFace = configVariables[6]
include_OpenPoseHands = configVariables[7]
include_OpenPoseSkeleton = configVariables[8]
baseFilePath = configVariables[13]
baseProjectPath = configVariables[12] 
calibrateCameras = configVariables[10]
useCheckerboardVid = configVariables[9]
num_of_cameras = int(configVariables[14])

##################needs update the variables here###########################


rawData = baseFilePath+'/Raw'
checkerVideoFolder = rawData+'/Checkerboard'
rawVideoFolder = rawData+'/RawGoProData'
calibrationFilePath = baseProjectPath +'/'+subject+'/Calibration'
cameraParamsFilePath = calibrationFilePath +'/CameraParams'
calibVideoFilepath = calibrationFilePath +'/CalibrationVideos'
interfilepath = baseFilePath + '/Intermediate'
videoOutputFilepath = interfilepath+'/VideoOutput'
openposeRawFilepath = interfilepath + '/OpenPoseRaw'
DLCfilepath = interfilepath + '/DeepLabCut'
undistortedFilepath = interfilepath + '/Undistorted'
combinedFilepath = interfilepath+'/CombinedVideo'
processedFilePath = baseFilePath +'/Processed'
trimFilepath = interfilepath +'/Trimmed'
openposeOutputFilepath = interfilepath + '/OpenPoseOutput'



if calibrateCameras:
    getCameraParams(calibrationFilePath)
#===========================Concat,Undistort and Trim Videos 
#concatVideos(rawVideoFolder,combinedFilepath)
undistortVideos(combinedFilepath,cameraParamsFilePath,undistortedFilepath)
if useCheckerboardVid == True:
    if not os.path.exists(interfilepath + '/CheckerboardUndistorted'):
        os.mkdir(interfilepath + '/CheckerboardUndistorted')
    checkerUndistortFilepath = interfilepath + '/CheckerboardUndistorted'
    checkerBoardUndistort(checkerVideoFolder,checkerUndistortFilepath)

trimVideos(undistortedFilepath,trimFilepath)

#==========================Run deeplabcut and parse through output
if include_DLC == True:
    runDeepLabCut(trimFilepath,DLCfilepath)

#==========================Run OpenPose and parse through output
if include_OpenPoseFace == True or include_OpenPoseHands ==True or include_OpenPoseSkeleton == True:
    runOpenPose(undistortedFilepath,videoOutputFilepath,openposeRawFilepath)
    points_inFrame = Parse_Openpose(openposeRawFilepath,openposeOutputFilepath)
    smoothOpenPose(openposeOutputFilepath)

#points_inFrame =67
butterFilt(openposeOutputFilepath)
#========================Get source video
if useCheckerboardVid == True:
    SourceVideoFolder = baseFilePath + '/Intermediate/CheckerboardUndistorted'
else: 
    SourceVideoFolder = trimFilepath

fullVideoFolder = trimFilepath #Always need to use this
#======================== Set up names for videos
cam1 = cam_names[0]
cam2 = cam_names[1]

if num_of_cameras ==2:
    Source_video_List = [[cam1+'.mp4',cam1],[cam2+'.mp4',cam2]]
if num_of_cameras ==3: 
    cam3 = cam_names[2]
    Source_video_List= [[cam1+'.mp4',cam1],[cam2+'.mp4',cam2],[cam3+'.mp4',cam3]]
if num_of_cameras ==4:
    cam3 = cam_names[2]
    cam4 = cam_names[3]
    Source_video_List= [[cam1+'.mp4',cam1],[cam2+'.mp4',cam2],[cam3+'.mp4',cam3],[cam4+'.mp4',cam4]]

#=====================Get files for dlc and openpose data 
rootOPFolder = openposeOutputFilepath+'/'
rootDLCFolder = DLCfilepath +'/DLCnpy'

if num_of_cameras ==2:
    Pixel_coord_FIlE_List = [[rootOPFolder+'KalmanOP_'+cam1+'.npy',cam1],
                             [rootOPFolder+'KalmanOP_'+cam2+'.npy',cam2]]

    Pixel_coord_FIlE_List_include_DLC = [[rootOPFolder+'KalmanOP_'+cam1+'.npy',rootDLCFolder+'dlc_'+cam1+'.npy',cam1],
                                          [rootOPFolder+'KalmanOP_'+cam2+'.npy',rootDLCFolder+'dlc_'+cam2+'.npy',cam2]]
if num_of_cameras ==3:
    Pixel_coord_FIlE_List = [[rootOPFolder+'KalmanOP_'+cam1+'.npy',cam1],
                             [rootOPFolder+'KalmanOP_'+cam2+'.npy',cam2],
                             [rootOPFolder+'KalmanOP_'+cam3+'.npy',cam3]]
                                                         

    Pixel_coord_FIlE_List_include_DLC = [[rootOPFolder+'KalmanOP_'+cam1+'.npy',rootDLCFolder+'dlc_'+cam1+'.npy',cam1],
                                          [rootOPFolder+'KalmanOP_'+cam2+'.npy',rootDLCFolder+'dlc_'+cam1+'.npy',cam2],
                                          [rootOPFolder+'KalmanOP_'+cam3+'.npy',rootDLCFolder+'dlc_'+cam3+'.npy',cam3]]
if num_of_cameras ==4:
    Pixel_coord_FIlE_List = [[rootOPFolder+'KalmanOP_'+cam1+'.npy',cam1],
                             [rootOPFolder+'KalmanOP_'+cam2+'.npy',cam2],
                             [rootOPFolder+'KalmanOP_'+cam3+'.npy',cam3],
                             [rootOPFolder+'KalmanOP_'+cam4+'.npy',cam4]]
                                                         

    Pixel_coord_FIlE_List_include_DLC = [[rootOPFolder+'KalmanOP_'+cam1+'.npy',rootDLCFolder+'dlc_'+cam1+'.npy',cam1],
                                          [rootOPFolder+'KalmanOP_'+cam2+'.npy',rootDLCFolder+'dlc_'+cam2+'.npy',cam2],
                                          [rootOPFolder+'KalmanOP_'+cam3+'.npy',rootDLCFolder+'dlc_'+cam3+'.npy',cam3],
                                          [rootOPFolder+'KalmanOP_'+cam4+'.npy',rootDLCFolder+'dlc_'+cam4+'.npy',cam4]]

if Len_of_frame == '-1':
    frameLengthAllCam = [] #create variable to stoe frame length
    for dir in [fullVideoFolder]:
        for video in os.listdir(dir):
            vidcap = cv2.VideoCapture(os.path.join(dir,video))
            frameLengthOneCam = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            frameLengthAllCam.append(frameLengthOneCam)
    
    minVideo = frameLengthAllCam.index(min(frameLengthAllCam))  
    shortestVideo = frameLengthAllCam[minVideo]
    Len_of_frame = shortestVideo

if num_of_cameras == 2:
    base_cam = {cam_names[0]:0,cam_names[1]:1}
if num_of_cameras == 3:
    base_cam = {cam_names[0]:0,cam_names[1]:1, cam_names[2]:2}
if num_of_cameras == 4:
    base_cam = {cam_names[0]:0,cam_names[1]:1,cam_names[2]:2,cam_names[3]:3}

    
    
#==================load image from videos 
for path in Source_video_List:
    video_resolution = video_loader(path[0],path[1])


#==================load pixel data to a dictionary
pixelCoord = {}
if include_DLC:
    for path in Pixel_coord_FIlE_List_include_DLC:
        skeleton = np.load(path[0],allow_pickle = True)[start_frame:start_frame+Len_of_frame,:,:]
        ball = np.load(path[1],allow_pickle = True)[start_frame:start_frame+Len_of_frame,:]
        ball = ball.astype(float)
        ball = ball.reshape((-1,3,3))
        pixelC = np.concatenate((skeleton,ball),axis=-2)
        pixelCoord[path[-1]] = pixelC
        #pixelCoord[path[-1]] = pixelCoord[path[-1]][start_frame:start_frame+Len_of_frame,:,:]

else:
    for path in Pixel_coord_FIlE_List:
        pixelCoord[path[1]] = np.load(path[0],allow_pickle = True)
        pixelCoord[path[1]] = pixelCoord[path[1]][start_frame:start_frame+Len_of_frame,:,:]




#================== calibrate the cameras


#==================load charucoboard pixel data

def calibration(calibrate_frame,points_perFrame):

    K_CamA,A_dist,A_rvecs,A_tvecs,A_corners = charuco_detect('Calibration/CamA_Calibration/*.jpg','A',video_resolution)#this function in ops line 437
    tvec_CamA,rvec_CamA = A_tvecs[1],A_rvecs[1]#tvec/rvec generated 1 per image in the calibration folder, so we only need one of them
    RoMat_A, _ = cv2.Rodrigues(rvec_CamA)#convert (3,1) rvec to (3,3) rotation matrix
    H_CamA = R_t2H(RoMat_A,tvec_CamA)#this funciton in ops line 87, combine rotation matrix and tvec into a Homogeniens transformation matrix
    A_corners = A_corners.reshape((2,points_perFrame,2))


    K_CamB,B_dist,B_rvecs,B_tvecs,B_corners = charuco_detect('Calibration/CamB_Calibration/*.jpg','B',video_resolution)
    tvec_CamB,rvec_CamB = B_tvecs[1],B_rvecs[1]
    RoMat_B, _ = cv2.Rodrigues(rvec_CamB) #convert 
    H_CamB = R_t2H(RoMat_B,tvec_CamB)
    B_corners = B_corners.reshape((2,points_perFrame,2))
    
    if num_of_cameras > 2:
        #=====================it seems not all cornors are detected in view C
        K_CamC,C_dist,C_rvecs,C_tvecs,C_corners = charuco_detect('Calibration/CamC_Calibration/*.jpg','C',video_resolution)
        tvec_CamC,rvec_CamC = C_tvecs[1],C_rvecs[1]
        RoMat_C, _ = cv2.Rodrigues(rvec_CamC) #convert 
        H_CamC = R_t2H(RoMat_C,tvec_CamC)
        C_corners = C_corners.reshape((2,points_perFrame,2))
    
    if num_of_cameras > 3:
        #=====================it seems not all cornors are detected in view C
        K_CamD,D_dist,D_rvecs,D_tvecs,D_corners = charuco_detect('Calibration/CamC_Calibration/*.jpg','D',video_resolution)
        tvec_CamD,rvec_CamD = D_tvecs[1],D_rvecs[1]
        RoMat_D, _ = cv2.Rodrigues(rvec_CamD) #convert 
        H_CamD = R_t2H(RoMat_D,tvec_CamD)
        D_corners = D_corners.reshape((2,points_perFrame,2))

    if optimize_projection_matrix:
        if num_of_cameras == 2:
            MA,MB = get_TransMat(H_CamA,H_CamB)
            PA,PB = np.dot(K_CamA,MA),np.dot(K_CamB,MB)
            Proj_points = np.stack((A_corners,B_corners),axis = 2)#project points format should be (frame,#_of_keypoints,#_of_views,3), in this case (2,24,2,2)
            Proj_Mat = np.stack((PA,PB),axis=0)
            BA_points2D = np.stack((A_corners,B_corners),axis = 0)
            input_param = np.hstack((PA.ravel(),PB.ravel()))
         elif num_of_cameras == 3:
            MA,MB,MC = get_TransMat(H_CamA,H_CamB,H_CamC)
            PA,PB,PC = np.dot(K_CamA,MA),np.dot(K_CamB,MB),np.dot(K_CamC,MC)
            Proj_points = np.stack((A_corners,B_corners,C_corners),axis = 2)#project points format should be (frame,#_of_keypoints,#_of_views,3), in this case (2,24,2,2)
            Proj_Mat = np.stack((PA,PB,PC),axis=0)
            BA_points2D = np.stack((A_corners,B_corners,C_corners),axis = 0)
            input_param = np.hstack((PA.ravel(),PB.ravel(),PC.ravel()))
         elif num_of_cameras == 4:
            MA,MB,MC,MD = get_TransMat(H_CamA,H_CamB,H_CamC,H_Cam_D)
            PA,PB,PC = np.dot(K_CamA,MA),np.dot(K_CamB,MB),np.dot(K_CamC,MC),np.dot(K_CamD,MD)
            Proj_points = np.stack((A_corners,B_corners,C_corners,D_corners),axis = 2)#project points format should be (frame,#_of_keypoints,#_of_views,3), in this case (2,24,2,2)
            Proj_Mat = np.stack((PA,PB,PC,PD),axis=0)
            BA_points2D = np.stack((A_corners,B_corners,C_corners,D_corners),axis = 0)
            input_param = np.hstack((PA.ravel(),PB.ravel(),PC.ravel(),PD.ravel())) 


        coords = triangulate(Proj_points,Proj_Mat).solveA()#tranigulate points
        coords = coords[:,:,:-1]

    input_points = coords.reshape((-1,))
    ba_input = np.hstack((input_points,input_param))

    VIS_cam_List = [0]*calibrate_frame*points_perFrame
    refined_points,refined_Pmat = SBA(calibrate_frame,BA_points2D,ba_input,VIS_cam_List,points_perFrame,'A')


    l = len(refined_Pmat)//num_of_cameras
    if num_of_cameras == 2:
        PA,PB = (refined_Pmat[:l].reshape((3,4)),refined_Pmat[l:].reshape((3,4)))
        return (PA,PB)
    elif num_of_cameras == 3:
        PA,PB,PC = (refined_Pmat[:l].reshape((3,4)),refined_Pmat[l:2*l].reshape((3,4),refined_Pmat[2*l:].reshape((3,4)))
        return (PA,PB,PC)
    elif num_of_cameras == 4:
        PA,PB,PC,PC = (refined_Pmat[:l].reshape((3,4)),refined_Pmat[l:2*l].reshape((3,4),refined_Pmat[2*l:3*l].reshape((3,4),refined_Pmat[3*l:4*l].reshape((3,4)))
        return (PA,PB,PC,PD)
                                                                                   
                                                                                   
                                                                                   

def run(ProjectMatix):
    #=================load skeleton 2d keypoints 
    pixelCoord = {}
    if include_ball:
        for path in Pixel_coord_FIlE_List_include_ball:
            skeleton = np.load(path[0],allow_pickle = True)[start_frame:start_frame+Len_of_frame,:,:]
            ball = np.load(path[1],allow_pickle = True)[start_frame:start_frame+Len_of_frame,:]
            ball = ball.astype(float)
            ball = ball.reshape((-1,3,3))
            pixelC = np.concatenate((skeleton,ball),axis=-2)
            pixelCoord[path[-1]] = pixelC
            #pixelCoord[path[-1]] = pixelCoord[path[-1]][start_frame:start_frame+Len_of_frame,:,:]

    else:
        for path in Pixel_coord_FIlE_List:
            pixelCoord[path[1]] = np.load(path[0],allow_pickle = True)
            pixelCoord[path[1]] = pixelCoord[path[1]][start_frame:start_frame+Len_of_frame,:,:]



    #==================concatnate data 
    if num_of_cameras == 2:
        if base_Cam_Index == 'A':
            Proj_points = np.stack((pixelCoord['CamA'],pixelCoord['CamB']),axis = 2)
            Proj_Mat = np.stack((ProjectMatix[0],ProjectMatix[1]),axis=0)
    
        elif base_Cam_Index == 'B':
            Proj_points = np.stack((pixelCoord['CamB'],pixelCoord['CamA']),axis = 2)
            Proj_Mat = np.stack((ProjectMatix[1],ProjectMatix[0]),axis=0)
    
    elif num_of_cameras == 3:
        if base_Cam_Index == 'A':
            Proj_points = np.stack((pixelCoord['CamA'],pixelCoord['CamB'],pixelCoord['CamC']),axis = 2)
            Proj_Mat = np.stack((ProjectMatix[0],ProjectMatix[1],ProjectMatix[2]),axis=0)
    
        elif base_Cam_Index == 'B':
            Proj_points = np.stack((pixelCoord['CamB'],pixelCoord['CamA'],pixelCoord['CamC']),axis = 2)
            Proj_Mat = np.stack((ProjectMatix[1],ProjectMatix[0],ProjectMatix[2]),axis=0)
        
        elif base_Cam_Index == 'C':
            Proj_points = np.stack((pixelCoord['CamC'],pixelCoord['CamA'],pixelCoord['CamB']),axis = 2)
            Proj_Mat = np.stack((ProjectMatix[2],ProjectMatix[0],ProjectMatix[1]),axis=0)
    

    elif num_of_cameras == 4:
        if base_Cam_Index == 'A':
            Proj_points = np.stack((pixelCoord['CamA'],pixelCoord['CamB'],pixelCoord['CamC'],pixelCoord['CamD']),axis = 2)
            Proj_Mat = np.stack((ProjectMatix[0],ProjectMatix[1],ProjectMatix[2],ProjectMatix[3]),axis=0)
    
        elif base_Cam_Index == 'B':
            Proj_points = np.stack((pixelCoord['CamB'],pixelCoord['CamA'],pixelCoord['CamC'],pixelCoord['CamD']),axis = 2)
            Proj_Mat = np.stack((ProjectMatix[1],ProjectMatix[0],ProjectMatix[2],ProjectMatix[3]),axis=0)
        
        elif base_Cam_Index == 'C':
            Proj_points = np.stack((pixelCoord['CamC'],pixelCoord['CamA'],pixelCoord['CamB'],pixelCoord['CamD']),axis = 2)
            Proj_Mat = np.stack((ProjectMatix[2],ProjectMatix[0],ProjectMatix[1],ProjectMatix[3]),axis=0)

        elif base_Cam_Index == 'D':
            Proj_points = np.stack((pixelCoord['CamD'],pixelCoord['CamA'],pixelCoord['CamB'],pixelCoord['CamC']),axis = 2)
            Proj_Mat = np.stack((ProjectMatix[3],ProjectMatix[0],ProjectMatix[1],ProjectMatix[2]),axis=0)


    coords = triangulate(Proj_points,Proj_Mat).solveA()#tranigulate points
    coords = coords[:,:,:-1]
    np.save(SAVE_FOLDER+'3Dcoords.npy',coords)
    print('3D coordinates saved'
          
    return coords
                                                                               
                                                                                   


# input_points = coords.reshape((-1,))


# ba_input = np.hstack((input_points,input_param))


# print("optimization started")

# #C,M = SBA(Len_of_frame,Proj_Mat,BA_points2D,ba_input,VIS_cam_List)
# print('coords shape',coords.shape)
# np.save(processedFilePath+'reconstructed'+'.npy',coords)
# #np.save(SAVE_FOLDER+'out_optimized.npy',C)
# print('save sussesful')

if __name__ == "__main__":
    ProjectMatix = calibration(2,24)
    run(ProjectMatix)

    np.save(baseFilePath+'/Processed/DataPoints3D.npy',coords)
    print('save sussesful')


    if num_of_cameras == 3:
        Vis(SourceVideoFolder+'/'+Source_video_List[0][0],SourceVideoFolder+'/'+Source_video_List[1][0],SourceVideoFolder+'/'+Source_video_List[2][0],coords).display()

    elif num_of_cameras == 2:
        #Vis(SourceVideoFolder+'/'+Source_video_List[1][0],SourceVideoFolder+'/'+Source_video_List[0][0],None,C).display()
        Vis(SourceVideoFolder+'/'+Source_video_List[1][0],SourceVideoFolder+'/'+Source_video_List[0][0],None,coords).display()

    elif num_of_cameras == 4:
        Vis(SourceVideoFolder+'/'+Source_video_List[0][0],SourceVideoFolder+'/'+Source_video_List[1][0],SourceVideoFolder+'/'+Source_video_List[2][0],coords).display()
    




