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



K_CamA,A_dist,A_rvecs,A_tvecs = charuco_detect(baseFilePath+'/Calibration/'+cam1+'_Calibration/*jpg',cam1,video_resolution)
# RoMat_A, _ = cv2.Rodrigues(np.array(A_rvecs[0]))
# tvec_CamA = np.array(A_tvecs[0])
# H_mark2CamA = R_t2H(RoMat_A,tvec_CamA)
# H_CamA = inverseH(H_mark2CamA)
tvec_CamA,rvec_CamA = A_tvecs[0],A_rvecs[0]
RoMat_A, _ = cv2.Rodrigues(rvec_CamA)
H_CamA = R_t2H(RoMat_A,tvec_CamA)


K_CamB,B_dist,B_rvecs,B_tvecs = charuco_detect(baseFilePath+'/Calibration/'+cam2+'_Calibration/*jpg',cam2,video_resolution)
# RoMat_B, _ = cv2.Rodrigues(np.array(B_rvecs[0]))
# tvec_CamB = np.array(B_tvecs[0])
# H_mark2CamB = R_t2H(RoMat_B,tvec_CamB)
# H_CamB = inverseH(H_mark2CamB)

tvec_CamB,rvec_CamB = B_tvecs[0],B_rvecs[0]
RoMat_B, _ = cv2.Rodrigues(rvec_CamB) #convert 
H_CamB = R_t2H(RoMat_B,tvec_CamB)


if num_of_cameras > 2:

    K_CamC,C_dist,C_rvecs,C_tvecs = charuco_detect(baseFilePath+'/Calibration/'+cam3+'_Calibration/*jpg',cam3,video_resolution)
    RoMat_C, _ = cv2.Rodrigues(np.array(C_rvecs[0]))
    tvec_CamC = np.array(C_tvecs[0])
    H_mark2CamC = R_t2H(RoMat_C,tvec_CamC)
    H_CamC = inverseH(H_mark2CamC)
    

if num_of_cameras > 3:
    K_CamD,D_dist,D_rvecs,D_tvecs = charuco_detect(baseFilePath+'/Calibration/'+cam4+'_Calibration/*jpg',cam4,video_resolution)
    RoMat_D, _ = cv2.Rodrigues(np.array(D_rvecs[0]))
    tvec_CamD = np.array(D_tvecs[0])
    H_mark2CamD = R_t2H(RoMat_D,tvec_CamD)
    H_CamD = inverseH(H_mark2CamD)






#=====================prepare proj matrix and pixel coords

#def get_ProjPoint_ProjMat(base_Cam_Index,num_of_cameras)------to do

Proj_points = None
Proj_Mat = None

if num_of_cameras == 4:
    if base_Cam_Index == cam1:
        MA,MB,MC,MD = get_TransMat(H_CamA,H_CamB,H_CamC,H_CamD)
        PA,PB,PC,PD = np.dot(K_CamA,MA),np.dot(K_CamB,MB),np.dot(K_CamC,MC),np.dot(K_CamD,MD)
        Proj_points = np.stack((pixelCoord[cam1],pixelCoord[cam2],pixelCoord[cam3],pixelCoord[cam4]),axis = 2)
        Proj_Mat = np.stack((PA,PB,PC,PD),axis=0)

    elif base_Cam_Index == cam2:
        MB,MA,MC,MD = get_TransMat(H_CamB,H_CamA,H_CamC,H_CamD)
        PB,PA,PC,PD = np.dot(K_CamB,MB),np.dot(K_CamA,MA),np.dot(K_CamC,MC),np.dot(K_CamD,MD)
        Proj_points = np.stack((pixelCoord[cam2],pixelCoord[cam1],pixelCoord[cam3],pixelCoord[cam4]),axis = 2)
        Proj_Mat = np.stack((PB,PA,PC,PD),axis=0)

    elif base_Cam_Index == cam3:
        MC,MA,MB,MD = get_TransMat(H_CamC,H_CamA,H_CamB,H_CamD)
        PC,PA,PB,PD = np.dot(K_CamC,MC),np.dot(K_CamA,MA),np.dot(K_CamB,MB),np.dot(K_CamD,MD)
        Proj_points = np.stack((pixelCoord[cam3],pixelCoord[cam1],pixelCoord[cam2],pixelCoord[cam4]),axis = 2)
        Proj_Mat = np.stack((PC,PA,PB,PD),axis=0)

    elif base_Cam_Index == cam4:
        MD,MA,MB,MC = get_TransMat(H_CamD,H_CamA,H_CamB,H_CamC)
        PD,PA,PB,PC = np.dot(K_CamD,MD),np.dot(K_CamA,MA),np.dot(K_CamB,MB),np.dot(K_CamC,MC)
        Proj_points = np.stack((pixelCoord[cam4],pixelCoord[cam1],pixelCoord[cam2],pixelCoord[cam3]),axis = 2)
        Proj_Mat = np.stack((PD,PA,PB,PC),axis=0)
    
    BA_points2D = np.stack((pixelCoord[cam1][:,:25,:-1],pixelCoord[cam2][:,:25,:-1],pixelCoord[cam3][:,:25,:-1],pixelCoord[cam4][:,:25,:-1]),axis = 0)
    input_param = np.hstack((Proj_Mat[0].ravel(),Proj_Mat[1].ravel(),Proj_Mat[2].ravel(),Proj_Mat[3].ravel()))



elif num_of_cameras == 3:
    if base_Cam_Index == cam1:
        MA,MB,MC = get_TransMat(H_CamA,H_CamB,H_CamC)
        PA,PB,PC = np.dot(K_CamA,MA),np.dot(K_CamB,MB),np.dot(K_CamC,MC)
        Proj_points = np.stack((pixelCoord[cam1],pixelCoord[cam2],pixelCoord[cam3]),axis = 2)
        Proj_Mat = np.stack((PA,PB,PC),axis=0)
    
    elif base_Cam_Index == cam2:
        MB,MA,MC = get_TransMat(H_CamB,H_CamA,H_CamC)
        PB,PA,PC = np.dot(K_CamB,MB),np.dot(K_CamA,MA),np.dot(K_CamC,MC)
        Proj_points = np.stack((pixelCoord[cam2],pixelCoord[cam1],pixelCoord[cam3]),axis = 2)
        Proj_Mat = np.stack((PB,PA,PC),axis=0)
    
    elif base_Cam_Index == cam3:
        MC,MA,MB = get_TransMat(H_CamC,H_CamA,H_CamB)
        PC,PA,PB = np.dot(K_CamC,MC),np.dot(K_CamA,MA),np.dot(K_CamB,MB)
        Proj_points = np.stack((pixelCoord[cam3],pixelCoord[cam1],pixelCoord[cam2]),axis = 2)
        Proj_Mat = np.stack((PC,PA,PB),axis=0)
    
    BA_points2D = np.stack((pixelCoord[cam1][:,:25,:-1],pixelCoord[cam2][:,:25,:-1],pixelCoord[cam3][:,:25,:-1]),axis = 0)
    input_param = np.hstack((Proj_Mat[0].ravel(),Proj_Mat[1].ravel(),Proj_Mat[2].ravel()))
    
elif num_of_cameras == 2:
    if base_Cam_Index == cam1:
        MA,MB = get_TransMat(H_CamA,H_CamB)
        PA,PB = np.dot(K_CamA,MA),np.dot(K_CamB,MB)
        Proj_points = np.stack((pixelCoord[cam1],pixelCoord[cam2]),axis = 2)
        Proj_Mat = np.stack((PA,PB),axis=0)
    
    elif base_Cam_Index == cam2:
        MB,MA = get_TransMat(H_CamB,H_CamA)
        PB,PA = np.dot(K_CamB,MB),np.dot(K_CamA,MA)
        Proj_points = np.stack((pixelCoord[cam2],pixelCoord[cam1]),axis = 2)
        Proj_Mat = np.stack((PB,PA),axis=0)
    
    BA_points2D = np.stack((pixelCoord[cam1][:,:25,:-1],pixelCoord[cam2][:,:25,:-1]),axis = 0)
    input_param = np.hstack((Proj_Mat[0].ravel(),Proj_Mat[1].ravel()))

print('Proj points shape:',Proj_points.shape)
coords_temp,VIS_cam_List = triangulateTest(Proj_points,Proj_Mat,base_cam[base_Cam_Index]).solveA()
coords = triangulate(Proj_points,Proj_Mat).solveA()#tranigulate points
coords = coords[:,:,:-1]

print('coords shape:',coords.shape)


# print('valid point length',len(valid_point_list))
# print('valid',valid_point_list)
# print('vis cam',VIS_cam_List)
# valid_point, invalid_point = 0,0
# for num in valid_point_list:
#     if num == 0:
#         invalid_point += 1
#     else:
#         valid_point += 1

# print('valid:',valid_point,'invalid:',invalid_point)


#===========sparse bundle adjustment
# if include_DLC:
#     ball_points = coords[:,-1,:].reshape((-1,3,3))
#     skeleton_points = coords[:,:-3,:]


input_points = coords.reshape((-1,))


ba_input = np.hstack((input_points,input_param))


print("optimization started")

#C,M = SBA(Len_of_frame,Proj_Mat,BA_points2D,ba_input,VIS_cam_List)
print('coords shape',coords.shape)
np.save(processedFilePath+'reconstructed'+'.npy',coords)
#np.save(SAVE_FOLDER+'out_optimized.npy',C)
print('save sussesful')

# if include_DLC:
#     l = len(M)//num_of_cameras
    
#     if num_of_cameras == 2:
#         P1,P2 = (M[:l].reshape((3,4)),M[l:].reshape((3,4)))
#         Proj_points = Proj_points[:,-1:,:,:]
#         Proj_Mat = np.stack((P1,P2),axis=0)
    
#     elif num_of_cameras == 3:
#         P1,P2,P3 = param[:l].reshape((3,4)),param[l:2*l].reshape((3,4)),param[2*l:].reshape((3,4))
#         Proj_points = Proj_points[:,-1:,:,:]
#         Proj_Mat = np.stack((P1,P2,P3),axis=0)
    
#     elif num_of_cameras == 4:
#         P1,P2,P3,P4 = param[:l].reshape((3,4)),param[l:2*l].reshape((3,4)),param[2*l:3*l].reshape((3,4)),param[3*l:].reshape((3,4))
#         Proj_points = Proj_points[:,-1:,:,:]
#         Proj_Mat = np.stack((P1,P2,P3),axis=0)

    
#     coords,VIS_cam_List = triangulateTest(Proj_points,Proj_Mat,base_cam[base_Cam_Index]).solveA()
#     ball_points = coords[:,:,:-1].reshape((-1,1,3))

np.save(baseFilePath+'/Processed/DataPoints3D.npy',coords)
print('save sussesful')


if num_of_cameras == 3:
    Vis(SourceVideoFolder+'/'+Source_video_List[0][0],SourceVideoFolder+'/'+Source_video_List[1][0],SourceVideoFolder+'/'+Source_video_List[2][0],coords).display()

elif num_of_cameras == 2:
    #Vis(SourceVideoFolder+'/'+Source_video_List[1][0],SourceVideoFolder+'/'+Source_video_List[0][0],None,C).display()
    Vis(SourceVideoFolder+'/'+Source_video_List[1][0],SourceVideoFolder+'/'+Source_video_List[0][0],None,coords).display()

elif num_of_cameras == 4:
    Vis(SourceVideoFolder+'/'+Source_video_List[0][0],SourceVideoFolder+'/'+Source_video_List[1][0],SourceVideoFolder+'/'+Source_video_List[2][0],coords).display()
    




