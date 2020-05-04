
'some system configuration will be defined as below'

__author__ = 'Yifan'

#USER INPUT

#Intials of subject
subject = 'CJC'

# Project Name
project = 'Chessboard'

#Enter date in format YYYYMMDD
date = '20200501'

#Enter session number as four digits. Example: for session 1, 0001
session_num = '0001'

#Base folder path where you would like to save the project to
baseProjectPath = 'D:/Calibration'

#Enter the Camera Names
cam_names = ['CamA','CamB','CamC','CamD']

#off of base project path
DLCconfigPath = 'DLCNetworks/Juggling-CC-2020-04-21/config.yaml'

#If you need to use a short clip of checkerboard set as true
useCheckerboardVid = False

#If you need to get camera parameters set to true( If first time processing set to true)
getCamParams = True

#If the go pro videos get cut into two videos  
num_of_cameras = 4 #Supports up to 4, if you use 4 cameras you must have 4 cameras in cam_names
base_Cam_Index = 'CamA'  #Put same name as you did in cam_names

#What features from video you are tracking
include_DLC = False
include_OpenPoseSkeleton = True
include_OpenPoseHands = False
include_OpenPoseFace = False

#What frame of video you want to start reconstruction
start_frame = 0
#How many frames you want to reconstruct, for full video input -1
Len_of_frame = -1





