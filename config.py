
'some system configuration will be defined as below'

__author__ = 'Yifan'

#USER INPUT

#Intials of subject
subject = 'CJC'

# Project Name
project = 'JugglingPractice'

#Enter date in format YYYYMMDD
date = '20200319'

#Enter session number as four digits. Example: for session 1, 0001
session_num = '0002'

#Base folder path where you would like to save the project to
baseProjectPath = 'D:/Juggling'

#Enter the Camera Names
cam_names = ['CamA','CamB','CamC','CamD']

#off of base project path
DLCconfigPath = 'DLCNetworks/Juggle-CC-2020-03-24/config.yaml'

#If you need to use a short clip of checkerboard set as true
useCheckerboardVid = True

#If the go pro videos get cut into two videos  
num_of_cameras = 4 #Supports up to 4
base_Cam_Index = 'A'    #A/B/C

#What features from video you are tracking
include_DLC = True
include_OpenPoseSkeleton = True
include_OpenPoseHands = True
include_OpenPoseFace = False

#What frame of video you want to start reconstruction
start_frame = 1000
#How many frames you want to reconstruct, for full video input -1
Len_of_frame = 10





