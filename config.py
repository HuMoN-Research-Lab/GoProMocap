
'some system configuration will be defined as below'

__author__ = 'Yifan'

#USER INPUT

#Intials of subject
subject = 'CJC'

# Project Name
project = 'JugglingPractice'

#Enter date in format YYYYMMDD
date = '20200318'

#Enter session number as four digits. Example: for session 1, 0001
session_num = '0001'

#Base folder path where you would like to save the project to
baseProjectPath = 'D:/Juggling'

#Enter the Camera Names
cam_names = ['CamA','CamB','CamD']

#off of base project path
DLCconfigPath = 'DLCNetworks/Juggle-CC-2020-03-24/config.yaml'

#If the checkerboard is NOT in the full video set this as True
checkerboardVid = True

#If the go pro videos get cut into two videos  
num_of_cameras = 3 #Supports up to 4
base_Cam_Index = 'A'    #A/B/C
video_resolution = (1280,960) #specified resized video size # decide from video

include_DLC = True
include_OpenPoseSkeleton = True
include_OpenPoseHands = True
include_OpenPoseFace = True



Len_of_frame = 50 #how many frames you want to reconstruct 3d #whole video option
start_frame = 300






