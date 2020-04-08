import os

####### To reconstruct the videos. Edit and run this script with your specific details, 
####### then from the baseProject path you provided go to the Raw folder which will be in 
####### the path of BaseProjectPath/subject/sessionID. In the Raw folder place your full videos 
###### in the RawGoProData folder and the checkerboard only videos in the checkerboard folder.\
#  
####### IMPORTANT: name each video CamA, CamB, CamC, CamD. If there is multiple parts name them CamA_part1,
####### CamA_part2 etc. After the videos are named correctly and placed in the proper folder, run the script
####### main.py to see your videos reconstructed in 3D!

'some system configuration will be defined as below'

__author__ = 'Yifan'

#USER INPUT

#Intials of subject
subject = 'CJC'

# Project Name
project = 'JugglingPractice'

#Enter date in format YYYYMMDD
date = '20200407'

#Enter session number as four digits. Example: for session 1, 0001
session_num = '0007'

#Base folder path where you would like to save the project to
baseProjectPath = 'C:/Users/chris/JugglingProject'

#off of base project path
DLCconfigPath = 'DLCNetworks/Juggle-CC-2020-03-24/config.yaml'

#If the checkerboard is NOT in the full video set this as True
checkerboardVid = True

#If the go pro videos get cut into two videos 
num_of_Video_parts = 2 #Could probably take this out 

num_of_cameras = 4  #Supports up to 4
base_Cam_Index = 'A'    #A/B/C
video_resolution = (1280,960) #specified resized video size # decide from video
include_ball = True
points_inFrame = 25


Len_of_frame = 500 #how many frames you want to reconstruct 3d #whole video option
start_frame = 20000


#END OF USER INPUT
##########################################################################

# Create folder name
sessionID =  project+session_num+'_'+date

#Create Folders for Project
if not os.path.exists(baseProjectPath+'/'+subject):
    os.mkdir(baseProjectPath+'/'+subject)

if not os.path.exists(baseProjectPath+'/'+subject+'/'+sessionID):
    os.mkdir(baseProjectPath+'/'+subject+'/'+sessionID)
baseFilePath = baseProjectPath+'/'+subject+'/'+sessionID

if not os.path.exists(baseFilePath+'/Raw'):
    os.mkdir(baseFilePath+'/Raw')
rawData = baseFilePath+'/Raw'

if not os.path.exists(rawData+'/RawGoProData'):
    os.mkdir(rawData+'/RawGoProData')
rawVideoFolder = rawData+'/RawGoProData'

if not os.path.exists(rawData+'/Checkerboard'):
    os.mkdir(rawData+'/Checkerboard')
checkerVideoFolder = rawData+'Checkerboard'


