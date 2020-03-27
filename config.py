


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'some system configuration will be defined as below'

__author__ = 'Yifan'

num_of_cameras = 2   #only support 2 and 3 rn
base_Cam_Index = 'A'    #A/B/C
video_resolution = (1280,960) #specified resized video size # decide from video
include_ball = True
points_inFrame = 25
SAVE_FOLDER = 'output/'

Len_of_frame = 500 #how many frames you want to reconstruct 3d #whole video option
start_frame = 0

#save video option



#=============================================USER INPUT DATA================================

#SourceVideoFolder = 'CalibrationData/SourceVideos' #can be cleaner, ALL video that will be used for calibrate should be in this folder. These videos should be openpose/deeplabcut resized videos
SourceVideoFolder = 'videos214' 



#video names with camera names
#Source_video_List = [['CamAresized.avi','CamA'],['CamCresized.avi','CamC'],['CheckerBoardCalibrateCamB.MP4','CamB']] #name of the video files and camera names, ONLY NAME THEM: CamA/CamB/CamC

Source_video_List =  [['CamA_labeled.avi','CamA'],['CamB_labeled.avi','CamB']]



#==========================================pixel data
# Pixel_coord_FIlE_List = [['PixelCoordData/OP_CamA.npy','CamA'],    #pixel data file path and camera name ,ONLY NAME THEM: CamA/CamB/CamC
#                          ['PixelCoordData/OP_CamB.npy','CamB'],
#                          ['PixelCoordData/OP_CamC.npy','CamC']]


Pixel_coord_FIlE_List = [['214OP_data/OP_CamA.npy','CamA'],
                         ['214OP_data/OP_CamB.npy','CamB']]

Pixel_coord_FIlE_List_include_ball = [['214OP_data/OP_CamA.npy','214OP_data/dlc_CamA.npy','CamA'],
                                        ['214OP_data/OP_CamB.npy','214OP_data/dlc_CamB.npy','CamB']]

                        