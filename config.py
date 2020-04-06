


#!/usr/bin/env python3e
# -*- coding: utf-8 -*-

'some system configuration will be defined as below'

__author__ = 'Yifan'


##############################################################################
# The Videos you are processing should be saved in the following way:
# BaseProjectPath/Subject/sessionID/Raw and in the Raw folder there should be a folder for RawVideos and raw checkerboard 
#Save Videos as CamA, CamB, CamC, CamD.  If more than one part save as CamA_part1, CamA_part2...

baseProjectPath = 'C:Users/chris/JugglingProject' 
subject = 'CJC'
sessionID = 'JugglingPractice0005_20200331'
rawVideoFolder = 'Checkerboard'

#off of base project path
DLCconfigPath = 'DLCNetworks/Juggle-CC-2020-03-24/config.yaml'

#If the go pro videos get cut into two videos 
num_of_Video_parts = 2

num_of_cameras = 4  #only support 2 and 3 rn
base_Cam_Index = 'A'    #A/B/C
video_resolution = (1280,960) #specified resized video size # decide from video
include_ball = True
points_inFrame = 25


Len_of_frame = 500 #how many frames you want to reconstruct 3d #whole video option
start_frame = 20000







                        