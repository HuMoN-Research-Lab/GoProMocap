import os
import h5py
import subprocess
import json
import numpy as np
import deeplabcut
from config import DLCconfigPath,  cam_names, useCheckerboardVid, num_of_cameras,baseProjectPath, getCamParams
from create_project import baseFilePath, rawData, checkerVideoFolder, rawVideoFolder
import glob
#from GetCameraParams import getCamParams
#from UndistortVideo import UndistortVideo


def runOPandDLC():
    #Set up camera names 
    cam1 = cam_names[0]
    cam2 = cam_names[1]
    if num_of_cameras >2:
        cam3 = cam_names[2]
    if num_of_cameras >3:
        cam4 = cam_names[3]
    
    #Set directory
    os.chdir("/Windows/system32")
    #Sets the file path to the where the videos are stored
    
    rawfilepath = baseFilePath+ '/Raw/RawGoProData'

    ################Create folders for each step of process###############
    #Create filepath for Intermediate processed 
    if not os.path.exists(baseFilePath + '/Intermediate'):
        os.mkdir(baseFilePath + '/Intermediate')
    interfilepath = baseFilePath + '/Intermediate'

    #Create filepath for Processed Data
    if not os.path.exists(baseFilePath + '/Processed'):
        os.mkdir(baseFilePath + '/Processed')
     
    #Create directory for raw videos
    rawDatadir = [rawVideoFolder]

    #Create a folder for joined videos 
    if not os.path.exists(interfilepath+'/CombinedVideo'):
        os.mkdir(interfilepath+'/CombinedVideo')
    combinedFilepath = interfilepath+'/CombinedVideo'
    combinedDatadir = [combinedFilepath]

    #Create a folder for the undistorted videos
    if not os.path.exists(interfilepath + '/Undistorted'):
        os.mkdir(interfilepath + '/Undistorted')
    undistortedFilepath = interfilepath + '/Undistorted'
    undistortDatadir = [undistortedFilepath]

    #Create a folder for the deeplabcut output
    if not os.path.exists(interfilepath + '/DeepLabCut'):
        os.mkdir(interfilepath + '/DeepLabCut')
    DLCfilepath = interfilepath + '/DeepLabCut'
    DLCDatadir = [DLCfilepath]

    #Create a folder for the openpose output
    if not os.path.exists(interfilepath + '/OpenPoseRaw'):
        os.mkdir(interfilepath + '/OpenPoseRaw')
    openposeRawFilepath = interfilepath + '/OpenPoseRaw'

    #Create a folder for videos
    if not os.path.exists(interfilepath+'/VideoOutput'):
        os.mkdir(interfilepath+'/VideoOutput')
    videoOutputFilepath = interfilepath+'/VideoOutput'
    
    
    ####################### Join video parts together ###################### 
    #create a text file for each camera 
    cam1vids = open(combinedFilepath+'/cam1vids.txt','a')
    cam2vids = open(combinedFilepath+'/cam2vids.txt','a')
    cam3vids = open(combinedFilepath+'/cam3vids.txt','a')
    cam4vids = open(combinedFilepath+'/cam4vids.txt','a')
    for dir in rawDatadir: #for loop parses through the resized video folder 
        for video in os.listdir(dir): 
            #Get length of the name of cameras
            cam1length = len(cam1); cam2length = len(cam2); cam3length = len(cam3); cam4length = len(cam4); 
            if video[:cam1length] == cam1: # if the video is from Cam1
                cam1vids.write('file'+" '" +'\\'+video+"'") #write the file name of the video to the text file
                cam1vids.write('\n')                     
            if video[:cam2length] == cam2: # if the video is from Cam2
                cam2vids.write('file'+" '" +'\\'+video+"'") #write the file name of the video to the text file
                cam2vids.write('\n')                   
            if video[:cam3length] == cam3: # if the video is from Cam3
                cam3vids.write('file'+" '" +'\\'+video+"'") #write the file name of the video to the text file
                cam3vids.write('\n') 
            if video[:cam4length] == cam4: # if the video is from Cam4
                cam4vids.write('file'+" '" +'\\'+video+"'") #write the file name of the video to the text file
                cam4vids.write('\n')                     
    #Close the text files
    cam1vids.close()
    cam2vids.close()
    cam3vids.close()
    cam4vids.close()
    #Use ffmpeg to join all parts of the video together
    subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', combinedFilepath+'/cam1vids.txt', '-c' ,'copy' ,combinedFilepath+'/'+ cam1+'.mp4'])
    subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', combinedFilepath+'/cam2vids.txt', '-c' ,'copy' ,combinedFilepath+'/'+ cam2+'.mp4'])
    subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', combinedFilepath+'/cam3vids.txt', '-c' ,'copy' ,combinedFilepath+'/'+ cam3+'.mp4'])
    subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', combinedFilepath+'/cam4vids.txt', '-c' ,'copy' ,combinedFilepath+'/'+ cam4+'.mp4'])


    #################### Undistortion #########################
    #for dir in combinedDatadir:
    #    for video in os.listdir(dir):
    #        subprocess.call(['ffmpeg', '-i', combinedFilepath+'/'+video, '-vf', "lenscorrection=cx=0.5:cy=0.5:k1=-.115:k2=-0.022", undistortedFilepath+'/'+video])
    
    #if getCamParams:
        #getCamParams()
    
    #UndistortVideo()

    
    #####################Copy Videos to DLC Folder############
    for dir in undistortDatadir:
        for video in os.listdir(dir):
            subprocess.call(['ffmpeg', '-i', undistortedFilepath+'/'+video,  DLCfilepath+'/'+video])


    #################### DeepLabCut ############################
    for dir in DLCDatadir:# Loop through the undistorted folder
        for video in os.listdir(dir):
            #Analyze the videos through deeplabcut
            deeplabcut.analyze_videos(baseProjectPath+'/'+DLCconfigPath, [DLCfilepath +'/'+ video], save_as_csv=True)
            deeplabcut.plot_trajectories(baseProjectPath+'/'+DLCconfigPath,[DLCfilepath +'/'+ video])
    
    for dir in DLCDatadir:
        for video in dir:   
            deeplabcut.create_labeled_video(baseProjectPath+'/'+DLCconfigPath, glob.glob(os.path.join(DLCfilepath ,'*mp4')))
    
    ###################### OpenPose ######################################
    os.chdir("C:/Users/MatthisLab/openpose") # change the directory to openpose
    j = 0
    for dir in undistortDatadir:# loop through undistorted folder
        for video in os.listdir(dir):
            subprocess.call(['bin/OpenPoseDemo.exe', '--video', undistortedFilepath+'/'+video, '--hand','--face','--write_video', videoOutputFilepath+'/OpenPose'+cam_names[j]+'.avi',  '--write_json', openposeRawFilepath+'/'+cam_names[j]])
            j = j +1
    
    
    ###############If you need To use checkerboard videos##################
    
    if useCheckerboardVid:
        checkerDatadir = [checkerVideoFolder]   

    #Create a folder for the undistorted videos
        if not os.path.exists(interfilepath + '/CheckerboardUndistorted'):
            os.mkdir(interfilepath + '/CheckerboardUndistorted')
        checkerUndistortFilepath = interfilepath + '/CheckerboardUndistorted'
        
        for dir in checkerDatadir:
            for video in os.listdir(dir):
                subprocess.call(['ffmpeg', '-i', checkerVideoFolder+'/'+video, '-vf', "lenscorrection=cx=0.5:cy=0.5:k1=-.115:k2=-0.022", checkerUndistortFilepath+'/'+video])
    
    ########## Put Openpose Data into h5   ######################
    if not os.path.exists(interfilepath + '/OpenPoseOutput'):
        os.mkdir(interfilepath + '/OpenPoseOutput')
    openposeOutputFilepath = interfilepath + '/OpenPoseOutput'
    
    j = 0 #Counter variable
    with  h5py.File(openposeOutputFilepath + '/OpenPoseh5Output.hdf5', 'w') as f:
        cams = f.create_group('Cameras')
        for cam in os.listdir(openposeRawFilepath):# Loops through each camera
            k =0
            cameraGroup = cams.create_group(cam_names[j])
            for files in os.listdir(openposeRawFilepath+'/'+cam): #loops through each json file   
                fileGroup = cameraGroup.create_group('Frame'+str(k))
                inputFile = open(openposeRawFilepath+'/'+cam+'/'+files) #open json file
                data = json.load(inputFile) #load json content
                inputFile.close() #close the input file
                ii = 0 
                for people in data['people']:
                    skeleton = np.array(people['pose_keypoints_2d']).reshape((-1,3))
                    hand_left = np.array(people['hand_left_keypoints_2d']).reshape((-1,3))
                    hand_right = np.array(people['hand_right_keypoints_2d']).reshape((-1,3))
                    face = np.array(people['face_keypoints_2d']).reshape((-1,3))  #Get skeleton points

                    persongroup = fileGroup.create_group('Person'+str(ii))
                    skeletondata = persongroup.create_dataset('Skeleton', data =skeleton)
                    rightHanddata = persongroup.create_dataset('RightHand', data =hand_right) 
                    leftHanddata = persongroup.create_dataset('LeftHand', data =hand_left)
                    facedata = persongroup.create_dataset('Face', data =face)                                       
                    ii = ii +1 
                k= k +1
            j = j + 1
