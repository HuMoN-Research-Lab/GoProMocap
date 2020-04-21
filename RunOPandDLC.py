import os
import subprocess
import deeplabcut
from config import DLCconfigPath,  cam_names, useCheckerboardVid, num_of_cameras,baseProjectPath
from create_project import baseFilePath, rawData, checkerVideoFolder, rawVideoFolder
import glob


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
    if not os.path.exists(interfilepath + '/OpenPose'):
        os.mkdir(interfilepath + '/OpenPose')
    openposeFilepath = interfilepath + '/OpenPose'

    #Create a folder for videos
    if not os.path.exists(interfilepath+'/VideoOutput'):
        os.mkdir(interfilepath+'/VideoOutput')
    videoOutputFilepath = interfilepath+'/VideoOutput'
    
    ####################### Join video parts together ###################### 
    #create a text file for each camera 
    cam1vids = open(rawfilepath+'/cam1vids.txt','a')
    cam2vids = open(rawfilepath+'/cam2vids.txt','a')
    cam3vids = open(rawfilepath+'/cam3vids.txt','a')
    cam4vids = open(rawfilepath+'/cam4vids.txt','a')
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
    subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', rawfilepath+'/cam1vids.txt', '-c' ,'copy' ,combinedFilepath+'/'+ cam1+'.mp4'])
    subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', rawfilepath+'/cam2vids.txt', '-c' ,'copy' ,combinedFilepath+'/'+ cam2+'.mp4'])
    subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', rawfilepath+'/cam3vids.txt', '-c' ,'copy' ,combinedFilepath+'/'+ cam3+'.mp4'])
    subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', rawfilepath+'/cam4vids.txt', '-c' ,'copy' ,combinedFilepath+'/'+ cam4+'.mp4'])


    #################### Undistortion #########################
    for dir in combinedDatadir:
        for video in os.listdir(dir):
            subprocess.call(['ffmpeg', '-i', combinedFilepath+'/'+video, '-vf', "lenscorrection=cx=0.5:cy=0.5:k1=-.115:k2=-0.022", undistortedFilepath+'/'+video])

    #####################Copy Videos to DLC Folder############
    for dir in undistortDatadir:
        for video in os.listdir(dir):
            subprocess.call(['ffmpeg', '-i', undistortDatadir+'/'+video,  DLCfilepath+'/'+video])


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
    for dir in datadir3:# loop through undistorted folder
        for video in os.listdir(dir):
            subprocess.call(['bin/OpenPoseDemo.exe', '--video', undistortedFilepath+'/'+video, '--hand','--face','--write_video', videoOutputFilepath+'/OpenPose'+cam_names[j]+'.avi',  '--write_json', openposeFilepath+'/'+cam_names[j]])
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

