import os
import subprocess
#import deeplabcut
from config import DLCconfigPath,  cam_names, useCheckerboardVid
from create_project import baseFilePath, rawData, checkerVideoFolder, rawVideoFolder



def runOPandDLC():
    #Set up camera names 
    cam1 = cam_names[0]
    cam2 = cam_names[1]
    cam3 = cam_names[2]
    cam4 = cam_names[3]

    #Set directory
    os.chdir("/Windows/system32")
    #Sets the file path to the where the videos are stored
    
    rawfilepath = baseFilePath+ '/Raw'

    ################Create folders for each step of process###############
    #Create filepath for Intermediate processed 
    if not os.path.exists(baseFilePath + '/Intermediate'):
        os.mkdir(baseFilePath + '/Intermediate')
    interfilepath = baseFilePath + '/Intermediate'

    #Create filepath for Processed Data
    if not os.path.exists(baseFilePath + '/Processed'):
        os.mkdir(baseFilePath + '/Processed')
     
    #Create directory for raw videos
    datadir1 = [rawVideoFolder]

    #Create a folder for the resized videos
    if not os.path.exists(interfilepath + '/Resized'):
        os.mkdir(interfilepath + '/Resized')
    filepath1 = interfilepath + '/Resized'
    datadir2 = [filepath1]

    #Create a folder for joined videos 
    if not os.path.exists(interfilepath+'/CombinedVideo'):
        os.mkdir(interfilepath+'/CombinedVideo')
    filepath2 = interfilepath+'/CombinedVideo'

    #Create a folder for the undistorted videos
    if not os.path.exists(interfilepath + '/Undistorted'):
        os.mkdir(interfilepath + '/Undistorted')
    filepath3 = interfilepath + '/Undistorted'
    datadir3 = [filepath3]

    #Create a folder for the deeplabcut output
    if not os.path.exists(interfilepath + '/DeepLabCut'):
        os.mkdir(interfilepath + '/DeepLabCut')
    filepath4 = interfilepath + '/DeepLabCut'
    datadir4 = [filepath4]

    #Create a folder for the openpose output
    if not os.path.exists(interfilepath + '/OpenPose'):
        os.mkdir(interfilepath + '/OpenPose')
    filepath5 = interfilepath + '/OpenPose'

    #Create a folder for videos
    if not os.path.exists(interfilepath+'/VideoOutput'):
        os.mkdir(interfilepath+'/VideoOutput')
    filepath6 = interfilepath+'/VideoOutput'

    
    '''
    
    ###################### Resize Videos ##################################
    for dir in datadir1: # for loop parses through the raw video folder
        for video in os.listdir(dir):
            #
            subprocess.call(['ffmpeg', '-i', rawVideoFolder+'/'+video, '-vf', 'scale=1280:960', filepath1+'/'+video])


    ####################### Join video parts together ###################### 
    #create a text file for each camera 
    cam1vids = open(filepath1+'/cam1vids.txt','a')
    cam2vids = open(filepath1+'/cam2vids.txt','a')
    cam3vids = open(filepath1+'/cam3vids.txt','a')
    cam4vids = open(filepath1+'/cam4vids.txt','a')
    for dir in datadir2: #for loop parses through the resized video folder 
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
    subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', filepath1+'/cam1vids.txt', '-c' ,'copy' ,filepath2+'/'+ cam1+'.mp4'])
    subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', filepath1+'/cam2vids.txt', '-c' ,'copy' ,filepath2+'/'+ cam2+'.mp4'])
    subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', filepath1+'/cam3vids.txt', '-c' ,'copy' ,filepath2+'/'+ cam3+'.mp4'])
    subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', filepath1+'/cam4vids.txt', '-c' ,'copy' ,filepath2+'/'+ cam4+'.mp4'])


    #################### Undistortion #########################
    for dir in datadir2:
        for video in os.listdir(dir):
            subprocess.call(['ffmpeg', '-i', filepath2+'/'+video, '-vf', "lenscorrection=cx=0.5:cy=0.5:k1=-.115:k2=-0.022", filepath3+'/'+video])

        
    #################### DeepLabCut ############################
    for dir in datadir3:# Loop through the undistorted folder
        for video in os.listdir(dir):
            #Analyze the videos through deeplabcut
            deeplabcut.analyze_videos(baseProjectPath+'/'+DLCconfigPath, [filepath3 +'/'+ video],videotype='mp4', destfolder = filepath4, save_as_csv = True)
            deeplabcut.plot_trajectories(baseProjectPath+'/'+DLCconfigPath,[filepath3 +'/'+ video],videotype= 'mp4', destfolder = filepath4)
          #  deeplabcut.create_labeled_video(DLCconfigPath,[filepath3 +'/'+ video],videotype = 'mp4', destfolder = filepath6)

    
    ###################### OpenPose ######################################
    os.chdir("/Users/MatthisLab/openpose") # change the directory to openpose
    j = 0
    for dir in datadir3:# loop through undistorted folder
        for video in os.listdir(dir):
            subprocess.call(['bin/OpenPoseDemo.exe', '--video', filepath3+'/'+video, '--hand','--face','--write_video', filepath6+'/OpenPose'+cam_names[j]+'.avi',  '--write_json', filepath5+'/'+cam_names[j]])
            j = j +1
    '''
    
    ###############If you need To use checkerboard videos##################
    
    if useCheckerboardVid:
        datadir5 = [checkerVideoFolder]
        
        if not os.path.exists(interfilepath + '/CheckerboardResized'):
            os.mkdir(interfilepath + '/CheckerboardResized')
        filepath7 = interfilepath + '/CheckerboardResized'
        datadir6 = [filepath7]

    #Create a folder for the undistorted videos
        if not os.path.exists(interfilepath + '/CheckerboardUndistorted'):
            os.mkdir(interfilepath + '/CheckerboardUndistorted')
        filepath8 = interfilepath + '/CheckerboardUndistorted'

        for dir in datadir5: # for loop parses through the raw video folder
            for video in os.listdir(dir):
                subprocess.call(['ffmpeg', '-i', checkerVideoFolder+'/'+video, '-vf', 'scale=1280:960', filepath7+'/'+video])

        for dir in datadir6:
            for video in os.listdir(dir):
                subprocess.call(['ffmpeg', '-i', filepath7+'/'+video, '-vf', "lenscorrection=cx=0.5:cy=0.5:k1=-.115:k2=-0.022", filepath8+'/'+video])
        
        

