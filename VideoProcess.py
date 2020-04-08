import os
import subprocess
import deeplabcut
from config import subject, sessionID, DLCconfigPath, rawVideoFolder, baseProjectPath, num_of_Video_parts,baseFilePath


def runOPandDLC():
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
    datadir1 = [rawfilepath+'/'+rawVideoFolder]

    #Create a folder for the resized videos
    if not os.path.exists(interfilepath + '/Resized'):
        os.mkdir(interfilepath + '/Resized')
    filepath1 = interfilepath + '/Resized'
    datadir2 = [filepath1]

    #Create a folder for the undistorted videos
    if not os.path.exists(interfilepath + '/Undistorted'):
        os.mkdir(interfilepath + '/Undistorted')
    filepath2 = interfilepath + '/Undistorted'
    datadir3 = [filepath2]

    #Create a folder for the deeplabcut output
    if not os.path.exists(interfilepath + '/DeepLabCut'):
        os.mkdir(interfilepath + '/DeepLabCut')
    filepath3 = interfilepath + '/DeepLabCut'
    datadir4 = [filepath3]

    #Create a folder for the openpose output
    if not os.path.exists(interfilepath + '/OpenPose'):
        os.mkdir(interfilepath + '/OpenPose')
    filepath4 = interfilepath + '/OpenPose'

    #Create a folder for videos
    if not os.path.exists(interfilepath+'/VideoOutput'):
        os.mkdir(interfilepath+'/VideoOutput')
    filepath5 = interfilepath+'/VideoOutput'

    ###################### Resize Videos ##################################
    for dir in datadir1: # for loop parses through the raw video folder
        for video in os.listdir(dir):
            #
            subprocess.call(['ffmpeg', '-i', rawfilepath+'/'+rawVideoFolder+'/'+video, '-vf', 'scale=1280:960', filepath1+'/'+video])


    ####################### Join video parts together ###################### 
    if num_of_Video_parts >1 :
        #create a text file for each camera 
        camAvids = open(filepath1+'/camAvids.txt','a')
        camBvids = open(filepath1+'/camBvids.txt','a')
        camCvids = open(filepath1+'/camCvids.txt','a')
        camDvids = open(filepath1+'/camDvids.txt','a')
        for dir in datadir2: #for loop parses through the resized video folder 
            for video in os.listdir(dir): 
                if video[:4] == 'CamA': # if the video is from CamA
                    camAvids.write('file'+" '" +'\\'+video+"'") #write the file name of the video to the text file
                    camAvids.write('\n')                     
                if video[:4] == 'CamB': # if the video is from CamB
                    camBvids.write('file'+" '" +'\\'+video+"'") #write the file name of the video to the text file
                    camBvids.write('\n')                   
                if video[:4] == 'CamC': # if the video is from CamC
                    camCvids.write('file'+" '" +'\\'+video+"'") #write the file name of the video to the text file
                    camCvids.write('\n') 
                if video[:4] == 'CamD': # if the video is from CamD
                    camDvids.write('file'+" '" +'\\'+video+"'") #write the file name of the video to the text file
                    camDvids.write('\n')                     
        #Close the text files
        camAvids.close()
        camBvids.close()
        camCvids.close()
        camDvids.close()
        #Use ffmpeg to join all parts of the video together
        subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', filepath1+'/camAvids.txt', '-c' ,'copy' ,filepath1+'/CamA.mp4'])
        subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', filepath1+'/camBvids.txt', '-c' ,'copy' ,filepath1+'/CamB.mp4'])
        subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', filepath1+'/camCvids.txt', '-c' ,'copy' ,filepath1+'/CamC.mp4'])
        subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', filepath1+'/camDvids.txt', '-c' ,'copy' ,filepath1+'/CamD.mp4'])


    #################### Undistortion #########################
    if num_of_Video_parts ==1: #If there is just one part to video, the entire resized folder is undistorted
        for dir in datadir2:
            for video in os.listdir(dir):
                subprocess.call(['ffmpeg', '-i', filepath1+'/'+video, '-vf', "lenscorrection=cx=0.5:cy=0.5:k1=-.115:k2=-0.022", filepath2+'/'+video])
    if num_of_Video_parts >1:# If there is multiple parts originally, only the videos in the folder that are joined together are undistorted
        for dir in datadir2:
            for video in os.listdir(dir):
                if len(video) == 8:
                    subprocess.call(['ffmpeg', '-i', filepath1+'/'+video, '-vf', "lenscorrection=cx=0.5:cy=0.5:k1=-.115:k2=-0.022", filepath2+'/'+video])

         
    #################### DeepLabCut ############################
    for dir in datadir3:# Loop through the undistorted folder
        for video in os.listdir(dir):
            #Analyze the videos through deeplabcut
            deeplabcut.analyze_videos(baseProjectPath+'/'+DLCconfigPath, [filepath2 +'/'+ video],videotype='mp4', destfolder = filepath3, save_as_csv = True)
            deeplabcut.plot_trajectories(baseProjectPath+'/'+DLCconfigPath,[filepath2 +'/'+ video],videotype= 'mp4', destfolder = filepath3)
          #  deeplabcut.create_labeled_video(DLCconfigPath,[filepath2 +'/'+ video],videotype = 'mp4', destfolder = filepath5)

    
    ###################### OpenPose ##########################   
    os.chdir("/Users/MatthisLab/openpose") # change the directory to openpose
    for dir in datadir3:# loop through undistorted folder
        for video in os.listdir(dir):
            videoName = video[:4] 
            subprocess.call(['bin/OpenPoseDemo.exe', '--video', filepath2+'/'+video, '--hand','--face','--write_video', filepath5+'/OpenPose'+videoName+'.avi',  '--write_json', filepath4+'/'+videoName])
runOPandDLC()