import os
import subprocess
import deeplabcut
from config import subject, sessionID, DLCconfigPath, rawVideoFolder, baseProjectPath, num_of_Video_parts

#Set file path details
#subject = 'CJC'
#sessionID = 'JugglingPractice0001_20200318'
#configPath = 'D:Juggling/DLCNetworks/Juggle-CC-2020-03-24/config.yaml'

def runOPandDLC():
    #Set directory, and original filepath 
    os.chdir("/Windows/system32")

    origfilepath = baseProjectPath+'/' +subject + '/' + sessionID 
    rawfilepath = baseProjectPath+'/' +subject + '/' + sessionID + '/Raw'

    #Create main filepath for Intermediate processed 
    if not os.path.exists(origfilepath + '/Intermediate'):
        os.mkdir(origfilepath + '/Intermediate')
    interfilepath = origfilepath + '/Intermediate'

    if not os.path.exists(origfilepath + '/Processed'):
        os.mkdir(origfilepath + '/Processed')
     
   
    datadir1 = [rawfilepath+'/'+rawVideoFolder]

    #Create a folder for the resized videos
    if not os.path.exists(interfilepath + '/Resized'):
        os.mkdir(interfilepath + '/Resized')
    filepath1 = interfilepath + '/Resized'
    datadir2 = [filepath1]

    #Create a folder for the resized videos
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


    for dir in datadir1:
        for video in os.listdir(dir):
            subprocess.call(['ffmpeg', '-i', rawfilepath+'/'+rawVideoFolder+'/'+video, '-vf', 'scale=1280:960', filepath1+'/'+video])

    #Use ffmpeg to resize videos and save them in the just created resized video folder
    
    #camBvids = open(filepath1+'/camBvids.txt','a')
    #camCvids = open(filepath1+'/camCvids.txt','a')
    #camDvids = open(filepath1+'/camDvids.txt','a')
    #if num_of_Video_parts >1 :
     #   for dir in datadir1:
     #       for video in os.listdir(dir):
      #          if video[:4] == 'CamA':
      #              camAvids = open(filepath1+'/camAvids.txt','a')
     #               camAvids.write('CamA')                   
      #          if video[:4] == 'CamB':
       #             camBvids.write(filepath1+'/'+video)                    
       #         if video[:4] == 'CamC':
       #             camCvids.write(filepath1+'/'+video)
        #        if video[:4] == 'CamD':
         #           camDvids.write(filepath1+'/'+video)
                    

       # subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', filepath1+'/camAvids.txt', '-c' ,'copy' ,filepath1+'/CamA.mp4'])
        #subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', filepath1+'/camBvids.txt', '-c' ,'copy' ,filepath1+'/CamB.mp4'])
        #subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', filepath1+'/camCvids.txt', '-c' ,'copy' ,filepath1+'/CamC.mp4'])
        #subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', filepath1+'/camDvids.txt', '-c' ,'copy' ,filepath1+'/CamD.mp4'])


    #Use ffmpeg to Undistort videos
    for dir in datadir2:
        for video in os.listdir(dir):
            subprocess.call(['ffmpeg', '-i', filepath1+'/'+video, '-vf', "lenscorrection=cx=0.5:cy=0.5:k1=-.115:k2=-0.022", filepath2+'/'+video])

    #Use deeplabcut to analyze videos and save the results to the folder for processed videos     

    for dir in datadir3:
        for video in os.listdir(dir):
           
            deeplabcut.analyze_videos(baseProjectPath+'/'+DLCconfigPath, [filepath2 +'/'+ video],videotype='mp4', destfolder = filepath3, save_as_csv = True)
            deeplabcut.plot_trajectories(baseProjectPath+'/'+DLCconfigPath,[filepath2 +'/'+ video],videotype= 'mp4', destfolder = filepath3)
          #  deeplabcut.create_labeled_video(DLCconfigPath,[filepath2 +'/'+ video],videotype = 'mp4', destfolder = filepath5)

    #Change directory to openpose and run openpose on the videos then save the results to the processed video folder

    os.chdir("/Users/MatthisLab/openpose")
    for dir in datadir3:
        for video in os.listdir(dir):
            videoName = video[:4]
            subprocess.call(['bin/OpenPoseDemo.exe', '--video', filepath2+'/'+video, '--hand','--face','--write_video', filepath5+'/OpenPose'+videoName+'.avi',  '--write_json', filepath4+'/'+videoName])
