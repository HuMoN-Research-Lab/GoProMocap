import os
import subprocess
import deeplabcut
from config import subject, sessionID, DLCconfigPath, rawVideoFolder, baseProjectPath 

#Set file path details
#subject = 'CJC'
#sessionID = 'JugglingPractice0001_20200318'
#configPath = 'D:Juggling/DLCNetworks/Juggle-CC-2020-03-24/config.yaml'

def runOPandDLC():
    #Set directory, and original filepath 
    os.chdir("/Windows/system32")
    origfilepath = baseProjectPath+'/' +subject + '/' + sessionID
    datadir1 = [origfilepath+'/'+rawVideoFolder]

    #Create a folder for the resized videos
    if not os.path.exists(origfilepath + '/Resized'):
        os.mkdir(origfilepath + '/Resized')
    filepath1 = origfilepath + '/Resized'
    datadir2 = [filepath1]

    #Create a folder for the resized videos
    if not os.path.exists(origfilepath + '/Undistorted'):
        os.mkdir(origfilepath + '/Undistorted')
    filepath2 = origfilepath + '/Undistorted'
    datadir3 = [filepath2]

    #Create a folder for the deeplabcut output
    if not os.path.exists(origfilepath + '/DeepLabCut'):
        os.mkdir(origfilepath + '/DeepLabCut')
    filepath3 = origfilepath + '/DeepLabCut'
    datadir4 = [filepath3]

    #Create a folder for the openpose output
    if not os.path.exists(origfilepath + '/OpenPose'):
        os.mkdir(origfilepath + '/OpenPose')
    filepath4 = origfilepath + '/OpenPose'

    #Create a folder for videos
    if not os.path.exists(origfilepath+'/VideoOutput'):
        os.mkdir(origfilepath+'/VideoOutput')
    filepath5 = origfilepath+'/VideoOutput'

    #Use ffmpeg to resize videos and save them in the just created resized video folder
    for dir in datadir1:
        for video in os.listdir(dir):
            #subprocess.call(['ffmpeg', '-i', origfilepath+'/CheckerBoard/'+video1, '-vf', 'scale=1280:960',"lenscorrection=cx=0.5:cy=0.5:k1=-.115:k2=-0.022", filepath1+'/'+video1])
            subprocess.call(['ffmpeg', '-i', origfilepath+'/'+rawVideoFolder+'/'+video, '-vf', 'scale=1280:960', filepath1+'/'+video])

    #Use ffmpeg to Undistort videos
    for dir in datadir2:
        for video in os.listdir(dir):
            subprocess.call(['ffmpeg', '-i', filepath1+'/'+video, '-vf', "lenscorrection=cx=0.5:cy=0.5:k1=-.115:k2=-0.022", filepath2+'/'+video])

    #Use deeplabcut to analyze videos and save the results to the folder for processed videos     

    for dir in datadir3:
        for video in os.listdir(dir):
           
            deeplabcut.analyze_videos(DLCconfigPath, [filepath2 +'/'+ video],videotype='mp4', destfolder = filepath3, save_as_csv = True)
            deeplabcut.plot_trajectories(DLCconfigPath,[filepath2 +'/'+ video],videotype= 'mp4', destfolder = filepath3)
           # deeplabcut.create_labeled_video(DLCconfigPath,[filepath2 +'/'+ video],videotype = 'mp4', destfolder = filepath5)

    #Change directory to openpose and run openpose on the videos then save the results to the processed video folder

    os.chdir("/Users/MatthisLab/openpose")
    for dir in datadir3:
        for video in os.listdir(dir):
            videoName = video[:4]
            subprocess.call(['bin/OpenPoseDemo.exe', '--video', filepath2+'/'+video, '--hand','--face', '--write_video', filepath5+'/OpenPose'+videoName+'avi', '--write_json', filepath4+'/'+videoName])
runOPandDLC()