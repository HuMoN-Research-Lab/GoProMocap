import os
import tkinter
from GUI import firstGUI, secondGUI
import csv
#from config import session_num, project, date, subject, baseProjectPath, calibrateCameras



def create_project():
    '''Function that opens the GUIs and creates folders'''
    root=tkinter.Tk()#Open tkinter
    firstGUI(root).createProject()#RUn the first GUI   
    root.mainloop()#Run GUI
    
    with open('ProjectFoldersConfig.csv','r') as f:#Open the csv that stores the variables
        reader = csv.reader(f)#Read the variables
        config = list(reader)#Place them into list
    
    #Put them into variable names
    config = config[0]
    
    subject = config[0]
    date = config[1]
    project = config[2]
    session_num = config[3]
    baseProjectPath = config[4]
    DLCfilepath = config[5]
    #Bool values are assigned T/F through if statements since they are saved as 0 or 1
    if config[6] == '0':
        useCheckerBoardVid = False
    else:
        useCheckerBoardVid = True
    
    if config[7] == '0':
        calibrateCameras = False
    else:
        calibrateCameras = True
    if config[8] == '0':
        portraitMode = False
    else:
        portraitMode = True
    numCamera = config[9]

    root2 = tkinter.Tk()#Open tkinter
    secondGUI(root2).runReconstruction()#Run the second GUI
    root2.mainloop()#Run GUI
    with open('runProject.csv', 'r') as f:#Open csv that stores variables
        reader = csv.reader(f)#Read the variables
        config2 = list(reader)#Place variables into a list
    #Name the variables
    config2 = config2[0]
    if config2[2] == '0' and config2[3] =='0':
        cam_names = [config2[0],config2[1]]
    elif config2[2] != '0' and config2[3] == '0':
        cam_names = [config2[0],config2[1],config2[2]]
    else:
        cam_names = [config2[0], config2[1], config2[2], config2[3]]
    base_cam_index = config2[4]
    startframe  = config2[5]
    lenFrame= config2[6]
    #Bool values are assigned T/F through if statements since they are saved as 0 or 1    
    if config2[7] =='0':
        include_DLC = False 
    else:
        include_DLC = True
    if config2[8] =='0':
        include_OpenPoseFace = False 
    else:
        include_OpenPoseFace = True
    if config2[9] =='0':
        include_OpenPoseHands = False 
    else:
        include_OpenPoseHands = True
    if config2[10] =='0':
        include_OpenPoseSkeleton = False 
    else:
        include_OpenPoseSkeleton = True
    #Create all Folder Names
    sessionID =  project+session_num+'_'+date
    baseFilePath = baseProjectPath+'/'+subject+'/'+sessionID    
    
    rawData = baseFilePath+'/Raw'
    checkerVideoFolder = rawData+'/Checkerboard'
    rawVideoFolder = rawData+'/RawGoProData'
    calibrationFilePath = baseProjectPath +'/'+subject+'/Calibration'
    cameraParamsFilePath = calibrationFilePath +'/CameraParams'
    calibVideoFilepath = calibrationFilePath +'/CalibrationVideos'
    interfilepath = baseFilePath + '/Intermediate'
    videoOutputFilepath = interfilepath+'/VideoOutput'
    openposeRawFilepath = interfilepath + '/OpenPoseRaw'
    DLCfilepath = interfilepath + '/DeepLabCut'
    undistortedFilepath = interfilepath + '/Undistorted'
    combinedFilepath = interfilepath+'/CombinedVideo'
    processedFilePath = baseFilePath +'/Processed'
    trimFilepath = interfilepath +'/Trimmed'
    openposeOutputFilepath = interfilepath + '/OpenPoseOutput'

    #Create Folders for Project
    if not os.path.exists(baseProjectPath+'/'+subject):
        os.mkdir(baseProjectPath+'/'+subject)

    if not os.path.exists(baseProjectPath+'/'+subject+'/'+sessionID):
        os.mkdir(baseProjectPath+'/'+subject+'/'+sessionID)
    
    if not os.path.exists(rawData):
        os.mkdir(rawData)
    
    if not os.path.exists(rawVideoFolder):
        os.mkdir(rawVideoFolder)
    
    if not os.path.exists(checkerVideoFolder):
        os.mkdir(checkerVideoFolder)

    if not os.path.exists(calibrationFilePath):
        os.mkdir(calibrationFilePath)
    
    if not os.path.exists(cameraParamsFilePath):
        os.mkdir(cameraParamsFilePath)

    if not os.path.exists(calibVideoFilepath):
        os.mkdir(calibVideoFilepath)

    if not os.path.exists(interfilepath):
        os.mkdir(interfilepath)
    
    if not os.path.exists(processedFilePath):
        os.mkdir(processedFilePath)
        
    if not os.path.exists(combinedFilepath):
        os.mkdir(combinedFilepath)
    
    if not os.path.exists(undistortedFilepath):
        os.mkdir(undistortedFilepath)
    
    if not os.path.exists(DLCfilepath):
        os.mkdir(DLCfilepath)
    
    if not os.path.exists(openposeRawFilepath):
        os.mkdir(openposeRawFilepath)
    
    if not os.path.exists(videoOutputFilepath):
        os.mkdir(videoOutputFilepath)
    if not os.path.exists(trimFilepath):
        os.mkdir(trimFilepath)
    if not os.path.exists(openposeOutputFilepath):
        os.mkdir(openposeOutputFilepath)

    #Put all variables into a list    
    configVariables = [subject, cam_names, base_cam_index, lenFrame, startframe, include_DLC,include_OpenPoseFace,include_OpenPoseHands,include_OpenPoseSkeleton,useCheckerBoardVid,calibrateCameras, DLCfilepath, baseProjectPath, baseFilePath,portraitMode, numCamera]
    #Return list of variables
    return configVariables 

def GetVariables():
    '''This Function is for other scripts of code to access variables without having to open the GUI'''
    with open('ProjectFoldersConfig.csv','r') as f:#Open the csv that stores the variables
        reader = csv.reader(f)#Read the variables
        config = list(reader)#Put variables into list
    #Name the variables
    config = config[0]
    subject = config[0]
    date = config[1]
    project = config[2]
    session_num = config[3]
    baseProjectPath = config[4]
    DLCfilepath = config[5]
    numCameras = config[6]
    #Bool values are assigned T/F through if statements since they are saved as 0 or 1    
    if config[6] == '0':
        useCheckerBoardVid = False
    else:
        useCheckerBoardVid = True
    if config[7] == '0':
        calibrateCameras = False
    else:
        calibrateCameras = True
    if config[8] == '0':
        portraitMode = False
    else:
        portraitMode = True

    with open('runProject.csv', 'r') as f:#Open the csv that stores the variables
        reader = csv.reader(f)#Read the variables
        config2 = list(reader)#Put variables into list
    #Name the variables
    config2 = config2[0]
    if config2[2] == '0' and config2[3] =='0':
        cam_names = [config2[0],config2[1]]
    elif config2[2] != '0' and config2[3] == '0':
        cam_names = [config2[0],config2[1],config2[2]]
    else:
        cam_names = [config2[0], config2[1], config2[2], config2[3]]
    base_cam_index = config2[4]
    startframe  = config2[5]
    lenFrame= config2[6]
    #Bool values are assigned T/F through if statements since they are saved as 0 or 1    
    if config2[7] =='0':
        include_DLC = False 
    else:
        include_DLC = True
    if config2[8] =='0':
        include_OpenPoseFace = False 
    else:
        include_OpenPoseFace = True
    if config2[9] =='0':
        include_OpenPoseHands = False 
    else:
        include_OpenPoseHands = True
    if config2[10] =='0':
        include_OpenPoseSkeleton = False 
    else:
        include_OpenPoseSkeleton = True
    
    sessionID =  project+session_num+'_'+date
    baseFilePath = baseProjectPath+'/'+subject+'/'+sessionID         
    #Place Variables intp a list
    configVariables = [subject, cam_names, base_cam_index, lenFrame, startframe, include_DLC,include_OpenPoseFace,include_OpenPoseHands,include_OpenPoseSkeleton,useCheckerBoardVid,calibrateCameras, DLCfilepath, baseProjectPath, baseFilePath,portraitMode, numCameras]
    return configVariables  

