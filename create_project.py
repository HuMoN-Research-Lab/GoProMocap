import os
from config import session_num, project, date, subject, baseProjectPath, calibrateCameras


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

def create_project():
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


    print("A project has now been created in the specified base file path.")
    print("Place raw videos in the following file path:")
    print("(base file path)/projectname/raw/RawGoProVideo")
    print("Optional: Place checkerboard videos in the following file path:")
    print("(base file path)/projectname/raw/Checkerboard")
    input("Press enter when finished moving videos to correct folder")
    if calibrateCameras == True:
        print('Place Calibration Videos into the following path')
        print('(base file path)/SubjectIntials/Calibration/CalibrationVideos')
        input('Press enter when finished')
        

