import os
from config import session_num, project, date, subject, baseProjectPath


sessionID =  project+session_num+'_'+date

baseFilePath = baseProjectPath+'/'+subject+'/'+sessionID
rawData = baseFilePath+'/Raw'
checkerVideoFolder = rawData+'/Checkerboard'
rawVideoFolder = rawData+'/RawGoProData'
calibrationFilePath = baseProjectPath +'/'+subject+'/Calibration'
cameraParamsFilePath = calibrationFilePath +'/CameraParams'
calibVideoFilepath = calibrationFilePath +'/CalibrationVideos'

def create_project():
    #Create Folders for Project
    if not os.path.exists(baseProjectPath+'/'+subject):
        os.mkdir(baseProjectPath+'/'+subject)

    if not os.path.exists(baseProjectPath+'/'+subject+'/'+sessionID):
        os.mkdir(baseProjectPath+'/'+subject+'/'+sessionID)
    
    if not os.path.exists(baseFilePath+'/Raw'):
        os.mkdir(baseFilePath+'/Raw')
    
    if not os.path.exists(rawData+'/RawGoProData'):
        os.mkdir(rawData+'/RawGoProData')
    
    if not os.path.exists(rawData+'/Checkerboard'):
        os.mkdir(rawData+'/Checkerboard')

    if not os.path.exists(calibrationFilePath):
        os.mkdir(calibrationFilePath)
    
    if not os.path.exists(cameraParamsFilePath):
        os.mkdir(cameraParamsFilePath)

    if not os.path.exists(calibVideoFilepath):
        os.mkdir(calibVideoFilepath)


