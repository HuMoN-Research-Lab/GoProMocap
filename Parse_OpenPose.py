import csv, json, sys
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import os 
from config import baseProjectPath, subject, sessionID
import glob


def Parse_OpenPose():
    
    OPfileDict = baseProjectPath+'/'+subject+'/'+sessionID+'/Intermediate/Openpose/' #Openpose pixel coord data file path
    fileDict = baseProjectPath+'/'+subject+'/'+sessionID+'/Intermediate/'

    if not os.path.exists( fileDict+'OpenPoseOutput'):
        os.mkdir(fileDict+'OpenPoseOutput')
    outputfileDict = fileDict + 'OpenPoseOutput'
    openPoseOutputFolders = glob.glob(OPfileDict+'/*')

    for cam in openPoseOutputFolders:
        cam_name = cam[len(OPfileDict):]

        ret = []
        for f in os.listdir(cam):   
            inputFile = open(os.path.join(cam,f)) #open json file
            data = json.load(inputFile) #load json content
            inputFile.close() #close the input file
            try:
                skeleton = np.array(data['people'][-1]['pose_keypoints_2d']).reshape((-1,3))
                #hand_left = np.array(data['people'][-1]["hand_left_keypoints_2d"]).reshape((-1,3))
                hand_right = np.array(data['people'][-1]["hand_right_keypoints_2d"]).reshape((-1,3))
              
                 
                d = np.concatenate((skeleton,hand_right),axis = 0)
                ret.append(skeleton)
            except IndexError:
                a = np.empty((25,3))
                a[:] = np.nan
                ret.append(a)
        
        ret = np.array(ret)
        #should be (1451,46,3)
        print(ret.shape)
        np.save(outputfileDict+'/OP_'+cam_name+'.npy',ret)
Parse_OpenPose()


