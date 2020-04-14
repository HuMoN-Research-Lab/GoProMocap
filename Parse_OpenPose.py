import csv, json, sys
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import os 
from config import baseFilePath, cam_names, points_inFrame
import glob


def Parse_OpenPose():
    
    OPfileDict = baseFilePath+'/Intermediate/Openpose/' #Openpose pixel coord data file path
    fileDict = baseFilePath+'/Intermediate/'

    
    
    if not os.path.exists( fileDict+'OpenPoseOutput'):
        os.mkdir(fileDict+'OpenPoseOutput')
    outputfileDict = fileDict + 'OpenPoseOutput'
    openPoseOutputFolders = glob.glob(OPfileDict+'/*')
    
    j = 0
    for cam in openPoseOutputFolders:
        cam_name = cam_names

        ret = []
        filelist = sorted(os.listdir(cam))
        inputFile = open(os.path.join(cam,filelist[0])) #open json file
        data = json.load(inputFile) #load json content
        inputFile.close() #close the input file
        target_skeleton = np.array(data['people'][0]['pose_keypoints_2d']).reshape((-1,3))

        for f in os.listdir(cam):   
            inputFile = open(os.path.join(cam,f)) #open json file
            data = json.load(inputFile) #load json content
            inputFile.close() #close the input file
            if len(data['people'] == 0):
                c = 10000000000
                res = 0
                RH = 0
                LH = 0
                for people in data['people']:
                    skeleton = np.array(people['pose_keypoints_2d']).reshape((-1,3))
                    hand_left = np.array(people["hand_left_keypoints_2d"]).reshape((-1,3))
                    hand_right = np.array(people["hand_right_keypoints_2d"]).reshape((-1,3))
                    distance = sum(sum(abs(target_skeleton-skeleton)))
                    if distance < c:
                        c = distance
                        res = skeleton 
                        HL = hand_left
                        HR = hand_right 
                target_skeleton = res
        
                d = np.concatenate((skeleton,HR,HL),axis = 0)
                ret.append(d)

            if len(data['people']) == 0:
                a = np.empty((points_inFrame,3))
                a[:] = np.nan
                ret.append(a)
        
        ret = np.array(ret)
        #should be (1451,46,3)
        print(ret.shape)
        np.save(outputfileDict+'/OP_'+cam_names[j]+'.npy',ret)
        j= j+1


