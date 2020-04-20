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

    
    
    
    openPoseOutputFolders = glob.glob(OPfileDict+'/*')
    
    
    empty  = []
    for cam in openPoseOutputFolders:
        filelist = sorted(os.listdir(cam))
        inputFile = open(os.path.join(cam,filelist[0])) #open json file
        data = json.load(inputFile) #load json content
        inputFile.close()
        j = 0
        for f in os.listdir:
            inputFile = open(os.path.join(cam,f)) #open json file
            data = json.load(inputFile) #load json content
            inputFile.close() #close the input file
            if len(data['people'] == 0):
                empty.append[j]
            j = j+1

        cam_name = cam_names

        ret = []
        filelist = sorted(os.listdir(cam))
        inputFile = open(os.path.join(cam,filelist[0])) #open json file
        data = json.load(inputFile) #load json content
        inputFile.close() #close the input file
        target_skeleton = np.array(data['people'][0]['pose_keypoints_2d']).reshape((-1,3))
        
        for f in os.listdir(cam):   
            print(f)
            inputFile = open(os.path.join(cam,f)) #open json file
            data = json.load(inputFile) #load json content
            inputFile.close() #close the input file
            if j in empty:
                continue
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
                empty.append[j]
                continue
    
            j =j+1
                
        
        ret = np.array(ret)
        print(ret.shape)
        np.save(outputfileDict+'/OP_'+cam_names[j]+'.npy',ret)
        j= j+1

Parse_OpenPose()

