import csv, json, sys
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import os 
from config import baseFilePath, cam_names, points_inFrame
import glob


def Parse_OpenPose():
    #Creat variables for file path
    OPfileDict = baseFilePath+'/Intermediate/Openpose/' 
    fileDict = baseFilePath+'/Intermediate/'

    
    #make a folder for the parsed openpose output
    if not os.path.exists( fileDict+'OpenPoseOutput'):
        os.mkdir(fileDict+'OpenPoseOutput')
    outputfileDict = fileDict + 'OpenPoseOutput'
    
    #Create a variable for all the cameras openpose data
    openPoseOutputFolders = glob.glob(OPfileDict+'/*')
    
    #Create a list variable to store all frame numbers where there is no person in frame
    empty_frame =[]

    #
    for cam in openPoseOutputFolders:
        j = 0
        for f in os.listdir(cam):
            inputFile = open(os.path.join(cam,f)) #open json file
            data = json.load(inputFile) #load json content
            inputFile.close() #close the input file
            if (len(data['people'] )) == 0: 
                empty_frame.append(j)
            j = j +1
    k = 0
    for cam in openPoseOutputFolders:
        cam_name = cam_names

        ret = []
        filelist = sorted(os.listdir(cam))
        inputFile = open(os.path.join(cam,filelist[0])) #open json file
        data = json.load(inputFile) #load json content
        inputFile.close() #close the input file
        target_skeleton = np.array(data['people'][0]['pose_keypoints_2d']).reshape((-1,3))
        j = 0 
        for f in os.listdir(cam):   
            inputFile = open(os.path.join(cam,f)) #open json file
            data = json.load(inputFile) #load json content
            inputFile.close() #close the input file
            j = j+1 
            if j-1 in empty_frame: 
                continue
            else:
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

        ret = np.array(ret)
        #should be (1451,46,3)
        print(ret.shape)
        np.save(outputfileDict+'/OP_'+cam_names[k]+'.npy',ret)
        k  = k+1
        print(k)
    return empty_frame

Parse_OpenPose()