import csv, json, sys
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import os 
from config import  cam_names, include_OpenPoseFace, include_OpenPoseHands, include_OpenPoseSkeleton
import glob
from create_project import baseFilePath

#========================================== Set Variable for amount of openpose points
if include_OpenPoseFace:
    points_from_face = 70
else:
    points_from_face = 0

if include_OpenPoseHands:
    points_from_Hands = 42
else:
    points_from_Hands = 0 

if include_OpenPoseSkeleton:
    points_from_skeleton = 25
else:
    points_from_skeleton = 0

points_inFrame = points_from_skeleton + points_from_Hands + points_from_face

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
    noPersonInFrame =[]


    
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
            c = 10000000000
            res = 0
            j = j+1 
            if (len(data['people'] )) == 0: 
                noPersonInFrame.append(j) 
                a = np.empty((points_inFrame,3))
                a[:] = np.zeros
                ret.append(a)
            else:
                if include_OpenPoseSkeleton:
                    for people in data['people']:
                        skeleton = np.array(people['pose_keypoints_2d']).reshape((-1,3))
                        distance = sum(sum(abs(target_skeleton-skeleton)))
                        if distance < c:
                            c = distance
                            res = skeleton 
                    target_skeleton = res
                ret.append(res)

        ret = np.array(ret)
        print(ret.shape)
        np.save(outputfileDict+'/OP_'+cam_names[k]+'.npy',ret)
        k  = k+1
    return noPersonInFrame


if include_OpenPoseHands:       
                    hand_left = np.array(data['people'][0]["hand_left_keypoints_2d"]).reshape((-1,3))
                    hand_right = np.array(data['people'][0]["hand_right_keypoints_2d"]).reshape((-1,3))
                else:
                    hand_left = []
                    hand_right =[]
                if include_OpenPoseFace:
                    face = np.array(data['people'][0]["face_keypoints_2d"]).reshape((-1,3))
                else:
                    face = []
                d = np.concatenate((skeleton,hand_left,hand_right),axis = 0)
                ret.append(d)'''
