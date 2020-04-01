import csv, json, sys
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import os 


def Parse_OpenPose():
    fileDict = baseProjectPath+'/'+subject+'/'+sessionID+'/Openpose/' #Openpose pixel coord data file path
    if not os.path.exists( fileDict+'OpenPoseOutput'):
        os.mkdir(fileDict+'OpenPoseOutput')

#fileDict ='214OP_data/CamB_OpenPoseOutput' #Openpose pixel coord data file path
#ret = []

    openPoseOutputFolders = glob.glob(fileDict+'/*')

    for cam in openPoseOutputFolders:
        ret = []
        for f in os.listdir(cam):
            inputFile = open(os.path.join(fileDict,cam)) #open json file
            data = json.load(inputFile) #load json content
            inputFile.close() #close the input file


            skeleton = np.array(data['people'][-1]['pose_keypoints_2d']).reshape((-1,3))
            #hand_left = np.array(data['people'][-1]["hand_left_keypoints_2d"]).reshape((-1,3))
            hand_right = np.array(data['people'][-1]["hand_right_keypoints_2d"]).reshape((-1,3))
    

            d = np.concatenate((skeleton,hand_right),axis = 0)
            ret.append(skeleton)

        ret = np.array(ret)
        #should be (1451,46,3)
        print(ret.shape)
        np.save(fileDict+'OpenPoseOutput/OP_'+cam_num+'.npy',ret)



