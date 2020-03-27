import csv, json, sys
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import os 



fileDict ='214OP_data/CamB_OpenPoseOutput' #Openpose pixel coord data file path
ret = []


for f in sorted(os.listdir(fileDict)):
    inputFile = open(os.path.join(fileDict,f)) #open json file
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
np.save('214OP_data/OP_CamB.npy',ret)



