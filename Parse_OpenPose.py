import csv, json, sys
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import os 
from config import  cam_names, include_OpenPoseFace, include_OpenPoseHands, include_OpenPoseSkeleton
import glob
import h5py
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

    k = 0#Initialize counter
    
    with h5py.File(outputfileDict + '/OpenPoseOutput.hdf5', 'r') as f:
        allCameras = f.get('Cameras')
        for camera in range(len(allCameras)):
            ret = []#intialize an array to store each json file
            target_skeleton = f.get('Cameras/'+str(cam_names[camera])+'/Frame0/Person0/Skeleton')
            target_skeleton = target_skeleton[()]
            allFrames = f.get('Cameras/'+str(cam_names[camera]))
                                  
            for frame in range(len(allFrames)):
                allPeople = f.get('Cameras/'+str(cam_names[camera])+'/Frame'+str(frame))
                if len(allPeople) == 0:
                    noPersonInFrame.append(j) 
                    a = np.empty((points_inFrame,3))
                    a[:] = np.zeros
                    ret.append(a)
                else:
                    c = 10000000
                    res = 0
                    for person in range(len(allPeople)):                                      
                        if include_OpenPoseSkeleton:#If you include skeleton
                            skeleton  = f.get('Cameras/'+str(cam_names[camera])+'/Frame'+str(frame)+'/Person'+str(person)+'/Skeleton')  
                            skeleton = skeleton[()]
                        if include_OpenPoseHands: #If you include hands
                            hand_left = f.get('Cameras/'+str(cam_names[camera])+'/Frame'+str(frame)+'/Person'+str(person)+'/Left Hand')
                            hand_left = [()]
                            hand_right = f.get('Cameras/'+str(cam_names[camera])+'/Frame'+str(frame)+'/Person'+str(person)+'/Right Hand')
                            hand_right = [()]
                        if include_OpenPoseFace:#If you include face
                            face = f.get('Cameras/'+str(cam_names[camera])+'/Frame'+str(frame)+'/Person'+str(person)+'/Face')
                            face = [()]
                        
                        distance = sum(sum(abs(target_skeleton-skeleton))) #Calculate the distance of the person in this frame compared to the target person from last frame

                        if distance < c: #If the distance is less than the threshold than this person is the target skeleton
                            c = distance #the distance becomes threshold
                            
                            if include_OpenPoseHands:
                                if include_OpenPoseFace:
                                    HL = hand_left
                                    HR = hand_right
                                    newFace = face     
                                    res = skeleton
                                    fullPoints = np.concatenate((res,HL,HR,newFace),axis = 0)
                                else:
                                    HL = hand_left
                                    HR = hand_right
                                    res = skeleton     
                                    fullPoints = np.concatenate((res,HL,HR),axis = 0)
                            else:   
                                if include_OpenPoseFace:
                                    newFace = face
                                    res = skeleton     
                                    fullPoints = np.concatenate((res,newFace),axis = 0)
                                else:
                                    res = skeleton        
                                    fullPoints = skeleton
                                
                            target_skeleton = res        
                    ret.append(fullPoints)

                ret = np.array(ret)
                print(ret.shape)
                np.save(outputfileDict+'/OP_'+cam_names[k]+'.npy',ret)
                np.savetxt(outputfileDict+'/OP_'+cam_names[k]+'.txt',ret[:,8,:])
                k  = k+1
    return noPersonInFrame


