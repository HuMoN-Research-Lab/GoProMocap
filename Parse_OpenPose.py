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
    j =0#Intialize Counter
    with h5py.File(outputfileDict + '/OpenPoseh5Output.hdf5', 'r') as f:#Open h5 file with openpose output in it
        allCameras = f.get('Cameras')#Open camera group
        for camera in range(len(allCameras)):#Iterate through camera group
            ret = []#create an empty list to add the points to 
            target_skeleton = f.get('Cameras/'+str(cam_names[camera])+'/Frame0/Person0/Skeleton') #Get the first skeleton in the first frame a
            target_skeleton = target_skeleton[()]#Access data
            allFrames = f.get('Cameras/'+str(cam_names[camera]))#Open frame group
            #print(target_skeleton[:,2])                   
            distancep =[]
            for frame in range(len(allFrames)):#iterate through all frames
                
                allPeople = f.get('Cameras/'+str(cam_names[camera])+'/Frame'+str(frame))#Open person group
                if len(allPeople) == 0:# If the amount of people in frame is zero
                    noPersonInFrame.append(j) #Add that frame to the list of frames with no person in list
                    a = np.empty((points_inFrame,3))#create an array the size of all other frames
                    a[:] = np.nan#Fill th array with nans
                    ret.append(a)#Add this frame to full video list 
                    j+=1    
                else:#If there are people in frame
                    c = 10000000#Intialize random large number
                    res = 0 #Intialize variable
                    for person in range(len(allPeople)):#iterates through each person in frame                                      
                        diffPoints =[]
                        if include_OpenPoseSkeleton:#If you include skeleton
                            skeleton  = f.get('Cameras/'+str(cam_names[camera])+'/Frame'+str(frame)+'/Person'+str(person)+'/Skeleton')  #Gets the skeleton Openpose data
                            skeleton = skeleton[()]#Acceses skeleton data
                        if include_OpenPoseHands: #If you include hands
                            hand_left = f.get('Cameras/'+str(cam_names[camera])+'/Frame'+str(frame)+'/Person'+str(person)+'/LeftHand')# Gets the left hand openpose data
                            hand_left = hand_left[()]# access left hand data
                            hand_right = f.get('Cameras/'+str(cam_names[camera])+'/Frame'+str(frame)+'/Person'+str(person)+'/RightHand')#Gets the right hand data
                            hand_right = hand_right[()]# access right hand data
                        if include_OpenPoseFace:#If you include face
                            face = f.get('Cameras/'+str(cam_names[camera])+'/Frame'+str(frame)+'/Person'+str(person)+'/Face') # gets left hand data
                            face = face[()]# access face data
                        
                        distance = (sum(abs(target_skeleton-skeleton))) #Calculate the distance of the person in this frame compared to the target person from last frame
                        #distancey = (sum(abs(target_skeleton[:,1]-skeleton[:,1])))
                        #distance = np.sqrt((float(distancex)**2)*(float(distancey)**2))
                        #distance = np.sqrt((float(distancex)**2)*(float(distancey)**2))
                      #  for jj in range(len(skeleton)):
                       #     for ii in range(2):
                        #        difference = abs(skeleton[jj,ii] - target_skeleton[jj,ii])
                        #        diffPoints.append(difference)
                        #distance = sum(diffPoints) 
                        distancep.append(sum(abs(target_skeleton[:,2]-skeleton[:,2])))       
                        #if distance < c: #If the distance is less than the threshold than this person is the target skeleton
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
            np.save(outputfileDict+'/OP_'+cam_names[k]+'.npy',ret)#Save out data
            np.savetxt(outputfileDict+'/OP_'+cam_names[k]+'.txt',ret[:,0,:])
            k  = k+1
    return noPersonInFrame
Parse_OpenPose()

