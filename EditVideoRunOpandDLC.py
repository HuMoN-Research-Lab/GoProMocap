import os
import h5py
import subprocess
import json
import numpy as np
import pandas as pd
import ffmpeg
import cv2
#import deeplabcut
from config import DLCconfigPath,  cam_names,  num_of_cameras,baseProjectPath, include_OpenPoseFace, include_OpenPoseSkeleton, include_OpenPoseHands
from create_project import baseFilePath, rawData, checkerVideoFolder, rawVideoFolder
import glob

def getCameraParams(filepath):
    amountOfCalImages = 7
    calibDatadir  = [filepath+'/CalibrationVideos']
    for dir in calibDatadir:
        k = 0 
        for video in os.listdir(dir):
            
            vidcap = cv2.VideoCapture(filepath+'/CalibrationVideos/'+video)
            frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            calImagesinVideo = frame_count/amountOfCalImages
            vidlength = range(int(frame_count)) 
            for ii in vidlength:

                success,image = vidcap.read()
                if success:
                    height , width , layers =  image.shape 
                    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    #single_video.append(image)   
                    if not os.path.exists(filepath + '/'+cam_names[k]+'_CalibrationImages'):
                        os.mkdir(filepath + '/'+cam_names[k]+'_CalibrationImages')                       
                    cv2.imwrite(filepath+'/'+cam_names[k]+'_CalibrationImages/frame%d.jpg' %ii , image)     # save frame as JPEG file    
                else:
                    continue
            # termination criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            objp = np.zeros((9*6,3), np.float32)
            objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)

            # Arrays to store object points and image points from all the images.
            objpoints = [] # 3d point in real world space
            imgpoints = [] # 2d points in image plane.

            images = glob.glob(filepath+'/'+cam_names[k]+'_CalibrationImages/*.jpg')
            p = 0
            for fname in images[::int(calImagesinVideo-1)]:
                img = cv2.imread(fname)
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                # Find the chess board corners
                ret, corners = cv2.findChessboardCorners(img, (6,9),None)
                # If found, add object points, image points (after refining them)
                if ret == True:
                    objpoints.append(objp)

                    corners2 = cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
                    imgpoints.append(corners2)
                
                    # Draw and display the corners
                    img = cv2.drawChessboardCorners(img, (6,9), corners2,ret)
                    cv2.imshow('img',img)
                    cv2.waitKey(50000)
                p+=1

            print(p)
            cv2.destroyAllWindows()

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
            if not os.path.exists(filepath+'/CameraParams' + '/'+cam_names[k]):
                os.mkdir(filepath+'/CameraParams' + '/'+cam_names[k])   

            np.save(filepath+'/CameraParams' + '/'+cam_names[k]+'/Calibration_ret.npy',ret)
            np.save(filepath+'/CameraParams' + '/'+cam_names[k]+'/Calibration_mtx.npy',mtx)
            np.save(filepath+'/CameraParams' + '/'+cam_names[k]+'/Calibration_dist.npy',dist)
            np.save(filepath+'/CameraParams' + '/'+cam_names[k]+'/Calibration_rvec.npy',rvecs)
            np.save(filepath+'/CameraParams' + '/'+cam_names[k]+'/Calibration_tvecs.npy',tvecs)
            

            tot_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                tot_error += error
            meanError = tot_error/len(objpoints)
            print(cam_names[k])
            print ("mean error: ", meanError)
            print(tot_error)
            k +=1

#Concate Videos
def concatVideos(filepath):
    cam1vids = open(filepath+'/cam1vids.txt','a')
    cam2vids = open(filepath+'/cam2vids.txt','a')
    cam3vids = open(filepath+'/cam3vids.txt','a')
    cam4vids = open(filepath+'/cam4vids.txt','a')
    for dir in [rawVideoFolder]: #for loop parses through the resized video folder 
        for video in os.listdir(dir): 
            #Get length of the name of cameras
            cam1length = len(cam_names[0]); cam2length = len(cam_names[1]); cam3length = len(cam_names[2]); cam4length = len(cam_names[3]); 
            if video[:cam1length] == cam_names[0]: # if the video is from Cam1
                cam1vids.write('file'+" '" +'\\'+video+"'") #write the file name of the video to the text file
                cam1vids.write('\n')                     
            if video[:cam2length] == cam_names[1]: # if the video is from Cam2
                cam2vids.write('file'+" '" +'\\'+video+"'") #write the file name of the video to the text file
                cam2vids.write('\n')                   
            if video[:cam3length] == cam_names[2]: # if the video is from Cam3
                cam3vids.write('file'+" '" +'\\'+video+"'") #write the file name of the video to the text file
                cam3vids.write('\n') 
            if video[:cam4length] == cam_names[3]: # if the video is from Cam4
                cam4vids.write('file'+" '" +'\\'+video+"'") #write the file name of the video to the text file
                cam4vids.write('\n')                     
    #Close the text files
    cam1vids.close()
    cam2vids.close()
    cam3vids.close()
    cam4vids.close()
    #Use ffmpeg to join all parts of the video together
    for jj in range(len(cam_names)):
        (ffmpeg
        .input(filepath+'/'+cam_names[jj]+'vids.txt', format='concat', safe=0)
        .output(filepath+'/'+cam_names[jj]+'.mp4', c='copy')
        .run()
        )
    #subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', filepath+'/cam1vids.txt', '-c' ,'copy' ,filepath+'/'+ cam1+'.mp4'])
    #subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', filepath+'/cam2vids.txt', '-c' ,'copy' ,filepath+'/'+ cam2+'.mp4'])
    #subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', filepath+'/cam3vids.txt', '-c' ,'copy' ,filepath+'/'+ cam3+'.mp4'])
    #subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', filepath+'/cam4vids.txt', '-c' ,'copy' ,filepath+'/'+ cam4+'.mp4'])


#################### Undistortion #########################
def undistortVideos(Inputfilepath,Outputfilepath):
    for dir in Inputfilepath:
        for video in os.listdir(dir):
            subprocess.call(['ffmpeg', '-i', Inputfilepath+'/'+video, '-vf', "lenscorrection=cx=0.5:cy=0.5:k1=-.115:k2=-0.022", Outputfilepath+'/'+video])

def trimVideos(Inputfilepath,OutputFilepath):    
    vidcap = cv2.VideoCapture(Inputfilepath)#Open video
    vidWidth  = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH) #Get video height
    vidHeight = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) #Get video width
    video_resolution = (int(vidWidth),int(vidHeight)) #Create variable for video resolution
    vidLength = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    vidfps = vidcap.get(cv2.CAP_PROP_FPS)
    success,image = vidcap.read() #read a frame
    count = [] #Intialize a counter variable
    avggray = []
    maxfirstGray = 0
    maxsecondGray = 0 
    
    for jj in range(int(vidLength)):
        
        success,image = vidcap.read() #read a frame
        if success: #If frame is correctly read
            if jj < int(vidLength/3):
                resize = cv2.resize(image, video_resolution) #Set image to same resolution of video
            
                gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                if np.average(gray) > maxfirstGray:
                    maxfirstGray = np.average(gray)
                    firstFlashFrame = jj
            if jj > int((2*vidLength)/3):
                resize = cv2.resize(image, video_resolution) #Set image to same resolution of video
            
                gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                if np.average(gray) > maxsecondGray:
                    maxsecondGray = np.average(gray)
                    secondFlashFrame = jj

        else:
            continue
        input1 = ffmpeg.input()

        node1_1 = input1.trim(start_frame=firstFlashFrame,end_frame=secondFlashFrame).setpts('PTS-STARTPTS')
        node1_1.output(OutputFilepath+'/'+cam_names[ii]).run()

def runDeepLabCut(Inputfilepath,OutputFilepath):
    #####################Copy Videos to DLC Folder############
    for dir in [Inputfilepath]:
        for video in os.listdir(dir):
            subprocess.call(['ffmpeg', '-i', Inputfilepath+'/'+video,  OutputFilepath+'/'+video])


    #################### DeepLabCut ############################
    for dir in [OutputFilepath]:# Loop through the undistorted folder
        for video in os.listdir(dir):
            #Analyze the videos through deeplabcut
            deeplabcut.analyze_videos(baseProjectPath+'/'+DLCconfigPath, [OutputFilepath +'/'+ video], save_as_csv=True)
            deeplabcut.plot_trajectories(baseProjectPath+'/'+DLCconfigPath,[OutputFilepath +'/'+ video])

    for dir in [OutputFilepath]:
        for video in dir:   
            deeplabcut.create_labeled_video(baseProjectPath+'/'+DLCconfigPath, glob.glob(os.path.join(OutputFilepath ,'*mp4')))

    if not os.path.exists(OutputFilepath + 'DLCnpy'):
        os.mkdir(OutputFilepath+ 'DLCnpy')

    #Load all dlc csv output files  
    csvfile = glob.glob(OutputFilepath+'/*csv')

    #For loop gets csv data from all cameras
    j = 0
    for data in csvfile:     
        datapoints = pd.read_csv(data) # read in the csv data 
        parsedDlcData = datapoints.iloc[3:,7:10].values#the last element in the array is the P value

        print(parsedDlcData.shape)
        np.save(OutputFilepath+'DLCnpy/dlc_'+cam_names[j]+'.npy',parsedDlcData)#Save data
        j = j+1

def runOpenPose(Inputfilepath,VideoOutputPath,DataOutputFilepath):
    ###################### OpenPose ######################################
    os.chdir("C:/Users/MatthisLab/openpose") # change the directory to openpose
    j = 0
    for dir in [Inputfilepath]:# loop through undistorted folder
        for video in os.listdir(dir):
            subprocess.call(['bin/OpenPoseDemo.exe', '--video', Inputfilepath+'/'+video, '--hand','--face','--write_video', VideoOutputPath+'/OpenPose'+cam_names[j]+'.avi',  '--write_json', DataOutputFilepath+'/'+cam_names[j]])
            j =+1



    ########## Put Openpose Data into h5   ######################
    
def Parse_Openpose(Inputfilepath,OutputFilepath):
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
    j = 0 #Counter variable
    with  h5py.File(OutputFilepath + '/OpenPoseh5Output.hdf5', 'w') as f:
        cams = f.create_group('Cameras')
        for cam in os.listdir(Inputfilepath):# Loops through each camera
            k =0
            cameraGroup = cams.create_group(cam_names[j])
            for files in os.listdir(Inputfilepath+'/'+cam): #loops through each json file   
                fileGroup = cameraGroup.create_group('Frame'+str(k))
                inputFile = open(Inputfilepath+'/'+cam+'/'+files) #open json file
                data = json.load(inputFile) #load json content
                inputFile.close() #close the input file
                ii = 0 
                for people in data['people']:
                    skeleton = np.array(people['pose_keypoints_2d']).reshape((-1,3))
                    hand_left = np.array(people['hand_left_keypoints_2d']).reshape((-1,3))
                    hand_right = np.array(people['hand_right_keypoints_2d']).reshape((-1,3))
                    face = np.array(people['face_keypoints_2d']).reshape((-1,3))  #Get skeleton points

                    persongroup = fileGroup.create_group('Person'+str(ii))
                    skeletondata = persongroup.create_dataset('Skeleton', data =skeleton)
                    rightHanddata = persongroup.create_dataset('RightHand', data =hand_right) 
                    leftHanddata = persongroup.create_dataset('LeftHand', data =hand_left)
                    facedata = persongroup.create_dataset('Face', data =face)                                       
                    ii = ii +1 
                k= k +1
            j = j + 1

    #Create a list variable to store all frame numbers where there is no person in frame
    noPersonInFrame =[]

    k = 0#Initialize counter
    
    with h5py.File(OutputFilepath+ '/OpenPoseh5Output.hdf5', 'r') as f:
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
            np.save(OutputFilepath+'/OP_'+cam_names[k]+'.npy',ret)
            #np.savetxt(outputfileDict+'/OP_'+cam_names[k]+'.txt',ret[:,8,:])
            k  = k+1
    return noPersonInFrame


''' 
FIX This Put in Undistorting function
###############If you need To use checkerboard videos##################

if useCheckerboardVid:
    checkerDatadir = [checkerVideoFolder]   

#Create a folder for the undistorted videos
    if not os.path.exists(interfilepath + '/CheckerboardUndistorted'):
        os.mkdir(interfilepath + '/CheckerboardUndistorted')
    checkerUndistortFilepath = interfilepath + '/CheckerboardUndistorted'
    
    for dir in checkerDatadir:
        for video in os.listdir(dir):
            subprocess.call(['ffmpeg', '-i', checkerVideoFolder+'/'+video, '-vf', "lenscorrection=cx=0.5:cy=0.5:k1=-.115:k2=-0.022", checkerUndistortFilepath+'/'+video])
'''