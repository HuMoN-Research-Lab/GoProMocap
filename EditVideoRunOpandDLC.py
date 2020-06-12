import os
import h5py
import subprocess
import json
import numpy as np
import pandas as pd
import ffmpeg
#from pykalman import KalmanFilter
import cv2
#import deeplabcut
#from config import DLCconfigPath,  cam_names,  num_of_cameras,baseProjectPath, include_OpenPoseFace, include_OpenPoseSkeleton, include_OpenPoseHands, portraitMode
from create_project import GetVariables
import glob


configVariables = GetVariables()
baseFilePath = configVariables[13]
cam_names = configVariables[1]
baseProjectPath = configVariables[12] 
DLCconfigPath = configVariables[11]
include_OpenPoseFace = configVariables[6]
include_OpenPoseHands = configVariables[7]
include_OpenPoseSkeleton = configVariables[8]
portraitMode = configVariables[14]

rawVideoFolder = baseFilePath+'/Raw'
def getCameraParams(filepath):
    '''Functions input is the filepath to the calibration folder. 
    The function utilizes opencv functions to find camera parameters based on calibration videos
    Saves the camera parameters to an output folder with a different npy file for each parameter
    '''
    amountOfCalImages = 7 #How many images to take from the videos, only really works between 5 and 10 
    calibDatadir  = [filepath+'/CalibrationVideos'] #Directory of calibration videos
    for dir in calibDatadir:#Iterates through the calibration folder
        k = 0 #Counter variable
        for video in os.listdir(dir):#Iterates through each video in folder 
            vidcap = cv2.VideoCapture(filepath+'/CalibrationVideos/'+video) #Read in video
            frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) #find frame count of video 
            calImagesinVideo = frame_count/amountOfCalImages #The number to iterate through the images in video
            vidlength = range(int(frame_count)) #Create list for loop
            for ii in vidlength:#Iterates through each frame of video

                success,image = vidcap.read()#reads in frame 
                if success:# If it successfully reads in a frame
                    height , width , layers =  image.shape # Get Shape of image 
                    if not os.path.exists(filepath + '/'+cam_names[k]+'_CalibrationImages'):# create a folder for calibration frames if there isnt one yet 
                        os.mkdir(filepath + '/'+cam_names[k]+'_CalibrationImages')                       
                    cv2.imwrite(filepath+'/'+cam_names[k]+'_CalibrationImages/frame%d.jpg' %ii , image)     # save frame as JPEG file    
                else: # If the frame is not successfully read
                    continue # Continue
            # termination criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            objp = np.zeros((9*6,3), np.float32)
            objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)

            # Arrays to store object points and image points from all the images.
            objpoints = [] # 3d point in real world space
            imgpoints = [] # 2d points in image plane.
        
            images = glob.glob(filepath+'/'+cam_names[k]+'_CalibrationImages/*.jpg')#Load in all images
            p = 0#Counter variables
            for fname in images[::int(calImagesinVideo-1)]:# Iterate through the amount of calibration images
                img = cv2.imread(fname)#Load image
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#Convert to grayscale

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
                    cv2.waitKey(1000)
                p+=1

            cv2.destroyAllWindows()#Close windows

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)#Get intrinsics based on chessboard
            if not os.path.exists(filepath+'/CameraParams/'+cam_names[k]):#Create Camera parameter folder for each camera
                os.mkdir(filepath+'/CameraParams/' +cam_names[k])   

            #Save out the Parameters 
            np.save(filepath+'/CameraParams' + '/'+cam_names[k]+'/Calibration_ret.npy',ret)
            np.save(filepath+'/CameraParams' + '/'+cam_names[k]+'/Calibration_mtx.npy',mtx)
            np.save(filepath+'/CameraParams' + '/'+cam_names[k]+'/Calibration_dist.npy',dist)
            np.save(filepath+'/CameraParams' + '/'+cam_names[k]+'/Calibration_rvec.npy',rvecs)
            np.save(filepath+'/CameraParams' + '/'+cam_names[k]+'/Calibration_tvecs.npy',tvecs)
            
            #Find reprojection Error and Print to check if the calibration is decent 
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

#Concat Videos
def concatVideos(InputFilePath,OutputFilepath):
    '''Functions input is filepath is path to raw video folder
    If the videos in the folder are multiple parts the function uses ffmpeg to concat the video parts together
    It saves the concated video to an output folder 
    '''

    #Create a txt file for names of video parts 
    cam1vids = open(InputFilePath+'/cam1vids.txt','a')
    cam2vids = open(InputFilePath+'/cam2vids.txt','a')
    cam3vids = open(InputFilePath+'/cam3vids.txt','a')
    cam4vids = open(InputFilePath+'/cam4vids.txt','a')
    for dir in [rawVideoFolder]: #for loop parses through the resized video folder 
        for video in os.listdir(dir): 
            #Get length of the name of cameras
            #cam1length = len(cam_names[0]); cam2length = len(cam_names[1]); cam3length = len(cam_names[2]); cam4length = len(cam_names[3]); 
            if len(cam_names) > 0:
                cam1length = len(cam_names[0])
                if video[:cam1length] == cam_names[0]: # if the video is from Cam1
                    cam1vids.write('file'+" '" +'\\'+video+"'") #write the file name of the video to the text file
                    cam1vids.write('\n')                     
            if len(cam_names) > 1:
                cam2length = len(cam_names[1])
                if video[:cam2length] == cam_names[1]: # if the video is from Cam2
                    cam2vids.write('file'+" '" +'\\'+video+"'") #write the file name of the video to the text file
                    cam2vids.write('\n')                   
            if len(cam_names) > 2:
                cam3length = len(cam_names[2])
                if video[:cam3length] == cam_names[2]: # if the video is from Cam3
                    cam3vids.write('file'+" '" +'\\'+video+"'") #write the file name of the video to the text file
                    cam3vids.write('\n') 
            if len(cam_names) > 3:
                cam4length = len(cam_names[3])
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
        .input(InputFilePath+'/cam'+str(jj+1)+'vids.txt', format='concat', safe=0)
        .output(OutputFilepath+'/'+cam_names[jj]+'.mp4', c='copy')
        .run()
        )
    #subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', filepath+'/cam1vids.txt', '-c' ,'copy' ,filepath+'/'+ cam1+'.mp4'])
    #subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', filepath+'/cam2vids.txt', '-c' ,'copy' ,filepath+'/'+ cam2+'.mp4'])
    #subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', filepath+'/cam3vids.txt', '-c' ,'copy' ,filepath+'/'+ cam3+'.mp4'])
    #subprocess.call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', filepath+'/cam4vids.txt', '-c' ,'copy' ,filepath+'/'+ cam4+'.mp4'])


#################### Undistortion #########################
def undistortVideos(Inputfilepath,Outputfilepath):
    '''Function input is raw distorted videos filepath and the filepath to save the videos to  
    Uses ffmpeg and camera intrinsics to undistort the video
    Outputs the undistorted video to the specified file path
    '''
    for jj in range(len(cam_names)):
        #Uses subprocess for a command line prompt to use ffmpeg to undistort video based on intrinsics 
        subprocess.call(['ffmpeg', '-i', Inputfilepath+'/'+cam_names[jj]+'.mp4', '-vf', "lenscorrection=cx=0.5:cy=0.5:k1=-.115:k2=-0.022", Outputfilepath+'/'+cam_names[jj]+'.mp4'])


def trimVideos(Inputfilepath,OutputFilepath):
    '''Function input is the filepath for undistorted videos and a filepath for the desired output path
    The function finds the frame at the beginning and end of the video where a light flash occurs 
    The video is then trimmed based on those frame numbers
    Outputs the trimmed video to specified filepath
    '''    
    for ii in range(len(cam_names)):
        vidcap = cv2.VideoCapture(Inputfilepath+'/'+cam_names[ii]+'.mp4')#Open video
        vidWidth  = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH) #Get video height
        vidHeight = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) #Get video width
        video_resolution = (int(vidWidth),int(vidHeight)) #Create variable for video resolution
        vidLength = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        vidfps = vidcap.get(cv2.CAP_PROP_FPS)
        success,image = vidcap.read() #read a frame
        maxfirstGray = 0 #Intialize the variable for the threshold of the max brightness of beginning of video
        maxsecondGray = 0 #Intialize the variable for the threshold of the max brightness of end of video
        print(cam_names[ii],vidLength)
        '''
        for jj in range(int(vidLength)):#For each frame in the video
            
            success,image = vidcap.read() #read a frame
            if success: #If frame is correctly read
                if jj < int(vidLength/3): #If the frame is in the first third of video
                    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #Convert image to greyscale
                    if np.average(gray) > maxfirstGray:#If the average brightness is greater than the threshold
                        maxfirstGray = np.average(gray)#That average brightness becomes the threshold
                        firstFlashFrame = jj#Get the frame number of the brightest frame
                if jj > int((2*vidLength)/3):
                    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #Convert image to greyscale
                    if np.average(gray) > maxsecondGray:#If the average brightness is greater than the threshold
                        maxsecondGray = np.average(gray)#That average brightness becomes the threshold
                        secondFlashFrame = jj #Get the frame number of the brightest frame
            else:#If the frame is not correctly read
                continue#Continue
        input1 = ffmpeg.input(Inputfilepath+'/'+cam_names[ii]+'.mp4')#input for ffmpeg

        node1_1 = input1.trim(start_frame=firstFlashFrame,end_frame=secondFlashFrame).setpts('PTS-STARTPTS')#Trim video based on the frame numbers
        node1_1.output(OutputFilepath+'/'+cam_names[ii]+'.mp4').run()#Save to output folder
        '''
def runDeepLabCut(Inputfilepath,OutputFilepath):
    '''Function inputs are filepath to videos to be tracked by DLC and the folder to save the output to
    Videos are copied to output folder, than processed in DLC based on the dlc config path 
    DLC output is saved in outputfilepath and the output is also converted to npy and saved as well
    '''
    
    #####################Copy Videos to DLC Folder############
    for dir in [Inputfilepath]:#Iterates through input folder
        for video in os.listdir(dir):#Iterates through each video in folder
            #ffmpeg call to copy videos to dlc folder
            subprocess.call(['ffmpeg', '-i', Inputfilepath+'/'+video,  OutputFilepath+'/'+video])


    #################### DeepLabCut ############################
    for dir in [OutputFilepath]:# Loop through dlc folder
        for video in os.listdir(dir):
            #Analyze the videos through deeplabcut
            deeplabcut.analyze_videos(baseProjectPath+'/'+DLCconfigPath, [OutputFilepath +'/'+ video], save_as_csv=True)
            deeplabcut.plot_trajectories(baseProjectPath+'/'+DLCconfigPath,[OutputFilepath +'/'+ video])

    for dir in [OutputFilepath]:#Loop through dlc folder
        for video in dir:# for each video in folder
            #Create a DLC video   
            deeplabcut.create_labeled_video(baseProjectPath+'/'+DLCconfigPath, glob.glob(os.path.join(OutputFilepath ,'*mp4')))

    #If there is not a folder for dlc npy output, create one
    if not os.path.exists(OutputFilepath + 'DLCnpy'):
        os.mkdir(OutputFilepath+ 'DLCnpy')
    
    #Load all dlc csv output files  
    csvfiles = glob.glob(OutputFilepath+'/*csv')
    #For loop gets csv data from all cameras
    j=0
    for data in csvfiles:     
        datapoints = pd.read_csv(data) # read in the csv data 
        print(datapoints)            

        parsedDlcData = datapoints.iloc[3:,7:10].values#the last element in the array is the P value
        #print(parsedDlcData)
    
        print(parsedDlcData)
        np.save(OutputFilepath+'/DLCnpy/dlc_'+cam_names[j]+'.npy',parsedDlcData)#Save data
        j+=1
           

def runOpenPose(Inputfilepath,VideoOutputPath,DataOutputFilepath):
    '''Function inputs are the undistorted video filepath, the filepath to save the video output, and the filepath to save the data output
    The function takes the undistorted video and processes the videos in openpose
    The output is openpose overlayed videos and raw openpose data
    '''
    if portraitMode:
        rotation = 90
    else:
        rotation = 0 
    ###################### OpenPose ######################################
    #os.chdir("C:/Users/MatthisLab/openpose") # change the directory to openpose
    j = 0
    for jj in range(len(cam_names)):
        subprocess.call(['bin/OpenPoseDemo.exe', '--video', Inputfilepath+'/'+cam_names[jj]+'.mp4', '--frame_rotate='+str(rotation) ,'--hand','--face','--write_video', VideoOutputPath+'/OpenPose'+cam_names[jj]+'.avi',  '--write_json', DataOutputFilepath+'/'+cam_names[jj]])
        j =+1


def Parse_Openpose(Inputfilepath,OutputFilepath):
    '''Function inputs is the filepath to rawopenpose data and the filepath to where to save the parsed openpose data
    Function takes the raw openpose data and organizes in a h5 file, that h5 file is then opened and the data is saved as an npy file
    Outputs one h5 file and an npy file for each camera and returns the amount of points in the frame
    '''
    #Establish how many points are being used from the user input
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
            framesOfPeople = []
            for frame in range(len(allFrames)):
                peopleInFrame = 0
                allPeople = f.get('Cameras/'+str(cam_names[camera])+'/Frame'+str(frame))
                if len(allPeople) == 0:
                    noPersonInFrame.append(frame) 
                    empty = (points_inFrame,3)
                    a = np.zeros(empty)
                    ret.append(a)
                    continue
                
                else:
                    c = 0
                    res = 0
                    for person in range(len(allPeople)):
                        zeroPoint =[]
                        peopleInFrame+=1
                        #========================Load body point data
                        if include_OpenPoseSkeleton:#If you include skeleton
                            skeleton  = f.get('Cameras/'+str(cam_names[camera])+'/Frame'+str(frame)+'/Person'+str(person)+'/Skeleton')  
                            skeleton = skeleton[()]
                        if include_OpenPoseHands: #If you include hands
                            hand_left = f.get('Cameras/'+str(cam_names[camera])+'/Frame'+str(frame)+'/Person'+str(person)+'/LeftHand')
                            hand_left = hand_left[()]
                            hand_right = f.get('Cameras/'+str(cam_names[camera])+'/Frame'+str(frame)+'/Person'+str(person)+'/RightHand')
                            hand_right = hand_right[()]
                        if include_OpenPoseFace:#If you include face
                            face = f.get('Cameras/'+str(cam_names[camera])+'/Frame'+str(frame)+'/Person'+str(person)+'/Face')
                            face = face[()]
                    
                        #============================Find correct skeleton
                        #distance = sum(sum(abs(target_skeleton-skeleton))) #Calculate the distance of the person in this frame compared to the target person from last frame
                        pval = skeleton[:,2]
                        avgPval = sum(pval)/len(pval)
                        #for jj in range(len(skeleton)):
                        #    if skeleton[jj,0] > .001:
                        #        zeroPoint.append(jj)


                        #if distance < c: #If the distance is less than the threshold than this person is the target skeleton
                        if avgPval > c:
                            c = avgPval
                            #c = distance #the distance becomes threshold
                            #c = len(zeroPoint)
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
                                    fullPoints =  res
                        
                framesOfPeople.append(peopleInFrame)
                ret.append(fullPoints)
            ret = np.array(ret)
            print(ret.shape)
            np.save(OutputFilepath+'/OP_'+cam_names[k]+'.npy',ret)
            #np.savetxt(OutputFilepath+'/OP_'+cam_names[k]+'.txt',ret[:,8,0])
            k  = k+1

    return noPersonInFrame


def checkerBoardUndistort(Inputfilepath,OutputFilepath):
    '''Function input is raw distorted checkerboard videos filepath and the filepath to save the videos to  
    Uses ffmpeg and camera intrinsics to undistort the video
    Outputs the undistorted video to the specified file path
    '''
    checkerDatadir = [Inputfilepath]   
    for dir in checkerDatadir:
        for video in os.listdir(dir):
            subprocess.call(['ffmpeg', '-i', Inputfilepath+'/'+video, '-vf', "lenscorrection=cx=0.5:cy=0.5:k1=-.115:k2=-0.022", OutputFilepath+'/'+video])


