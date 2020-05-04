import cv2 
import numpy as np
from config import cam_names
import os 

videofile = 'D:/Calibration/FiducialMarkerTest/2.7kpWide'
# Load the predefined dictionary
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)



k = 0
'''
for dir in [videofile]:
        
        for video in os.listdir(dir):
            
            vidcap = cv2.VideoCapture(videofile+'/'+video)
            frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
           
            vidlength = range(int(frame_count)) 
            
            for ii in range(20):

                success,image = vidcap.read()
                if success:
                    height , width , layers =  image.shape 
                    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    #single_video.append(image)   
                    if not os.path.exists(videofile + '/'+cam_names[k]+'_CalibrationImages'):
                        os.mkdir(videofile + '/'+cam_names[k]+'_CalibrationImages')                       
                    cv2.imwrite(videofile+'/'+cam_names[k]+'_CalibrationImages/frame%d.jpg' %ii , image)     # save frame as JPEG file    
                else:
                    continue

        k+=1        
'''
framepath = videofile+'/CamA_CalibrationImages/frame0.jpg'
frame = cv2.imread(framepath)

# Initialize the detector parameters using default values
parameters =  cv2.aruco.DetectorParameters_create()
# Detect the markers in the image
markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)

markerImage = cv2.aruco.drawDetectedMarkers(frame,markerCorners)
cv2.imshow('frame',markerImage)
cv2.waitKey(50000)