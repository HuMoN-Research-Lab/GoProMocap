import numpy as np
import cv2
import glob
import os 
baseFilePath = 'C:/Users/chris/JugglingProject/ChessboardCalibration/Undistortion'

if not os.path.exists(baseFilePath + '/Calibration'):
        os.mkdir(baseFilePath + '/Calibration')
calibratefilepath = baseFilePath + '/Calibration'



vidcap = cv2.VideoCapture(baseFilePath +'/CamC.mp4')
framelength = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
print(framelength)
vidWidth  = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)  
vidHeight = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
video_resolution = (int(vidWidth),int(vidHeight))
video_resolution = (1920,1280)
success,image = vidcap.read()
count = 0
#success = True
'''
for ii in range(int(framelength)):
    success,image = vidcap.read()
    if success:
        height , width , layers =  image.shape
        resize = cv2.resize(image, video_resolution) 
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #single_video.append(image)   
        if not os.path.exists(baseFilePath + '/Calibration/CamC_Calibration'):
            os.mkdir(baseFilePath + '/Calibration/CamC_Calibration')                       
        cv2.imwrite(baseFilePath+'/Calibration/CamC_Calibration/frame%d.jpg' %count , image)     # save frame as JPEG file
        #print(resize.shape)
    


        
        count += 1
        
    else:
        continue
print(count)
'''





# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(calibratefilepath+'/CamC_Calibration/*.jpg')

for fname in images:
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
        #cv2.imshow('img',img)
        #cv2.waitKey(500)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

img = cv2.imread(calibratefilepath+'/CamC_Calibration/frame7.jpg')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite(calibratefilepath+'/Undistorted.png',dst)

np.save(baseFilePath + '_ret.npy',ret)
np.save(baseFilePath + '_mtx.npy',mtx)
np.save(baseFilePath + '_dist.npy',dist)
np.save(baseFilePath + '_rvec.npy',rvecs)
np.save(baseFilePath + '_tvecs.npy',tvecs)
cv2.imshow('frame', dst)
cv2.waitKey(50000)

