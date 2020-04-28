import cv2  
   
   
   
    # termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(path)
count = 0
for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img,video_resolution)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners   
    ret, corners = cv2.findChessboardCorners(gray, (6,9),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        img = cv2.drawChessboardCorners(img, (6,9), corners2,ret)
        cv2.imshow('img',img)
        #cv2.waitKey(500)
    
    count += 1
    print(count)
        
cv2.destroyAllWindows()
