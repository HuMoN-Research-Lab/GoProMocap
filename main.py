import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
from ops import toCsv,vec2skewMat,inverseH,R_t2H,get_RT_mtx,video_loader,get_TransMat,triangulate,triangulateTest
from config import base_Cam_Index,num_of_cameras,video_resolution,Len_of_frame,SAVE_FOLDER,start_frame,Source_video_List,Pixel_coord_FIlE_List,SourceVideoFolder,Source_video_List,include_ball,Pixel_coord_FIlE_List_include_ball,points_inFrame, baseProjectPath,subject,sessionID
from visualize import Vis
from scipy.optimize import least_squares
import time
from scipy.sparse import lil_matrix
from VideoProcess import runOPandDLC
from Parse_dlc import Parse_dlc
from Parse_OpenPose import Parse_OpenPose


#=====================Run OpenPose and DeepLabCut and parse through the output
runOPandDLC()
Parse_dlc()
Parse_OpenPose()

#=====================Get files for dlc and openpose data 
rootOPFolder = baseProjectPath+'/'+subject+'/'+sessionID+'/OpenPose/OpenPoseOutPut/'
rootDLCFolder = baseProjectPath+'/'+subject+'/'+sessionID+'/DeepLabCut/DLCnpy/'

Pixel_coord_FIlE_List = [['OP_CamA.npy','CamA'],
                         ['OP_CamB.npy','CamB']]

Pixel_coord_FIlE_List_include_ball = [['OP_CamA.npy','dlc_CamA.npy','CamA'],
                                        ['OP_CamB.npy','dlc_CamB.npy','CamB']]


base_cam = {'A':0,'B':1,'C':2} #make a dictionary that convert camera index from string to integer

#==================load image from videos 
for path in Source_video_List:
    video_loader(path[0],path[1])


#==================load pixel data to a dictionary
pixelCoord = {} #define a dictiionary which will hold pixel data for each camera
if include_ball:
    for path in Pixel_coord_FIlE_List_include_ball:
        skeleton = np.load(path[0])[:1450] #this part is for testing purpose, 1450 is the frame number for testing video
        ball = np.load(path[1],allow_pickle = True)
        ball = ball.astype(float)
        ball = ball.reshape((1450,1,3))
        pixelC = np.concatenate((skeleton,ball),axis=-2)
        pixelCoord[path[-1]] = pixelC
        pixelCoord[path[-1]] = pixelCoord[path[-1]][start_frame:start_frame+Len_of_frame,:,:]

else:
    for path in Pixel_coord_FIlE_List:
        pixelCoord[path[1]] = np.load(path[0],allow_pickle = True) 
        pixelCoord[path[1]] = pixelCoord[path[1]][start_frame:start_frame+Len_of_frame,:,:] 
    
    #at the end, we get a dictionart which each key is the camera name, for example pixelCoord['CamA'] = (frame,#of points,xyz) np array





#================== calibrate the cameras

#load camera matrix, Translaton vector, rotation vector
_,K_CamB,B_dist,B_rvecs,B_tvecs = get_RT_mtx('CalibrationData/CamB_Calibration/*.jpg','B',video_resolution)
#mtx,tvecs and rvecs has same length as how many images we used to calibrate the camera. We only need the first one. They all identical.
tvec_CamB,rvec_CamB = B_tvecs[0],B_rvecs[0]
#convert rotation vector to its matrix form
RoMat_B, _ = cv2.Rodrigues(rvec_CamB) #convert 
#combine rotation matrix and translation vector as a 4by4 homogenies transformation matrix
H_CamB = R_t2H(RoMat_B,tvec_CamB)

_,K_CamA,A_dist,A_rvecs,A_tvecs = get_RT_mtx('CalibrationData/CamA_Calibration/*.jpg','A',video_resolution)
tvec_CamA,rvec_CamA = A_tvecs[0],A_rvecs[0]
RoMat_A, _ = cv2.Rodrigues(rvec_CamA)
H_CamA = R_t2H(RoMat_A,tvec_CamA)


if num_of_cameras == 3:
    _,K_CamC,C_dist,C_rvecs,C_tvecs = get_RT_mtx('CalibrationData/CamC_Calibration/*.jpg','C',video_resolution)
    tvec_CamC,rvec_CamC = C_tvecs[0],C_rvecs[0]
    RoMat_C, _ = cv2.Rodrigues(rvec_CamC)
    H_CamC = R_t2H(RoMat_C,tvec_CamC)


#=====================prepare proj matrix and pixel coords

#the transformation matrx of the principle camera will be a 4by4 diagonal matrix, if cam A is the principle camera, MA is a 4by4 diagonal
#matrix and MB,MC describe the position of CamB and CamC correspond to CamA
if base_Cam_Index == 'A' and num_of_cameras == 3:
 
    MA,MB,MC = get_TransMat(H_CamA,H_CamB,H_CamC)
    # projection matrix = camera matrix * transformation matrix
    PA,PB,PC = np.dot(K_CamA,MA),np.dot(K_CamB,MB),np.dot(K_CamC,MC)
    # stack points and projection matrix for points triangulation
    Proj_points = np.stack((pixelCoord['CamA'],pixelCoord['CamB'],pixelCoord['CamC']),axis = 2)
    Proj_Mat = np.stack((PA,PB,PC),axis=0)
    #BA_points2D = np.vstack((pixelCoord['CamA'],pixelCoord['CamB'],pixelCoord['CamC']))

elif base_Cam_Index == 'B' and num_of_cameras == 3:
    MB,MA,MC = get_TransMat(H_CamB,H_CamA,H_CamC)
    PB,PA,PC = np.dot(K_CamB,MB),np.dot(K_CamA,MA),np.dot(K_CamC,MC)
    Proj_points = np.stack((pixelCoord['CamB'],pixelCoord['CamA'],pixelCoord['CamC']),axis = 2)
    Proj_Mat = np.stack((PB,PA,PC),axis=0)
    #BA_points2D = np.vstack((pixelCoord['CamB'],pixelCoord['CamA'],pixelCoord['CamC']))

elif base_Cam_Index == 'C' and num_of_cameras == 3:
    MC,MA,MB = get_TransMat(H_CamC,H_CamA,H_CamB)
    PC,PA,PB = np.dot(K_CamC,MC),np.dot(K_CamA,MA),np.dot(K_CamB,MB)
    Proj_points = np.stack((pixelCoord['CamC'],pixelCoord['CamA'],pixelCoord['CamB']),axis = 2)
    Proj_Mat = np.stack((PC,PA,PB),axis=0)
    #BA_points2D = np.vstack((pixelCoord['CamC'],pixelCoord['CamA'],pixelCoord['CamB']))

elif base_Cam_Index == 'A' and num_of_cameras == 2:
    MA,MB = get_TransMat(H_CamA,H_CamB)
    PA,PB = np.dot(K_CamA,MA),np.dot(K_CamB,MB)
    Proj_points = np.stack((pixelCoord['CamA'],pixelCoord['CamB']),axis = 2)
    Proj_Mat = np.stack((PA,PB),axis=0)
    #BA_points2D = np.vstack((pixelCoord['CamA'],pixelCoord['CamB']))

elif base_Cam_Index == 'B' and num_of_cameras == 2:
    MB,MA = get_TransMat(H_CamB,H_CamA)
    PB,PA = np.dot(K_CamB,MB),np.dot(K_CamA,MA)
    Proj_points = np.stack((pixelCoord['CamB'],pixelCoord['CamA']),axis = 2)
    Proj_Mat = np.stack((PB,PA),axis=0)
    #BA_points2D = np.vstack((pixelCoord['CamB'],pixelCoord['CamA']))

#triangulate points and return a list which tells each point is visiable to which cameras (we assume priciple camera can always see every
#points, an additional camera with best view will be used to reconstruct 3D
coords,VIS_cam_List = triangulateTest(Proj_points,Proj_Mat,base_cam[base_Cam_Index]).solveA()
coords = coords[:,:,:-1]


#===========sparse bundle adjustment
if include_ball:
    ball_points = coords[:,-1,:].reshape((-1,1,3))
    skeleton_points = coords[:,:-1,:]

input_points = skeleton_points.reshape((-1,))
if num_of_cameras == 3:
    #just a different format of stacking all pixel coordinates, sparse bundle adjustment need flattenn all data
    BA_points2D = np.stack((pixelCoord['CamA'][:,:,:-1],pixelCoord['CamB'][:,:,:-1],pixelCoord['CamC'][:,:,:-1]),axis = 0)
    input_param = np.hstack((Proj_Mat[0].ravel(),Proj_Mat[1].ravel(),Proj_Mat[2].ravel()))
elif num_of_cameras == 2:
    BA_points2D = np.stack((pixelCoord['CamA'][:,:-1,:-1],pixelCoord['CamB'][:,:-1,:-1]),axis = 0)
    input_param = np.hstack((Proj_Mat[0].ravel(),Proj_Mat[1].ravel()))

#final input of sparse bundle ajustment
ba_input = np.hstack((input_points,input_param))



def SBA(Len_of_frame,ProjMats,points2d,ba_input,VIS_cam_List):
    """
    Len_of_points:how many points to be recontrusct
    points2D: all pixel locations
    ba_input:1D vector include flattened 3d points and projection matrix
    """

    def fun(ba_input):
        """
        function that reproject 3d points back to 2D points, calculate the error (reproject 2d - real 2d)
        """
        #reshape points back to (len,25,3)
        p = ba_input[:Len_of_frame*3*points_inFrame].reshape((-1,points_inFrame,3)) 
        #extract projection matrixs from input
        param = ba_input[Len_of_frame*points_inFrame*3:]

        temp = np.ones((p.shape[0],p.shape[1],1))
        #attach 1 to the end to the 3d points so it becomes homogenies coordinate
        x = np.concatenate((p,temp),axis=2)
        #define an array that will hold true pixel data
        true_pixel_coord = np.zeros((2*Len_of_frame,points_inFrame,2))

        if num_of_cameras == 2:
            l = len(param)//2
            #reshape projection matrix back to (3,4) 
            ProjMats = (param[:l].reshape((3,4)),param[l:].reshape((3,4)))
            #first half of true pixel data must come from the principle camera
            true_pixel_coord[:Len_of_frame] = points2d[base_cam[base_Cam_Index]]
            #reprojected 2d = 3D * projection matrix
            reproj1 = x.dot(ProjMats[base_cam[base_Cam_Index]].T)
            #define an array that will hold second half of reprojected 3d. they are projected to one of the camera excepts principle cam
            reproj2 = np.zeros((Len_of_frame,points_inFrame,3))
            for i in range(Len_of_frame):
                #reprojected 2d = 3D * projection matrix, we picked each of projection based on the visiable cam list we got from triangulation
                reproj2[i] = x[i].dot(ProjMats[VIS_cam_List[i]].T)
                true_pixel_coord[Len_of_frame+i] = points2d[VIS_cam_List[i]][i]
            
            
        elif num_of_cameras == 3:
            l = len(param)//3
            ProjMats = (param[:l].reshape((3,4)),param[l:2*l].reshape((3,4)),param[2*l:].reshape((3,4)))
            true_pixel_coord[:Len_of_frame] = points2d[base_cam[base_Cam_Index]]
            reproj1 = x.dot(ProjMats[base_cam[base_Cam_Index]].T)
            reproj2 = np.zeros((Len_of_frame,points_inFrame,3))
            for i in range(Len_of_frame):
                reproj2[i] = x[i].dot(ProjMats[VIS_cam_List[i]].T)
                true_pixel_coord[Len_of_frame+i] = points2d[VIS_cam_List[i]][i]
            
        #stack reprojected 2d   
        reproj_points = np.vstack((reproj1,reproj2))
        #devide x,y by z 
        reproj_points = reproj_points[:,:,:2] / reproj_points[:,:,2,np.newaxis]
        #calculate the error term
        res = (reproj_points-true_pixel_coord)

        return res.ravel()


    def bundle_adjustment_sparsity(n_point3D):
        """
        n_observation:total length of pixel coordinates
        define a sparsity matrix. 1 if the jaccobian matrix has value, else 0

        """
        m = n_point3D * 2 * 2 #row
        n = n_point3D * 3 + 12*num_of_cameras #col 
        A = lil_matrix((m, n), dtype=int)
        
        #fill out projection matrix
        if num_of_cameras == 2:
            if base_Cam_Index == 'A':
                A[:m//2,-24:-12] = 1
                A[m//2:,-12:] = 1
            else:
                A[m//2:,-24:-12] = 1
                A[:m//2,-12:] = 1
        
        elif num_of_cameras == 3:
            if base_Cam_Index == 'A':
                A[:m//2,-36:-24] = 1
            elif base_Cam_Index == 'B':
                A[:m//2,-24:-12] = 1
            elif base_Cam_Index == 'C':
                A[:m//2,-12:] = 1
            

            for i in range(n_point3D):
                s1,s2 = (VIS_cam_List[i]-3)*12,(VIS_cam_List[i]-3)*12+12
                if s2 == 0:
                    A[2*i,s1:] = 1
                else:
                    A[2*i+1,s1:s2] = 1 
        
        #===============fill out points

        for i in range(n_point3D):
            for s in range(3):
                A[2*i,i*3+s] =1 
                A[2*i+1,i*3+s] =1
    

        A[m//2:,:-12*num_of_cameras] = A[:m//2,:-12*num_of_cameras]

        
        return A
    
    
    #residual before bundle adjustment
    residual = fun(ba_input)
    #generate sparisty mat
    A = bundle_adjustment_sparsity(Len_of_frame*points_inFrame)
    plt.plot(residual)
    plt.show()

    x0 = ba_input
    
    t0 = time.time()
    #optimizing
    res = least_squares(fun,x0,jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf')
    #res = least_squares(fun,x0, verbose=2, x_scale='jac', ftol=1e-4, method='trf')
    t1 = time.time()
    print("Optimization took {0:.0f} seconds".format(t1 - t0))
    #plot residual after optimization
    plt.plot(res.fun)
    plt.show()
    
    #extract optimized 3d points and projection matrix
    param = res.x

    #extract 3d points and reshape
    optimized_3D = param[:Len_of_frame*3*points_inFrame]
    coords = optimized_3D.reshape((-1,points_inFrame,3))

    return coords

print("optimization started")

#C is the optimizaed 3d pooints
C = SBA(Len_of_frame,Proj_Mat,BA_points2D,ba_input,VIS_cam_List)
#print('coords shape',optimized_coords.shape)
np.save(SAVE_FOLDER+'output_3d.npy',C)
print('save sussesful')

if include_ball:
    C = np.concatenate((C,ball_points),axis=-2)

print(C.shape)

if num_of_cameras == 3:
    Vis(SourceVideoFolder+'/'+Source_video_List[0][0],SourceVideoFolder+'/'+Source_video_List[1][0],SourceVideoFolder+'/'+Source_video_List[2][0],C).display()

elif num_of_cameras == 2:
    Vis(SourceVideoFolder+'/'+Source_video_List[0][0],SourceVideoFolder+'/'+Source_video_List[1][0],None,C).display()



