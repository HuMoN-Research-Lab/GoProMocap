import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
from ops import toCsv,vec2skewMat,inverseH,R_t2H,get_RT_mtx,video_loader,get_TransMat,triangulate,triangulateFlex,triangulateTest
from config import cam_names, base_Cam_Index,num_of_cameras,Len_of_frame,start_frame,include_DLC, useCheckerboardVid,include_OpenPoseFace,include_OpenPoseHands,include_OpenPoseSkeleton, calibrateCameras
#from visulize_with_out_head import Vis
from visualize import Vis
from scipy.optimize import least_squares
import time
from scipy.sparse import lil_matrix
from create_project import openposeOutputFilepath,interfilepath, checkerVideoFolder, rawVideoFolder, rawData, baseFilePath, trimFilepath, create_project,processedFilePath,combinedFilepath,undistortedFilepath,DLCfilepath,openposeRawFilepath,videoOutputFilepath,interfilepath,calibrationFilePath,cameraParamsFilePath
from EditVideoRunOpandDLC import getCameraParams, concatVideos, undistortVideos,trimVideos, runDeepLabCut,runOpenPose, Parse_Openpose, checkerBoardUndistort



base_cam = {'A':0,'B':1,'C':2,'D':3}
#=========================Create Folders for project
create_project()

if calibrateCameras:
    getCameraParams(calibrationFilePath)
#===========================Concat,Undistort and Trim Videos 
concatVideos(rawVideoFolder,combinedFilepath)
undistortVideos(combinedFilepath,undistortedFilepath)
if useCheckerboardVid == True:
    if not os.path.exists(interfilepath + '/CheckerboardUndistorted'):
        os.mkdir(interfilepath + '/CheckerboardUndistorted')
    checkerUndistortFilepath = interfilepath + '/CheckerboardUndistorted'
    checkerBoardUndistort(checkerVideoFolder,checkerUndistortFilepath)
trimVideos(undistortedFilepath,trimFilepath)

#==========================Run deeplabcut and parse through output
runDeepLabCut(trimFilepath,DLCfilepath)
#==========================Run OpenPose and parse through output
if include_OpenPoseFace == True or include_OpenPoseHands ==True or include_OpenPoseSkeleton == True:
    runOpenPose(undistortedFilepath,videoOutputFilepath,openposeRawFilepath)
    points_inFrame = Parse_Openpose(openposeRawFilepath,openposeOutputFilepath)
#========================Get source video
if useCheckerboardVid == True:
    SourceVideoFolder = baseFilePath + '/Intermediate/CheckerboardUndistorted'
else: 
    SourceVideoFolder = undistortedFilepath

fullVideoFolder = undistortedFilepath #Always need to use this
#======================== Set up names for videos
cam1 = cam_names[0]
cam2 = cam_names[1]

if num_of_cameras ==2:
    Source_video_List = [[cam1+'.MP4',cam1],[cam2+'.MP4',cam2]]
if num_of_cameras ==3: 
    cam3 = cam_names[2]
    Source_video_List= [[cam1+'.MP4',cam1],[cam2+'.MP4',cam2],[cam3+'.MP4',cam3]]
if num_of_cameras ==4:
    cam3 = cam_names[2]
    cam4 = cam_names[3]
    Source_video_List= [[cam1+'.MP4',cam1],[cam2+'.MP4',cam2],[cam3+'.MP4',cam3],[cam4+'.MP4',cam4]]

#=====================Get files for dlc and openpose data 
rootOPFolder = openposeOutputFilepath
rootDLCFolder = DLCfilepath +'/DLCnpy'

if num_of_cameras ==2:
    Pixel_coord_FIlE_List = [[rootOPFolder+'OP_'+cam1+'.npy',cam1],
                             [rootOPFolder+'OP_'+cam2+'.npy',cam2]]

    Pixel_coord_FIlE_List_include_DLC = [[rootOPFolder+'OP_'+cam1+'.npy',rootDLCFolder+'dlc_'+cam1+'.npy',cam1],
                                          [rootOPFolder+'OP_'+cam2+'.npy',rootDLCFolder+'dlc_'+cam2+'.npy',cam2]]
if num_of_cameras ==3:
    Pixel_coord_FIlE_List = [[rootOPFolder+'OP_'+cam1+'.npy',cam1],
                             [rootOPFolder+'OP_'+cam2+'.npy',cam2],
                             [rootOPFolder+'OP_'+cam3+'.npy',cam3]]
                                                         

    Pixel_coord_FIlE_List_include_DLC = [[rootOPFolder+'OP_'+cam1+'.npy',rootDLCFolder+'dlc_'+cam1+'.npy',cam1],
                                          [rootOPFolder+'OP_'+cam2+'.npy',rootDLCFolder+'dlc_'+cam1+'.npy',cam2],
                                          [rootOPFolder+'OP_'+cam3+'.npy',rootDLCFolder+'dlc_'+cam3+'.npy',cam3]]
if num_of_cameras ==4:
    Pixel_coord_FIlE_List = [[rootOPFolder+'OP_'+cam1+'.npy',cam1],
                             [rootOPFolder+'OP_'+cam2+'.npy',cam2],
                             [rootOPFolder+'OP_'+cam3+'.npy',cam3],
                             [rootOPFolder+'OP_'+cam4+'.npy',cam4]]
                                                         

    Pixel_coord_FIlE_List_include_DLC = [[rootOPFolder+'OP_'+cam1+'.npy',rootDLCFolder+'dlc_'+cam1+'.npy',cam1],
                                          [rootOPFolder+'OP_'+cam2+'.npy',rootDLCFolder+'dlc_'+cam2+'.npy',cam2],
                                          [rootOPFolder+'OP_'+cam3+'.npy',rootDLCFolder+'dlc_'+cam3+'.npy',cam3],
                                          [rootOPFolder+'OP_'+cam4+'.npy',rootDLCFolder+'dlc_'+cam4+'.npy',cam4]]

if Len_of_frame == -1:
    frameLengthAllCam = [] #create variable to stoe frame length
    for dir in [fullVideoFolder]:
        for video in os.listdir(dir):
            vidcap = cv2.VideoCapture(os.path.join(dir,video))
            frameLengthOneCam = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            frameLengthAllCam.append(frameLengthOneCam)
    
    minVideo = frameLengthAllCam.index(min(frameLengthAllCam))  
    shortestVideo = frameLengthAllCam[minVideo]
    Len_of_frame = shortestVideo

if num_of_cameras == 2:
    base_cam = {cam_names[0]:0,cam_names[1]:1}
if num_of_cameras == 3:
    base_cam = {cam_names[0]:0,cam_names[1]:1, cam_names[2]:2}
if num_of_cameras == 4:
    base_cam = {cam_names[0]:0,cam_names[1]:1,cam_names[2]:2,cam_names[3]:3}
#==================load image from videos 
for path in Source_video_List:
    video_resolution = video_loader(path[0],path[1])


#==================load pixel data to a dictionary
pixelCoord = {}
if include_DLC:
    for path in Pixel_coord_FIlE_List_include_DLC:
        skeleton = np.load(path[0],allow_pickle = True)[start_frame:start_frame+Len_of_frame,:,:]
        ball = np.load(path[1],allow_pickle = True)[start_frame:start_frame+Len_of_frame,:]
        ball = ball.astype(float)
        ball = ball.reshape((-1,3,3))
        pixelC = np.concatenate((skeleton,ball),axis=-2)
        pixelCoord[path[-1]] = pixelC
        #pixelCoord[path[-1]] = pixelCoord[path[-1]][start_frame:start_frame+Len_of_frame,:,:]

else:
    for path in Pixel_coord_FIlE_List:
        pixelCoord[path[1]] = np.load(path[0],allow_pickle = True)
        pixelCoord[path[1]] = pixelCoord[path[1]][start_frame:start_frame+Len_of_frame,:,:]




#================== calibrate the cameras

__,K_CamB,B_dist,B_rvecs,B_tvecs = get_RT_mtx(baseFilePath+'/Calibration/'+cam_names[1]+'_Calibration/*jpg',cam_names[1],video_resolution)
tvec_CamB,rvec_CamB = B_tvecs[0],B_rvecs[0]
RoMat_B, _ = cv2.Rodrigues(rvec_CamB) #convert 
H_CamB = R_t2H(RoMat_B,tvec_CamB)

_,K_CamA,A_dist,A_rvecs,A_tvecs = get_RT_mtx(baseFilePath+'/Calibration/'+cam_names[0]+'_Calibration/*jpg',cam_names[0],video_resolution)
tvec_CamA,rvec_CamA = A_tvecs[0],A_rvecs[0]
RoMat_A, _ = cv2.Rodrigues(rvec_CamA)
H_CamA = R_t2H(RoMat_A,tvec_CamA)
if num_of_cameras > 2:
    _,K_CamC,C_dist,C_rvecs,C_tvecs = get_RT_mtx(baseFilePath+'/Calibration/'+cam_names[2]+'_Calibration/*jpg',cam_names[2],video_resolution)
    tvec_CamC,rvec_CamC = C_tvecs[0],C_rvecs[0]
    RoMat_C, _ = cv2.Rodrigues(rvec_CamC)
    H_CamC = R_t2H(RoMat_C,tvec_CamC)

if num_of_cameras > 3:
    _,K_CamD,D_dist,D_rvecs,D_tvecs = get_RT_mtx(baseFilePath+'/Calibration/'+cam_names[3]+'_Calibration/*jpg',cam_names[3],video_resolution)
    tvec_CamD,rvec_CamD = D_tvecs[0],D_rvecs[0]
    RoMat_D, _ = cv2.Rodrigues(rvec_CamD)
    H_CamD = R_t2H(RoMat_D,tvec_CamD)


#=====================prepare proj matrix and pixel coords

#def get_ProjPoint_ProjMat(base_Cam_Index,num_of_cameras)------to do

Proj_points = None
Proj_Mat = None

if num_of_cameras == 4:
    if base_Cam_Index == 'A':
        MA,MB,MC,MD = get_TransMat(H_CamA,H_CamB,H_CamC,H_CamD)
        PA,PB,PC,PD = np.dot(K_CamA,MA),np.dot(K_CamB,MB),np.dot(K_CamC,MC),np.dot(K_CamD,MD)
        Proj_points = np.stack((pixelCoord['CamA'],pixelCoord['CamB'],pixelCoord['CamC'],pixelCoord['CamD']),axis = 2)
        Proj_Mat = np.stack((PA,PB,PC,PD),axis=0)

    elif base_Cam_Index == 'B':
        MB,MA,MC,MD = get_TransMat(H_CamB,H_CamA,H_CamC,H_CamD)
        PB,PA,PC,PD = np.dot(K_CamB,MB),np.dot(K_CamA,MA),np.dot(K_CamC,MC),np.dot(K_CamD,MD)
        Proj_points = np.stack((pixelCoord['CamB'],pixelCoord['CamA'],pixelCoord['CamC'],pixelCoord['CamD']),axis = 2)
        Proj_Mat = np.stack((PB,PA,PC,PD),axis=0)

    elif base_Cam_Index == 'C':
        MC,MA,MB,MD = get_TransMat(H_CamC,H_CamA,H_CamB,H_CamD)
        PC,PA,PB,PD = np.dot(K_CamC,MC),np.dot(K_CamA,MA),np.dot(K_CamB,MB),np.dot(K_CamD,MD)
        Proj_points = np.stack((pixelCoord['CamC'],pixelCoord['CamA'],pixelCoord['CamB'],pixelCoord['CamD']),axis = 2)
        Proj_Mat = np.stack((PC,PA,PB,PD),axis=0)

    elif base_Cam_Index == 'D':
        MD,MA,MB,MC = get_TransMat(H_CamD,H_CamA,H_CamB,H_CamC)
        PD,PA,PB,PC = np.dot(K_CamD,MD),np.dot(K_CamA,MA),np.dot(K_CamB,MB),np.dot(K_CamC,MC)
        Proj_points = np.stack((pixelCoord['CamD'],pixelCoord['CamA'],pixelCoord['CamB'],pixelCoord['CamC']),axis = 2)
        Proj_Mat = np.stack((PD,PA,PB,PC),axis=0)
    
    BA_points2D = np.stack((pixelCoord['CamA'][:,:25,:-1],pixelCoord['CamB'][:,:25,:-1],pixelCoord['CamC'][:,:25,:-1],pixelCoord['CamD'][:,:25,:-1]),axis = 0)
    input_param = np.hstack((Proj_Mat[0].ravel(),Proj_Mat[1].ravel(),Proj_Mat[2].ravel(),Proj_Mat[3].ravel()))



elif num_of_cameras == 3:
    if base_Cam_Index == 'A':
        MA,MB,MC = get_TransMat(H_CamA,H_CamB,H_CamC)
        PA,PB,PC = np.dot(K_CamA,MA),np.dot(K_CamB,MB),np.dot(K_CamC,MC)
        Proj_points = np.stack((pixelCoord['CamA'],pixelCoord['CamB'],pixelCoord['CamC']),axis = 2)
        Proj_Mat = np.stack((PA,PB,PC),axis=0)
    
    elif base_Cam_Index == 'B':
        MB,MA,MC = get_TransMat(H_CamB,H_CamA,H_CamC)
        PB,PA,PC = np.dot(K_CamB,MB),np.dot(K_CamA,MA),np.dot(K_CamC,MC)
        Proj_points = np.stack((pixelCoord['CamB'],pixelCoord['CamA'],pixelCoord['CamC']),axis = 2)
        Proj_Mat = np.stack((PB,PA,PC),axis=0)
    
    elif base_Cam_Index == 'C':
        MC,MA,MB = get_TransMat(H_CamC,H_CamA,H_CamB)
        PC,PA,PB = np.dot(K_CamC,MC),np.dot(K_CamA,MA),np.dot(K_CamB,MB)
        Proj_points = np.stack((pixelCoord['CamC'],pixelCoord['CamA'],pixelCoord['CamB']),axis = 2)
        Proj_Mat = np.stack((PC,PA,PB),axis=0)
    
    BA_points2D = np.stack((pixelCoord['CamA'][:,:25,:-1],pixelCoord['CamB'][:,:25,:-1],pixelCoord['CamC'][:,:25,:-1]),axis = 0)
    input_param = np.hstack((Proj_Mat[0].ravel(),Proj_Mat[1].ravel(),Proj_Mat[2].ravel()))
    
elif num_of_cameras == 2:
    if base_Cam_Index == 'A':
        MA,MB = get_TransMat(H_CamA,H_CamB)
        PA,PB = np.dot(K_CamA,MA),np.dot(K_CamB,MB)
        Proj_points = np.stack((pixelCoord['CamA'],pixelCoord['CamB']),axis = 2)
        Proj_Mat = np.stack((PA,PB),axis=0)
    
    elif base_Cam_Index == 'B':
        MB,MA = get_TransMat(H_CamB,H_CamA)
        PB,PA = np.dot(K_CamB,MB),np.dot(K_CamA,MA)
        Proj_points = np.stack((pixelCoord['CamB'],pixelCoord['CamA']),axis = 2)
        Proj_Mat = np.stack((PB,PA),axis=0)
    
    BA_points2D = np.stack((pixelCoord['CamA'][:,:25,:-1],pixelCoord['CamB'][:,:25,:-1]),axis = 0)
    input_param = np.hstack((Proj_Mat[0].ravel(),Proj_Mat[1].ravel()))

print('Proj points shape:',Proj_points.shape)
#coords_temp,VIS_cam_List = triangulateTest(Proj_points,Proj_Mat,base_cam[base_Cam_Index]).solveA()
coords = triangulate(Proj_points,Proj_Mat).solveA()#tranigulate points
coords = coords[:,:,:-1]

print('coords shape:',coords.shape)


# print('valid point length',len(valid_point_list))
# print('valid',valid_point_list)
# print('vis cam',VIS_cam_List)
# valid_point, invalid_point = 0,0
# for num in valid_point_list:
#     if num == 0:
#         invalid_point += 1
#     else:
#         valid_point += 1

# print('valid:',valid_point,'invalid:',invalid_point)


#===========sparse bundle adjustment
# if include_DLC:
#     ball_points = coords[:,-1,:].reshape((-1,3,3))
#     skeleton_points = coords[:,:-3,:]


input_points = coords.reshape((-1,))


ba_input = np.hstack((input_points,input_param))


def SBA(Len_of_frame,ProjMats,points2d,ba_input,VIS_cam_List):
    """
    Len_of_points:how many points to be recontrusct
    points2D: all pixel locations
    ba_input:1D vector include flattened 3d points and projection matrix
    """

    def fun(ba_input):
        p = ba_input[:Len_of_frame*3*points_inFrame].reshape((-1,points_inFrame,3)) #reshape back to(len,25,3)
        param = ba_input[Len_of_frame*points_inFrame*3:]

        temp = np.ones((p.shape[0],p.shape[1],1))
        x = np.concatenate((p,temp),axis=2)
        true_pixel_coord = np.zeros((2*Len_of_frame,points_inFrame,2))

        if num_of_cameras == 2:
            l = len(param)//2
            ProjMats = (param[:l].reshape((3,4)),param[l:].reshape((3,4)))
            true_pixel_coord[:Len_of_frame] = points2d[base_cam[base_Cam_Index]]
            reproj1 = x.dot(ProjMats[base_cam[base_Cam_Index]].T)
            reproj2 = np.zeros((Len_of_frame,points_inFrame,3))
            
            for i in range(Len_of_frame):
                for j in range(points_inFrame):
                    k = i*points_inFrame + j
                    reproj2[i][j] = x[i][j].dot(ProjMats[VIS_cam_List[k]].T)
                    true_pixel_coord[Len_of_frame+i][j] = points2d[VIS_cam_List[k]][i][j]
            
            
        elif num_of_cameras == 3:
            l = len(param)//3
            ProjMats = (param[:l].reshape((3,4)),param[l:2*l].reshape((3,4)),param[2*l:].reshape((3,4)))
            true_pixel_coord[:Len_of_frame] = points2d[base_cam[base_Cam_Index]]
            reproj1 = x.dot(ProjMats[base_cam[base_Cam_Index]].T)
            reproj2 = np.zeros((Len_of_frame,points_inFrame,3))
            
            
            for i in range(Len_of_frame):
                for j in range(points_inFrame):
                    k = i*points_inFrame + j
                    reproj2[i][j] = x[i][j].dot(ProjMats[VIS_cam_List[k]].T)
                    true_pixel_coord[Len_of_frame+i][j] = points2d[VIS_cam_List[k]][i][j]
        

        elif num_of_cameras == 4:
            l = len(param)//4
            ProjMats = (param[:l].reshape((3,4)),param[l:2*l].reshape((3,4)),param[2*l:3*l].reshape((3,4)),param[3*l:].reshape((3,4)))
            true_pixel_coord[:Len_of_frame] = points2d[base_cam[base_Cam_Index]]
            reproj1 = x.dot(ProjMats[base_cam[base_Cam_Index]].T)
            reproj2 = np.zeros((Len_of_frame,points_inFrame,3))
            
            
            for i in range(Len_of_frame):
                for j in range(points_inFrame):
                    k = i*points_inFrame + j
                    reproj2[i][j] = x[i][j].dot(ProjMats[VIS_cam_List[k]].T)
                    true_pixel_coord[Len_of_frame+i][j] = points2d[VIS_cam_List[k]][i][j]

            
            
        reproj_points = np.vstack((reproj1,reproj2))
 
        reproj_points = reproj_points[:,:,:2] / reproj_points[:,:,2,np.newaxis]
        res = (reproj_points-true_pixel_coord)

        return res.ravel()


    def bundle_adjustment_sparsity(n_point3D):
        """
        n_observation:total length of pixel coordinates

        """
        m = n_point3D * 2 * 2 #row
        n = n_point3D * 3 + 12*num_of_cameras #col 
        A = lil_matrix((m, n), dtype=int)

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
                    A[2*i+1,s1:] = 1
                else:
                    A[2*i,s1:s2] = 1
                    A[2*i+1,s1:s2] = 1 
        
        elif num_of_cameras == 4:
            if base_Cam_Index == 'A':
                A[:m//2,-48:-36] = 1
            elif base_Cam_Index == 'B':
                A[:m//2,-36:-24] = 1
            elif base_Cam_Index == 'C':
                A[:m//2,-24:-12] = 1
            elif base_Cam_Index == 'D':
                A[:m//2,-12:] = 1
            
            for i in range(n_point3D):
                s1,s2 = (VIS_cam_List[i]-4)*12,(VIS_cam_List[i]-4)*12+12
                if s2 == 0:
                    A[2*i,s1:] = 1
                    A[2*i+1,s1:] = 1
                else:
                    A[2*i,s1:s2] = 1
                    A[2*i+1,s1:s2] = 1 
                    

        for i in range(n_point3D):
            for s in range(3):
                A[2*i,i*3+s] =1 
                A[2*i+1,i*3+s] =1
    

        A[m//2:,:-12*num_of_cameras] = A[:m//2,:-12*num_of_cameras]

        
        return A
    
    
    
    residual = fun(ba_input)
    print('max index',np.argmax(residual))
    A = bundle_adjustment_sparsity(Len_of_frame*points_inFrame)
    plt.plot(residual)
    plt.show()

    x0 = ba_input

    t0 = time.time()
    res = least_squares(fun,x0,jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf')
    #res = least_squares(fun,x0, verbose=2, x_scale='jac', ftol=1e-4, method='trf')
    t1 = time.time()
    print("Optimization took {0:.0f} seconds".format(t1 - t0))

    plt.plot(res.fun)
    plt.show()

    param = res.x
    print(param.shape)
    optimized_3D = param[:Len_of_frame*3*points_inFrame]
    Optimized_Proj_mat = param[Len_of_frame*3*points_inFrame:]

    coords = optimized_3D.reshape((-1,points_inFrame,3))

    return coords,Optimized_Proj_mat

print("optimization started")

#C,M = SBA(Len_of_frame,Proj_Mat,BA_points2D,ba_input,VIS_cam_List)
print('coords shape',coords.shape)
np.save(SAVE_FOLDER+'out_original.npy',coords)
#np.save(SAVE_FOLDER+'out_optimized.npy',C)
print('save sussesful')

# if include_DLC:
#     l = len(M)//num_of_cameras
    
#     if num_of_cameras == 2:
#         P1,P2 = (M[:l].reshape((3,4)),M[l:].reshape((3,4)))
#         Proj_points = Proj_points[:,-1:,:,:]
#         Proj_Mat = np.stack((P1,P2),axis=0)
    
#     elif num_of_cameras == 3:
#         P1,P2,P3 = param[:l].reshape((3,4)),param[l:2*l].reshape((3,4)),param[2*l:].reshape((3,4))
#         Proj_points = Proj_points[:,-1:,:,:]
#         Proj_Mat = np.stack((P1,P2,P3),axis=0)
    
#     elif num_of_cameras == 4:
#         P1,P2,P3,P4 = param[:l].reshape((3,4)),param[l:2*l].reshape((3,4)),param[2*l:3*l].reshape((3,4)),param[3*l:].reshape((3,4))
#         Proj_points = Proj_points[:,-1:,:,:]
#         Proj_Mat = np.stack((P1,P2,P3),axis=0)

    
#     coords,VIS_cam_List = triangulateTest(Proj_points,Proj_Mat,base_cam[base_Cam_Index]).solveA()
#     ball_points = coords[:,:,:-1].reshape((-1,1,3))

#     C = np.concatenate((C,ball_points),axis=-2)

  

if num_of_cameras == 3:
    Vis(SourceVideoFolder+'/'+Source_video_List[0][0],SourceVideoFolder+'/'+Source_video_List[1][0],SourceVideoFolder+'/'+Source_video_List[2][0],coords).display()

elif num_of_cameras == 2:
    #Vis(SourceVideoFolder+'/'+Source_video_List[1][0],SourceVideoFolder+'/'+Source_video_List[0][0],None,C).display()
    Vis(SourceVideoFolder+'/'+Source_video_List[1][0],SourceVideoFolder+'/'+Source_video_List[0][0],None,coords).display()

elif num_of_cameras == 4:
    Vis(SourceVideoFolder+'/'+Source_video_List[0][0],SourceVideoFolder+'/'+Source_video_List[1][0],SourceVideoFolder+'/'+Source_video_List[2][0],C).display()
    




