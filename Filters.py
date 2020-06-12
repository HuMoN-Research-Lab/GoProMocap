from pykalman import KalmanFilter
import numpy as np
from scipy import signal
from create_project import GetVariables
from matplotlib import pyplot as plt
from ops import cam_names
configVariables = GetVariables()
cam_names = configVariables[1]

def smoothOpenPose(Inputfilepath):
    '''Function Input is Parsed Openpose data filepath
    The function smooths the Openpose data and then saves it to the same folder with Smooth in front of file name'''
    for jj in range(len(cam_names)):
        data = np.load(Inputfilepath +'/OP_'+cam_names[jj]+'.npy')
        amountOfOpPoints = len(data[0])
        for ii in range(amountOfOpPoints):
            for kk in range(3):
                if kk == 0 or kk ==1:
                    filt = signal.savgol_filter(data[:,ii,kk],51,3)
                if kk ==2:
                    filt = data[:,ii,kk]
            data[:,ii,kk] = filt
        np.save(Inputfilepath +'/SmoothedOP_'+cam_names[jj]+'.npy', data)

def kalman(Inputfilepath):
    ''' Function Input is the filepath for the openpose output
    The function smooths the output using the kalman filter'''
    #Intialize a Kalman Filter with a standard transistion matrix znd covariance
    kf = KalmanFilter(transition_matrices=np.array([[0, 1], [0, 1]]),
                            transition_covariance=0.1 * np.eye(2))
    colors = ['b','g','k','m']#List for colors of plot
    colors2 = ['c','y','.5','r']#List for corresponding colors of plot
    markers =['*','o']#List of marker types
    for ii in range(len(cam_names)):#iterates through the amount of cameras
        data= np.load(Inputfilepath+'/SmoothedOP_'+cam_names[ii]+'.npy')#load in the npy file of each camera
        plt.plot(data[:,0,2],marker=markers[0],color = colors[ii])#plot the original data before filter
        for jj in range(data.shape[1]):#Iterate through every point 
            for kk in range(3):#Iterates through XYZ coords
                kalFilt, _ = kf.smooth(data[:,jj,kk])#Kal filter for every frame of for each point
                data[:,jj,kk] = kalFilt[:,0]#replace the data in the data array with the new filtered data

            print('Loop #',jj)
        plt.plot(data[:,0,2],marker=markers[1],color = colors2[ii])#Plot Filtered Data
        np.save(Inputfilepath +'/KalmanOP_'+cam_names[ii]+'.npy', data)#Save out the new 

    
def butterFilt(Inputfilepath):
    '''Function Input is Parsed Openpose data filepath
    The function smooths the Openpose data and then saves it to the same folder with Smooth in front of file name'''
    
    colors = ['b','g','k','m']#List for colors of plot
    colors2 = ['c','y','.5','r']#List for corresponding colors of plot
    markers =['*','o']#List of marker types
    fc = 30 
    fs = 1000
    w = fc/ (fs/2)
    #for jj in range(len(cam_names)):
    data = np.load(Inputfilepath +'/OP_CamE.npy')
    data2 = np.load(Inputfilepath+'/RAWOP_CamE.npy')
    filtData = data
    plt.plot(data[:,0,1],marker=markers[0],color = colors[0])#plot the original data before filter
    plt.plot(data2[:,0,1],marker=markers[1],color = colors[1])#plot the original data before filter

    amountOfOpPoints = len(data[0])
    '''
    for ii in range(amountOfOpPoints):
        for kk in range(3):
            if kk == 0 or kk ==1:
                b, a  = signal.butter(4,w,'low')
                filtData[:,ii,kk] = signal.filtfilt(b,a, data[:,ii,kk])
            if kk ==2:
                filtData[:,ii,kk] = data[:,ii,kk]
        
    #lt.plot(filtData[:,0,1],marker=markers[1],color = colors2[0])#Plot Filtered Data
    np.save(Inputfilepath +'/OP_CamE.npy', data)                 
    '''
    plt.xlabel('Frame #')#Plot xlabel
    plt.ylabel('Pixel Coordinates')#Plot ylabel
    plt.title('Y-Coord of Randomly Selected OpenPose point ')#Plot title
    plt.legend(('CamE Higher Quality Open Pose Process','CamE Lower Quality Open Pose Process','CamF Unfiltered','CamF Filtered','CamG Unfiltered','CamG Filtered','CamH Unfiltered','CamH Filtered'),
                loc = 'upper right')#plot legend
    plt.show()#Show the plot
    print('')