#from pykalman import KalmanFilter
import numpy as np
from scipy import signal
from config import cam_names



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

def kalmansmoothReconstruction(Inputfilepath):
    ''' Function Input is the filepath for the reconstructed output
    The function smooths the output using the kalman filter'''
    x = range(n_timesteps)
    data= np.load(Inputfilepath)
    
    # create a Kalman Filter by hinting at the size of the state and observation
    # space.  If you already have good guesses for the initial parameters, put them
    # in here.  The Kalman Filter will try to learn the values of all variables.
    kf = KalmanFilter(transition_matrices=np.array([[0, 1], [0, 1]]),
                    transition_covariance=0.1 * np.eye(2))
    
   
    states_pred = kf.em(onePointX).filter(onePointX)
    likelihood = kf.loglikelihood(onePointX)
    amountOfPoints = len(data[0])
    likelihood= data
    for ii in range(amountOfPoints):
        for kk in range(3):
            kalFilt = kf.em(data[:,ii,kk]).smooth(data[:,ii,kk])
            data[:,ii,kk] = kalFilt
            pointLikelihood = kf.loglikelihood(data[:,ii,kk])
            likelihood[:,ii,kk] = pointLikelihood
    np.save(Inputfilepath +'/Filtered3DPoints.npy', data)
      
    