import pandas as pd 
import numpy as np 
import glob 
import os
from config import baseFilePath
        
def Parse_dlc():
    path = baseFilePath+'/Intermediate/DeepLabCut/'
 
    if not os.path.exists(path + 'DLCnpy'):
        os.mkdir(path+ 'DLCnpy')
    
    pathLength = len(path)
    
    csvfile = glob.glob(path+'/*csv')
    
    for data in csvfile: 
        
        cam_name  = data[pathLength:]
        cam_name = cam_name[:4]
       
        value = pd.read_csv(data) 
        BallMotion = value.iloc[3:,7:10].values#the last element in the array is the P value

        print(BallMotion.shape)
        np.save(path+'DLCnpy/dlc_'+cam_name+'.npy',BallMotion)
