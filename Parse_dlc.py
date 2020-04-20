import pandas as pd 
import numpy as np 
import glob 
import os
from config import  cam_names
from create_project import baseFilePath


def Parse_dlc():
    #set file path to dlc data
    path = baseFilePath+'/Intermediate/DeepLabCut/'
    #create folder for parsed output
    if not os.path.exists(path + 'DLCnpy'):
        os.mkdir(path+ 'DLCnpy')

    #Load all dlc csv output files  
    csvfile = glob.glob(path+'/*csv')

    #For loop gets csv data from all cameras
    j = 0
    for data in csvfile:     
        datapoints = pd.read_csv(data) # read in the csv data 
        parsedDlcData = datapoints.iloc[3:,7:10].values#the last element in the array is the P value

        print(parsedDlcData.shape)
        np.save(path+'DLCnpy/dlc_'+cam_names[j]+'.npy',parsedDlcData)#Save data
        j = j+1
        