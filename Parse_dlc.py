import pandas as pd 
import numpy as np 



        
path = '214OP_data/coordData214/CamB.csv'
cam_name = 'CamB'      

value = pd.read_csv(path) 
            
BallMotion = value.iloc[3:,7:10].values#the last element in the array is the P value

print(BallMotion.shape)
np.save('PixelCoordData/dlc_'+cam_name+'.npy',BallMotion)