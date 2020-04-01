import pandas as pd 
import numpy as np 

        
def Parse_dlc():
    path = baseProjectPath+'/'+subject+'/'+sessionID+'/DeepLabCut/'
 
    if not os.path.exists(path + 'DLCnpy'):
        os.mkdir(path+ 'DLCnpy')
    

    csvfile = glob.glob(path+'/*csv')
    
    for data in csvfile: 
        
        cam_name  = data[56:]
        cam_name = cam_name[:4]
       
        value = pd.read_csv(data) 
        BallMotion = value.iloc[3:,7:10].values#the last element in the array is the P value

        print(BallMotion.shape)
        np.save(baseProjectPath+'/'+subject+'/'+sessionID+'/DeepLabCut/DLCnpy/dlc_'+cam_name+'.npy',BallMotion)