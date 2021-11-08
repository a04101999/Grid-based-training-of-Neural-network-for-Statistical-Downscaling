import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import keras
import keras.backend as K
import sys

import hyperopt
import numpy as np
import os
import time





import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler



import pandas as pd 
import os
from keras.models import load_model

data = {'Days': [i for i in range(0,8148)]}
df=pd.DataFrame(data)
  
path='/home/akashaw/iitk_data/Bais_Correc_PCA_Data/BIAS_CORR_PCA_ANA_GCM_OUTPUT/CCCma-CanESM2/DJF'
os.chdir(path)

da=np.loadtxt('future_rcp45_djfidx1.csv', delimiter=",")
scaler_y = MinMaxScaler(feature_range=(0.1, 0.9))
x=da[:,0:16]
print(scaler_y.fit(x))
yscale=scaler_y.transform(x)
s=np.zeros((8148,1))
scaler_x = MinMaxScaler(feature_range=(0.1, 0.9))
print(scaler_x.fit(s))
xscale=scaler_x.transform(s)


  
path='/home/akashaw/iitk_data/NCEP_obs/Central_djf_obs_idx_sorted/model_idx1_djf_central'
os.chdir(path)
for i in range(0,1276):
    #import tensorflow.compat.v1 as tf
    #tf.disable_v2_behavior()
    
        model = load_model('model{}.h5'.format(i))
        model.compile(optimizer='adam',loss='mse',metrics=['mape'])
        result = model.predict(x)
        
        rf = scaler_x.inverse_transform(result) 

        
        df['Grid{}'.format(i+1)]=rf


df.to_csv('futute45_idx1.csv')