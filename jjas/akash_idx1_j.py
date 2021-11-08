






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
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping










from hyperopt import Trials, STATUS_OK, tpe, rand
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras import optimizers

path="/home/akashaw/iitk_data"
os.chdir(path)
os.getcwd()


def data():
    import os;
    path="/home/akashaw/iitk_data/Bais_Correc_PCA_Data/BIAS_CORR_PCA_ANA_GCM_OUTPUT/CCCma-CanESM2/DJF/INPUT"
    os.chdir(path)
    os.getcwd()
    logdir = os.path.join('cn_selu_callbacks')
    
    path="/home/akashaw/iitk_data/Bais_Correc_PCA_Data/BIAS_CORR_PCA_ANA_GCM_OUTPUT/CCCma-CanESM2/JJAS"
    os.chdir(path)
    os.getcwd()
    v=np.loadtxt("canesm_testing_jjas_idx1.csv", delimiter=",")
    path="/home/akashaw//iitk_data/NCEP_obs/Central_djf_obs_idx_sorted"
    os.chdir(path)
    os.getcwd()
    t2=np.loadtxt("obs_testing_jjas_idx1.csv", delimiter=",")
    
    X_test=v[:,0:14]
    Y_test=t2[:,int(sys.argv[1])]
    path="/home/akashaw//iitk_data/Bais_Correc_PCA_Data/BIAS_CORR_PCA_ANA_GCM_OUTPUT/CCCma-CanESM2/JJAS"
    os.chdir(path)
    os.getcwd()
    
    
    train_x=np.loadtxt("canesm_training_jjas_idx1.csv", delimiter=",")
    
    path="/home/akashaw/iitk_data/NCEP_obs/Central_djf_obs_idx_sorted"
    os.chdir(path)
    os.getcwd()
    
    train_y=np.loadtxt("obs_taining_jjas_idx1.csv", delimiter=",")
    path="/home/akashaw/iitk_data"

    os.chdir(path)
    os.getcwd()
    
    x=train_x[:,0:14]
    y=train_y[:,int(sys.argv[1])]
    y=np.reshape(y, (-1,1))
    scaler_x = MinMaxScaler(feature_range=(0.1, 0.9))
    print(scaler_x.fit(x))
    xscale=scaler_x.transform(x)
    scaler_y = MinMaxScaler(feature_range=(0.1, 0.9))
    print(scaler_y.fit(y))
    yscale=scaler_y.transform(y)
    Y_test=np.reshape(Y_test, (-1,1))
    X_test=scaler_x.transform(X_test)
    y_test=scaler_y.transform(Y_test)
   


    

    X_train, X_val, y_train, y_val = train_test_split(xscale, yscale, test_size=0.25, random_state=777)
    return X_train, X_val, X_test, y_train, y_val, y_test





def predict1( model):  

    path="/home/akashaw//iitk_data/Bais_Correc_PCA_Data/BIAS_CORR_PCA_ANA_GCM_OUTPUT/CCCma-CanESM2/DJF/INPUT"
    os.chdir(path)
    os.getcwd()
    logdir = os.path.join('cn_selu_callbacks')

    path="/home/akashaw//iitk_data/Bais_Correc_PCA_Data/BIAS_CORR_PCA_ANA_GCM_OUTPUT/CCCma-CanESM2/JJAS"
    os.chdir(path)
    os.getcwd()
    v=np.loadtxt("canesm_testing_jjas_idx1n.csv", delimiter=",")
    path="/home/akashaw//iitk_data/NCEP_obs/Central_djf_obs_idx_sorted"
    os.chdir(path)
    os.getcwd()
    t2=np.loadtxt("obs_testing_jjas_idx1.csv", delimiter=",")

    X_test=v[:,0:14]
    Y_test=t2[:,int(sys.argv[1])]
    path="/home/akashaw//iitk_data/Bais_Correc_PCA_Data/BIAS_CORR_PCA_ANA_GCM_OUTPUT/CCCma-CanESM2/JJAS"
    os.chdir(path)
    os.getcwd()


    train_x=np.loadtxt("canesm_training_jjas_idx1.csv", delimiter=",")

    path="/home/akashaw//iitk_data/NCEP_obs/Central_djf_obs_idx_sorted"
    os.chdir(path)
    os.getcwd()

    train_y=np.loadtxt("obs_taining_jjas_idx1.csv", delimiter=",")
    path="/home/akashaw/iitk_data"
    os.chdir(path)
    os.getcwd()

    x=train_x[:,0:14]
    y=train_y[:,int(sys.argv[1])]
    y=np.reshape(y, (-1,1))
    scaler_x = MinMaxScaler(feature_range=(0.1, 0.9))
    print(scaler_x.fit(x))
    xscale=scaler_x.transform(x)
    scaler_y = MinMaxScaler(feature_range=(0.1, 0.9))
    print(scaler_y.fit(y))
    yscale=scaler_y.transform(y)
    Y_test=np.reshape(Y_test, (-1,1))
    X_test=scaler_x.transform(X_test)
    y_test=scaler_y.transform(Y_test)
    ynew= model.predict(X_test)
    #invert normalize
    ytestoutput = scaler_y.inverse_transform(ynew) 
    Xtestgcm = scaler_x.inverse_transform(X_test)
    ytestobs=scaler_y.inverse_transform(y_test) 
    yn_tr= model.predict(xscale)
    #invert normalize
    yobs = scaler_y.inverse_transform(yscale) 
    yn_tr = scaler_y.inverse_transform(yn_tr) 
    Xgcm = scaler_x.inverse_transform(xscale)   



    return yobs,yn_tr,Xgcm,ytestoutput,Xtestgcm,ytestobs



from hyperopt import Trials, STATUS_OK, tpe, rand


def create_model(X_train, y_train, X_val, y_val):
    from keras import models
    from keras import layers
    import numpy as np

    model = models.Sequential()
    model.add(layers.Dense({{choice([5,10,15,20,25,30,35,40])}}, input_dim=14,kernel_initializer='he_uniform'))
    model.add(LeakyReLU(alpha={{uniform(0.5, 1)}}))
    model.add(Dropout({{uniform(0, 1)}}))
    
    model.add(layers.Dense(1, activation='relu'))

    from keras import callbacks
        
           
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)


    model.compile(optimizer='adam',loss='mse',metrics=['mape'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    

    model.fit(X_train,y_train,epochs=500,batch_size={{choice([16, 32, 64])}},validation_data=(X_val, y_val),callbacks=[reduce_lr,es])

    score, acc = model.evaluate(X_val, y_val, verbose=0)
    print('Test accuracy:', acc)
    return {'loss':score , 'status': STATUS_OK, 'model': model}



      
T = Trials()
best_run, best_model = optim.minimize(model=create_model,data=data,algo=tpe.suggest,max_evals=15,trials=T)
X_train, X_val, X_test, y_train, y_val, y_test = data()
print("Evalutation of best performing model:")
print(best_model.evaluate(X_test, y_test))
print("Best performing model chosen hyper-parameters:")
print(best_run)

best_model.save('/home/akashaw/iitk_data/JJAS/model_idx1_jjas_central/model{}.h5'.format(int(sys.argv[1])))



x,y,z,x1,y1,z1=predict1(best_model)




#c=np.mean(f)





from numpy import asarray
from numpy import savetxt
savetxt("/home/akashaw/iitk_data/JJAS/output_idx1_central_testing_jjas/output_again_test_jjas_idx1{}.csv".format(int(sys.argv[1])),x1,delimiter=',',fmt='%10f')
savetxt("/home/akashaw/iitk_data/JJAS/output_idx1_central_training_jjas/output_again_training_jjas_idx1{}.csv".format(int(sys.argv[1])),y,delimiter=',',fmt='%10f')


