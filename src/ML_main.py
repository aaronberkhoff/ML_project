import numpy as np
import matplotlib.pyplot as plt
import sklearn 
from data_handler import DataHandler
from sklearn.linear_model import \
    LinearRegression, Ridge, ElasticNet, Lasso, LogisticRegression
from feature_matrix import *

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


#load data
#directory = 'C:/Users/ajber/Desktop/College Classes/Fall_2024/Machine_Learning/project/ML_project/data'
directory = 'C:/Users/jacob/Documents/Python/ML_project/data'

data_handler = DataHandler(directory)

name_data = 'Position_2_Body_No_Noise'

X_data_just2Body = data_handler.data[name_data][0:582,0]/1000
Y_data_just2Body = data_handler.data[name_data][0:582,1]/1000
Z_data_just2Body = data_handler.data[name_data][0:582,2]/1000

name_data = 'Position_Spher_Noise'

X_data = data_handler.data[name_data][0:582,0]/1000
Y_data = data_handler.data[name_data][0:582,1]/1000
Z_data = data_handler.data[name_data][0:582,2]/1000

name_data = 'Position_Spher_No_Noise'

X_data_noNoise = data_handler.data[name_data][0:582,0]/1000
Y_data_noNoise = data_handler.data[name_data][0:582,1]/1000
Z_data_noNoise = data_handler.data[name_data][0:582,2]/1000

time_between_measurements=10 #seconds
time_data=np.arange(0,X_data.shape[0]*time_between_measurements,time_between_measurements)


for i in range(3):
    if i==0:
        current_data_just2Body=X_data_just2Body
        current_data=X_data
        current_data_noNoise=X_data_noNoise
        axisLetter='X'
    if i==1:
        current_data_just2Body=Y_data_just2Body
        current_data=Y_data
        current_data_noNoise=Y_data_noNoise
        axisLetter='Y'
    if i==2:
        current_data_just2Body=Z_data_just2Body
        current_data=Z_data
        current_data_noNoise=Z_data_noNoise
        axisLetter='Z'


    phase=time_data[np.argmin(np.abs(current_data_just2Body))]
    # phase=0

    # peak1=np.argmax(current_data_noNoise[600:650])+600
    # peak2=np.argmax(current_data_noNoise[1200:1250])+1200
    # peak12=np.argmax(current_data_noNoise[7400:7450])+7400
    # period=10*(peak12-peak1)/11
    # period=6177.272727272727272727
    period=time_data.shape[0]*10

    numSinusoidals=20
    # period=6200
    polyDegree=2

    # time_data_featurized=fourier_featurize(time_data,numSinusoidals,period,phase)
    # time_data_featurized=polynomial_featurize(time_data,polynomialDegree=polyDegree)
    time_data_featurized=combination_featurize(time_data,numSinusoidals,period,polyDegree,phase)


    linreg=LinearRegression()
    # linreg=Ridge(alpha=1)
    # linreg=Lasso(alpha=1)
    # linreg.fit(time_data_featurized,current_data)
    linreg.fit(time_data_featurized,current_data-current_data_just2Body)


    testing_time_data=np.arange(0,current_data.shape[0]*time_between_measurements*2,time_between_measurements)
    # testing_time_data_featurized=fourier_featurize(testing_time_data,numSinusoidals,period,phase)
    # testing_time_data_featurized=polynomial_featurize(testing_time_data,polyDegree)
    testing_time_data_featurized=combination_featurize(testing_time_data,numSinusoidals,period,polyDegree,phase)



    # predictions=linreg.predict(fourier_featurize(np.linspace(0,max(time_data),150001),numSinusoidals,period,phase))
    # predictions=linreg.predict(polynomial_featurize(np.linspace(0,max(time_data),150001),polyDegree))
    predictions=linreg.predict(combination_featurize(np.linspace(0,max(time_data),150001),numSinusoidals,period,polyDegree,phase))


    # plt.figure()
    # plt.scatter(time_data,current_data)#-current_data_noNoise)
    # # plt.plot(time_data,predictions,color='r')
    # plt.plot(np.linspace(0,max(time_data),150001),predictions,color='r')
    # plt.show()


    prediction=linreg.predict(time_data_featurized)

    # plt.figure()
    # plt.scatter(time_data,current_data-current_data_noNoise)
    # # plt.plot(time_data,predictions,color='r')
    # plt.plot(time_data,prediction-current_data_noNoise,color='r')
    # plt.show()



    fig,axs=plt.subplots(2,1,figsize=(8,6),constrained_layout=True)

    axs[0].scatter(time_data,current_data-current_data_just2Body,label='Observed Data',alpha=0.6)
    # axs[0].scatter(time_data,current_data,label='Observed Data',alpha=0.6)
    axs[0].plot(np.linspace(0,max(time_data),150001),predictions,color='r',label='Prediction')
    axs[0].set_title('Prediction vs Observed Data')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel(axisLetter+' Data')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].scatter(time_data,current_data-current_data_noNoise,label='Noisy Data',alpha=0.6)
    axs[1].plot(time_data,prediction+current_data_just2Body-current_data_noNoise,color='r',label='Prediction Residuals')
    # axs[1].plot(time_data,prediction-current_data_noNoise,color='r',label='Prediction Residuals')
    axs[1].set_title('Prediction Residuals')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Residuals')
    axs[1].legend()
    axs[1].grid(True)

    


plt.show()


pass 
