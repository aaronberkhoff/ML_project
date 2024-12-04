import numpy as np
import matplotlib.pyplot as plt
import sklearn 
from data_handler import DataHandler
from sklearn.linear_model import \
    LinearRegression, Ridge, ElasticNet, Lasso, LogisticRegression
from feature_matrix import *

#load data
#directory = 'C:/Users/ajber/Desktop/College Classes/Fall_2024/Machine_Learning/project/ML_project/data'
directory = 'C:/Users/jacob/Documents/Python/ML_project/data'

data_handler = DataHandler(directory)

name_data = 'Position_2_Body_Noise'

X_data = data_handler.data[name_data][:,0]
Y_data = data_handler.data[name_data][:,0]
Z_data = data_handler.data[name_data][:,0]

name_data = 'Position_2_Body_No_Noise'

X_data_noNoise = data_handler.data[name_data][:,0]
Y_data_noNoise = data_handler.data[name_data][:,0]
Z_data_noNoise = data_handler.data[name_data][:,0]

time_between_measurements=10 #seconds
time_data=np.arange(0,X_data.shape[0]*time_between_measurements,time_between_measurements)

phaseX=time_data[np.argmax(X_data_noNoise)]
# phaseX=0


numSinusoidals=200
period=6200
polyDegree=5

time_data_featurized=fourier_featurize(time_data,numSinusoidals,period,phaseX)
# time_data_featurized=polynomial_featurize(time_data,polynomialDegree=polyDegree)


linreg=LinearRegression()
# linreg=Ridge()
linreg.fit(time_data_featurized,X_data)

testing_time_data=np.arange(0,X_data.shape[0]*time_between_measurements*2,time_between_measurements)
testing_time_data_featurized=fourier_featurize(testing_time_data,numSinusoidals,period)
# testing_time_data_featurized=polynomial_featurize(testing_time_data,polyDegree)



predictions=linreg.predict(fourier_featurize(np.linspace(0,max(time_data),150001),numSinusoidals,period,phaseX))

# plt.figure()
# plt.scatter(time_data,X_data)#-X_data_noNoise)
# # plt.plot(time_data,predictions,color='r')
# plt.plot(np.linspace(0,max(time_data),150001),predictions,color='r')
# plt.show()


prediction=linreg.predict(time_data_featurized)

# plt.figure()
# plt.scatter(time_data,X_data-X_data_noNoise)
# # plt.plot(time_data,predictions,color='r')
# plt.plot(time_data,prediction-X_data_noNoise,color='r')
# plt.show()




fig,axs=plt.subplots(2,1,figsize=(8,6),constrained_layout=True)

axs[0].scatter(time_data,X_data,label='Observed Data',alpha=0.6)
axs[0].plot(np.linspace(0,max(time_data),150001),predictions,color='r',label='Prediction')
axs[0].set_title('Prediction vs Observed Data')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('X Data')
axs[0].legend()
axs[0].grid(True)

axs[1].scatter(time_data,X_data-X_data_noNoise,label='Noisy Data',alpha=0.6)
axs[1].plot(time_data,prediction-X_data_noNoise,color='r',label='Prediction Residuals')
axs[1].set_title('Prediction Residuals')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Residuals')
axs[1].legend()
axs[1].grid(True)

plt.show()





pass 
