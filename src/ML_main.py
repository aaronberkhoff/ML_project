import numpy as np
import matplotlib.pyplot as plt
import sklearn 
from data_handler import DataHandler
from sklearn.linear_model import \
    LinearRegression, Ridge, ElasticNet, Lasso, LogisticRegression
from feature_matrix import *

#load data
directory = 'C:/Users/ajber/Desktop/College Classes/Fall_2024/Machine_Learning/project/ML_project/data'
# directory = 'C:/Users/jacob/Documents/Python/ML_project/data'

data_handler = DataHandler(directory)

name_data = 'Position_2_Body_Noise'
# name_data = 'Position_J2_Noise'
# name_data = 'Position_Spher_Noise'

X_data = data_handler.data[name_data][:,0] / 6378.14e3
Y_data = data_handler.data[name_data][:,1] / 6378.14e3
Z_data = data_handler.data[name_data][:,2] / 6378.14e3

name_data = 'Position_2_Body_No_Noise'
# name_data = 'Position_J2_No_Noise'
# name_data = 'Position_Spher_Noise'

X_data_noNoise = data_handler.data[name_data][:,0] / 6378.14e3
Y_data_noNoise = data_handler.data[name_data][:,1] / 6378.14e3
Z_data_noNoise = data_handler.data[name_data][:,2] / 6378.14e3

time_between_measurements=1 #seconds
time_data=np.arange(0,X_data.shape[0]*time_between_measurements,time_between_measurements)

numSinusoidals=10
period=620

time_data_featurized=fourier_featurize(time_data,numSinusoidals,period)
# time_data_featurized=polynomial_featurize(time_data,100)


# linreg=LinearRegression()
Xlinreg=Ridge(alpha=1,fit_intercept=False)
Ylinreg=Ridge(alpha=1,fit_intercept=False)
Zlinreg=Ridge(alpha=1,fit_intercept=False)
# linreg=Lasso()

Xlinreg.fit(time_data_featurized,X_data)
Ylinreg.fit(time_data_featurized,Y_data)
Zlinreg.fit(time_data_featurized,Z_data)

testing_time_data=np.arange(0,X_data.shape[0]*time_between_measurements*3,time_between_measurements)
testing_time_data_featurized=fourier_featurize(testing_time_data,numSinusoidals,period)


Xpredictions=Xlinreg.predict(testing_time_data_featurized)
Ypredictions=Ylinreg.predict(testing_time_data_featurized)
Zpredictions=Zlinreg.predict(testing_time_data_featurized)


plt.figure(1)
plt.scatter(time_data,X_data)#-X_data_noNoise)
# plt.scatter(time_data,X_data_noNoise)#-X_data_noNoise)
plt.plot(testing_time_data,Xpredictions,color='r')
plt.show()

plt.figure(2)
plt.scatter(time_data,Y_data)#-Y_data_noNoise)
# plt.scatter(time_data,Y_data_noNoise)#-Y_data_noNoise)
plt.plot(testing_time_data,Ypredictions,color='r')
plt.show()

plt.figure(3)
plt.scatter(time_data,Z_data)#-Z_data_noNoise)
# plt.scatter(time_data,Z_data_noNoise)#-Z_data_noNoise)
plt.plot(testing_time_data,Zpredictions,color='r')
plt.show()




pass 
