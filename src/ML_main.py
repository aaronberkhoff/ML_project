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
print(time_data[0:3])
print(time_data[-3:-1])

#optimal_theta=np.linalg.inv(X_values.T@X_values)@X_values.T@y_train.values
plt.figure()
plt.scatter(time_data,X_data-X_data_noNoise)
plt.show()




pass 
