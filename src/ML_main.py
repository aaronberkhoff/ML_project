import numpy as np
import matplotlib.pyplot as plt
import sklearn 
from data_handler import DataHandler

#load data
directory = 'C:/Users/ajber/Desktop/College Classes/Fall_2024/Machine_Learning/project/ML_project/data'

data_handler = DataHandler(directory)

name_data = 'Position_2_Body_No_Noise'

X_data = data_handler.data[name_data][:,0]
Y_data = data_handler.data[name_data][:,0]
Z_data = data_handler.data[name_data][:,0]


pass 
