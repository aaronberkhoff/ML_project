import numpy as np
import matplotlib.pyplot as plt
import sklearn 
from data_handler import DataHandler
from plot import MLPlot

#load data
directory = 'C:/Users/ajber/Desktop/College Classes/Fall_2024/Machine_Learning/project/ML_project/data'

data_handler = DataHandler(directory)

name_data_truth = 'Position_2_Body_No_Noise'
name_data_training = 'Position_2_Body_Noise'

X_data_truth = data_handler.data[name_data_truth].X_data
Y_data_truth = data_handler.data[name_data_truth].Y_data
Z_data_truth = data_handler.data[name_data_truth].Z_data

X_data_training = data_handler.data[name_data_training].X_data
Y_data_training = data_handler.data[name_data_training].Y_data
Z_data_training = data_handler.data[name_data_training].Z_data










pass 
