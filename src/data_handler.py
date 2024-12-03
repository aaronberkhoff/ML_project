import numpy as np
import os
from scipy.io import loadmat


class DataHandler:

    def __init__(self, directory,dt = 10):

        self.data = {}

        for file in os.listdir(directory):

            if file.endswith('.mat'):

                file_path = os.path.join(directory,file)

                try: 

                    # Use the filename without the .mat extension as the key
                    key = os.path.splitext(file)[0]
                    data = loadmat(file_path)
                    variable_keys = [k for k in data.keys() if not k.startswith('__')]
                    time_data = np.linspace(0,data[variable_keys[0]][0].shape[0])

                    self.data[key] = Data(name = key,
                                          X_data = data[variable_keys[0]][:,0],
                                          Y_data = data[variable_keys[0]][:,1],
                                          Z_data = data[variable_keys[0]][:,2],
                                          time_data = time_data)
                    pass
                    
                     # Load the .mat file
                except Exception as e:
                    print(f"Error loading {file}: {e}")


class Data:

    def __init__(self,name,X_data,Y_data,Z_data,time_data):

        self.X_data = X_data
        self.Y_data = Y_data
        self.Z_data = Z_data
        self.time_data = time_data
        self.name      = name

        self.X_fitted_data = None
        self.Y_fitted_data = None
        self.Z_fitted_data = None




        
            

                    



