import numpy as np
import os
from scipy.io import loadmat


class DataHandler:

    def __init__(self, directory):

        self.data = {}

        for file in os.listdir(directory):

            if file.endswith('.mat'):

                file_path = os.path.join(directory,file)

                try: 

                    # Use the filename without the .mat extension as the key
                    key = os.path.splitext(file)[0]
                    data = loadmat(file_path)
                    variable_keys = [k for k in data.keys() if not k.startswith('__')]

                    self.data[key] = data[variable_keys[0]]
                    pass
                    
                
                     # Load the .mat file
                except Exception as e:
                    print(f"Error loading {file}: {e}")

                    



