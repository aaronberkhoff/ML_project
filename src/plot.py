import matplotlib.pyplot as plt
from data_handler import *


class MLPlot:

    def __init__(self,data:Data):

        self.data = data        

    def plot(figsize = (10,8), saveas = 'MLPlot.png'):

        fig = plt.figure()
        ax = fig.add_subplot()





        
