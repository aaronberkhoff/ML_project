#Intro to ML Final Project
#This file generates the feature matrix for the final project
#12-3-2024
import numpy as np

def fourier_featurize(x,numSinusoidals=20,period=2*np.pi):
    # if x is a vector of 5 input values and numSinusoidals=20
    # x_featurized will be 5x41, the first one is the intercept
    # period is the orbit period in seconds
    coefficient=2*np.pi/period
    x_featurized=np.zeros([x.shape[0],(numSinusoidals*2)+1])
    x_featurized[:,0]=1
    for i in range(numSinusoidals):
        x_featurized[:,2*i+1]=np.cos((i+1)*coefficient*x)
        x_featurized[:,2*i+2]=np.sin((i+1)*coefficient*x)
    return x_featurized

def polynomial_featurize(x,polynomialDegree=5):
    # if x is a vector of 5 input values and polynomialDegree=20
    # x_featurized will be 5x21, the first one is the intercept
    x_featurized=np.zeros([x.shape[0],(polynomialDegree)+1])
    for i in range(polynomialDegree+1):
        x_featurized[:,i]=x**i
    return x_featurized

def combination_featurize(x,numSinusoidals=20,period=2*np.pi,polynomialDegree=5):
    # if x is a vector of 5 input values and polynomialDegree=20
    # x_featurized will be 5x21, the first one is the intercept
    # period is the orbit period in seconds
    coefficient=2*np.pi/period
    x_featurized=np.zeros([x.shape[0],(numSinusoidals*2)+(polynomialDegree)+1])
    for i in range(polynomialDegree+1):
        x_featurized[:,i]=x**i
    for i in range(numSinusoidals):
        x_featurized[:,2*i+polynomialDegree+1]=np.cos((i+1)*coefficient*x)
        x_featurized[:,2*i+polynomialDegree+2]=np.sin((i+1)*coefficient*x)
    return x_featurized

