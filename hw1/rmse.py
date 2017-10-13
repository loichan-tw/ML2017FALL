import numpy as np
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
data = pd.read_csv('./lamda2/test9allf01.csv',header = None, usecols = range(2)).as_matrix()
data_r = pd.read_csv('test_y.csv',header = None, usecols = range(2)).as_matrix()
y_v = np.zeros((120,1),dtype = np.float)
y = np.zeros((120,1),dtype = np.float)
for i in range(120):
    y_v[i,0] = float(data[i+1,1])
    y[i,0] = float(data_r[i+1,1])
c  = np.sum(np.square((y_v- y)),axis = 0)/120

print(c**0.5)