import numpy as np
import pandas as pd
import time
import pickle

def test(w,w2,b):
  x_te = np.zeros((240,27),dtype = np.float)
  with open('./data/test.csv') as fp:
    id_ = 0
    for line in fp:
      vector = line.strip().split(',')
      if vector[1] == 'PM2.5':
        x_te[id_,:9] = vector[2:]
        id_ += 1
      elif vector[1] == 'AMB_TEMP':
        x_te[id_,9:18] = vector[2:]
      elif vector[1] == 'PM10':
        x_te[id_,18:] = vector[2:]
  
  y_te = b + x_te.dot(w) + np.square(x_te).dot(w2)# + np.power(x_te,3).dot(w3)# + np.power(x_te,4).dot(w4) + np.power(x_te,5).dot(w5)
  with open('./result/res.csv','w') as fp:
    fp.write('id,value\n')
    for i in range(240):
      fp.write('id_' + str(i) + ',' + str(y_te[i,0]) + '\n')
  fp.close()
w=pickle.load(open("w.dat", "rb"))
w2=pickle.load(open("w2.dat", "rb"))
b=pickle.load(open("b.dat", "rb"))
test(w,w2,b)



