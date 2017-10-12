import numpy as np
import pandas as pd
import pickle
import time

def gendata():
  data = pd.read_csv('./data/train.csv',usecols = range(3,27),encoding='big5').as_matrix()
  data_np = np.zeros((18,5760),dtype = np.float)
  for i in range(18):
    for j in range(240):
      for k in range(24):
        tmp = data[j*18 + i,k]
        if tmp == 'NR':
          d = 0
        elif tmp == '-1':
          d = d
        else:
          d = float(tmp)
        data_np[i,j*24+k] = d
  '''
  train_list = []
  with open('./data/train.csv') as fp:
    for line in fp:
      vector = line.strip().split(',')
      if vector[2] == 'PM2.5':
        for v in vector[3:]:
          train_list.append(float(v))
  '''
  x  = np.zeros((5652,162),dtype = np.float)
  y_ = np.zeros((5652,1),dtype = np.float)
  for i in range(12):
    for j in range(471):
      for k in range(18):
        x[i*471+j,k*9:(k+1)*9] = data_np[k,i*480+j:i*480+j+9]
      y_[i*471+j,0] = data_np[9,i*480+j+9]
        
  #x = (x - np.mean(x,0).reshape(1,162))/np.std(x,0).reshape(1,162)
  #y_std = np.square(np.std(y_,0))
  #y_ = (y_ - np.mean(y_,0))/np.std(y_,0)
  return [x,y_]

def linearRegression(x,y_):

#  x_mean = np.mean(x,0)
#  x_std  = np.std(x,0)
#  y_mean = np.mean(y_,0)
#  y_std  = np.std(y_,0)
#  x = (x-x_mean)
#  y_ = (y_-y_mean)
  [x_t,x_v] = np.array_split(x,[5000],axis = 0)
  [y_t,y_v] = np.array_split(y_,[5000],axis = 0)
  w  = 1e-13*np.random.random((162,1))
  dw = np.zeros((162,1),dtype = np.float)
  dw_t = np.zeros((162,1),dtype = np.float)
  m1 = np.zeros((162,1),dtype = np.float)

  w2 = 1e-13*np.random.random((162,1))
  dw2= np.zeros((162,1),dtype = np.float)
  dw2_t = np.zeros((162,1),dtype = np.float)
  m2 = np.zeros((162,1),dtype = np.float)
  '''
  w3  = 1e-13*np.random.random((9,1))
  dw3 = np.zeros((9,1),dtype = np.float)
  dw3_t = np.zeros((9,1),dtype = np.float)
  m3 = np.zeros((9,1),dtype = np.float)

  w4  = 1e-13*np.random.random((9,1))
  dw4 = np.zeros((9,1),dtype = np.float)
  dw4_t = np.zeros((9,1),dtype = np.float)
  m4 = np.zeros((9,1),dtype = np.float)

  w5  = 1e-13*np.random.random((9,1))
  dw5 = np.zeros((9,1),dtype = np.float)
  dw5_t = np.zeros((9,1),dtype = np.float)
  m5 = np.zeros((9,1),dtype = np.float)
  '''
  b  = 1e-13*np.random.random((1,1))
  db = np.zeros((1,1),dtype = np.float)
  db_t = np.zeros((1,1),dtype = np.float)
  m0 = np.zeros((1,1),dtype = np.float)

  r = 0.001 
  rd = 0.0
  rg = 0.0
  x_t2 = np.square(x_t)
  '''
  x_t3 = x_t * x_t2
  x_t4 = x_t * x_t3
  x_t5 = x_t * x_t4
  '''
  x_v2 = np.square(x_v)
  '''
  x_v3 = x_v * x_v2
  x_v4 = x_v * x_v3
  x_v5 = x_v * x_v4
  '''
  tstart = time.time()
  for i in range(1000000):
    y  = b + x_t.dot(w) + x_t2.dot(w2)# + x_t3.dot(w3)# + x_t4.dot(w4)# + x_t5.dot(w5)
    e  = y_t - y
    c  = np.sum(np.square(e),axis = 0)/5000
    if i%100 == 0:
      print('train cost is %f:' %c)

    dw = np.sum(-x_t*(e),axis = 0).reshape(162,1) - rg * w
    dw_t = np.sqrt(np.square(dw_t) +  np.square(dw))
    m1 = rd * m1 - r * (dw/dw_t)

    dw2= np.sum(-x_t2*(e),axis =0).reshape(162,1) - rg * w2
    dw2_t = np.sqrt(np.square(dw2_t) +  np.square(dw2))
    m2 = rd * m2 - r * (dw2/dw2_t)
    '''    
    dw3= np.sum(-x_t3*(e),axis =0).reshape(9,1) - rg * w3
    dw3_t = np.sqrt(np.square(dw3_t) + np.square(dw3))
    m3 = rd * m3 - r * (dw3/dw3_t)

    dw4= np.sum(-x_t4*(e),axis =0).reshape(9,1) - rg * w4
    dw4_t = np.sqrt(np.square(dw4_t) + np.square(dw4))
    m4 = rd * m4 - r * (dw4/dw4_t)

    dw5= np.sum(-x_t5*(e),axis =0).reshape(9,1) - rg * w5
    dw5_t = np.sqrt(np.square(dw5_t) + np.square(dw5))
    m5 = rd * m5 - r * (dw5/dw5_t)
    '''
    db = np.sum((e),axis = 0) - rg *b
    db_t = np.sqrt(np.square(db_t) + np.square(db))
    m0 = rd * m0 - r * (db/db_t)

    y  = b + x_v.dot(w) + x_v2.dot(w2)# + x_v3.dot(w3)# + x_v4.dot(w4)# + x_v5.dot(w5)
    e  = y_v - y
    c  = np.sum(np.square(e),axis = 0)/652
    if i%100 == 0:
      print ('valid cost is %f:' %c)
    if i%10000 == 0:
      print ('%d%% took %f sec:' %((i/10000),(time.time() - tstart)))
      tstart = time.time()
    w  = w + m1
    w2 = w2 + m2
    #w3 = w3 + m3
    #w4 = w4 + m4
    #w5 = w5 + m5
    b  = b + m0
  return [w,w2,b]

def test(w,w2,b):
  data = pd.read_csv('./data/test.csv',header = None, usecols = range(2,11)).as_matrix()
  x_te = np.zeros((240,162),dtype = np.float)
  for i in range(240):
    for j in range(18):
      for k in range(9):
        tmp = data[i*18+j,k]
        if tmp == 'NR':
          x_te[i,j*9+k] = 0
        else:
          x_te[i,j*9+k] = float(tmp)
  #x_te = (x_te - np.mean(x_te,0))/np.std(x_te,0)
  y_te = b + x_te.dot(w) + np.square(x_te).dot(w2)# + np.power(x_te,3).dot(w3)# + np.power(x_te,4).dot(w4) + np.power(x_te,5).dot(w5)
  with open('./data/test_Y.csv','w') as fp:
    fp.write('id,value\n')
    for i in range(240):
      fp.write('id_' + str(i) + ',' + str(y_te[i,0]) + '\n')
  print ('testing complete')

[x,y_] = gendata()
[w,w2,b] = linearRegression(x,y_)

f = open("w.dat", "wb")
pickle.dump(w,f)
f.close()
'''
f1 = open("w2.dat", "wb")
pickle.dump(w2,f1)
f1.close()
'''
f2 = open("b.dat", "wb")
pickle.dump(b,f)
f2.close()
test(w,w2,b)

