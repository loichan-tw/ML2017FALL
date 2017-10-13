import numpy as np
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
num_i=10000
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

def linearRegression(x,y_,num_i,rg,w):
 # num_i=100
#  x_mean = np.mean(x,0)
#  x_std  = np.std(x,0)
#  y_mean = np.mean(y_,0)
#  y_std  = np.std(y_,0)
#  x = (x-x_mean)
#  y_ = (y_-y_mean)
  [x_t,x_v] = np.array_split(x,[5000],axis = 0)
  [y_t,y_v] = np.array_split(y_,[5000],axis = 0)
  #w  = 0.1*np.random.random((162,1))
  #w = 0.1*np.ones((162,1),dtype = np.float)
  dw = np.zeros((162,1),dtype = np.float)
  dw_t = np.zeros((162,1),dtype = np.float)
  m1 = np.zeros((162,1),dtype = np.float)

  b  = 1e-13*np.random.random((1,1))
  db = np.zeros((1,1),dtype = np.float)
  db_t = np.zeros((1,1),dtype = np.float)
  m0 = np.zeros((1,1),dtype = np.float)

  r = 0.001 
  rd = 0.0
  #rg = 0.0001
#  x_t2 = np.square(x_t)

 # x_v2 = np.square(x_v)

  tstart = time.time()
  rms = np.zeros((int(num_i),1),dtype = np.float)
  rms2 = np.zeros((int(num_i),1),dtype = np.float)
  for i in range(num_i):
    y  = b + x_t.dot(w) #+ x_t2.dot(w2)# + x_t3.dot(w3)# + x_t4.dot(w4)# + x_t5.dot(w5)
    e  = y_t - y
    c  = np.sum(np.square(e),axis = 0)/5000
    #if i%100 == 0:
    #print('train cost is %f:' %c)
    rms[int(i),0]=c**0.5
    dw = np.sum(-x_t*(e),axis = 0).reshape(162,1) - rg * w
    dw_t = np.sqrt(np.square(dw_t) +  np.square(dw))
    m1 = rd * m1 - r * (dw/dw_t)


    db = np.sum((e),axis = 0) - rg *b
    db_t = np.sqrt(np.square(db_t) + np.square(db))
    m0 = rd * m0 - r * (db/db_t)

    y  = b + x_v.dot(w)# + x_v2.dot(w2)# + x_v3.dot(w3)# + x_v4.dot(w4)# + x_v5.dot(w5)
    e  = y_v - y
    c  = np.sum(np.square(e),axis = 0)/652
    #if i%100 == 0:
    #print ('valid cost is %f:' %c)
    rms2[int(i),0]=c**0.5
    if i%10000 == 0:
      print ('%d%% took %f sec:' %((i/10000),(time.time() - tstart)))
      tstart = time.time()
    w  = w + m1

    b  = b + m0
  return [w,b,rms,rms2]

def test(w,b,nnn):
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
  y_te = b + x_te.dot(w)# + np.square(x_te).dot(w2)# + np.power(x_te,3).dot(w3)# + np.power(x_te,4).dot(w4) + np.power(x_te,5).dot(w5)
  with open('./lamda/test9allf'+nnn+'.csv','w') as fp:
    fp.write('id,value\n')
    for i in range(240):
      fp.write('id_' + str(i) + ',' + str(y_te[i,0]) + '\n')
  print ('testing complete')



w  = 0.1*np.random.random((162,1))

[x,y_] = gendata()
[w,b,rms01,rms012] = linearRegression(x,y_,num_i,0.1,w)
f = open("./lamda/w_lamda01.dat", "wb")
pickle.dump(w,f)
f.close()
f2 = open("./lamda/b_lamda01.dat", "wb")
pickle.dump(b,f2)
f2.close()
test(w,b,'01')

[w,b,rms001,rms0012] = linearRegression(x,y_,num_i,0.01,w)
f = open("./lamda/w_lamda001.dat", "wb")
pickle.dump(w,f)
f.close()
f2 = open("./lamda/b_lamda001.dat", "wb")
pickle.dump(b,f2)
f2.close()
test(w,b,'001')

[w,b,rms0001,rms00012] = linearRegression(x,y_,num_i,0.001,w)
f = open("./lamda/w_lamda0001.dat", "wb")
pickle.dump(w,f)
f.close()
f2 = open("./lamda/b_lamda0001.dat", "wb")
pickle.dump(b,f2)
f2.close()
test(w,b,'0001')

[w,b,rms00001,rms000012] = linearRegression(x,y_,num_i,0.0001,w)
f = open("./lamda/w_lamda00001.dat", "wb")
pickle.dump(w,f)
f.close()
f2 = open("./lamda/b_lamda00001.dat", "wb")
pickle.dump(b,f2)
f2.close()
test(w,b,'00001')

t =np.arange(0., num_i, 1)
ls = 'solid'
fig, ax = plt.subplots(figsize=(12, 6))  # create figure 
erb1 =ax.errorbar( t, rms01 ,linestyle=ls)
erb2 = ax.errorbar(t, rms001, linestyle=ls)
erb3 = ax.errorbar(t, rms0001, linestyle=ls)
erb4 = ax.errorbar(t, rms00001, linestyle=ls)

ax.legend((erb1[0], erb2[0], erb3[0],erb4[0]), ('$\lambda$ = 0.1','$\lambda$ = 0.01','$\lambda$ = 0.001','$\lambda$ = 0.0001'))
ax.set_title(r'RMSE 9hr - all features training rmse, learning rate = 0.001')
ax.set_xlabel('number of iteration')
ax.set_ylabel('root mean square error')
#ax.set_ylim([5,30])
#ax.plot([0,1,2], [10,20,3])
fig.savefig('./lamda/9h_lamda.png')   # save the figure to file
#plt.show()
fig2, ax2 = plt.subplots(figsize=(12, 6))  # create figure 
erb12 =ax2.errorbar( t, rms012 ,linestyle=ls)
erb22 = ax2.errorbar(t, rms0012, linestyle=ls)
erb32 = ax2.errorbar(t, rms00012, linestyle=ls)
erb42 = ax2.errorbar(t, rms000012, linestyle=ls)
#erb2 = ax.errorbar(t, rms2, linestyle=ls)
ax.legend((erb12[0], erb22[0], erb32[0],erb42[0]), ('$\lambda$ = 0.1','$\lambda$ = 0.01','$\lambda$ = 0.001','$\lambda$ = 0.0001'))
ax.set_title(r'RMSE 9hr - all features testing rmse, learning rate = 0.001')
ax.set_xlabel('number of iteration')
ax.set_ylabel('root mean square error')
#ax.set_ylim([5,30])
#ax.plot([0,1,2], [10,20,3])
fig.savefig('./lamda/9h_lamda2.png')
'''
f = open("./lamda/w_lamda00001.dat", "wb")
pickle.dump(w,f)
f.close()

f2 = open("./lamda/b_lamda00001.dat", "wb")
pickle.dump(b,f2)
f2.close()
test(w,b)
'''

