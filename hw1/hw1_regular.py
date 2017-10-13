import numpy as np
import pickle

def gendata():
  pm25_list = []
  temp_list = []
  pm10_list = []
  with open('./data/train.csv') as fp:
    for line in fp:
      vector = line.strip().split(',')
      if vector[2] == 'PM2.5':
        for v in vector[3:]:
          pm25_list.append(float(v))
#      elif vector[2] == 'AMB_TEMP':
#        for v in vector[3:]:
#          temp_list.append(float(v))
#      elif vector[2] == 'PM10':
#        for v in vector[3:]:
#          pm10_list.append(float(v))
  for i in range(5760):
    if pm25_list[i] == -1:
      pm25_list[i] = tmp
    else:
      tmp = pm25_list[i]
  x  = np.zeros((5652,9),dtype = np.float)
  y_ = np.zeros((5652,1),dtype = np.float)
  for m in range(12):
    for d in range(471):
      x[471*m+d,:9] = pm25_list[m*480+d:m*480+d+9]
#      x[471*m+d,9:18] = temp_list[m*480+d:m*480+d+9]
#      x[471*m+d,18:27] = pm10_list[m*480+d:m*480+d+9]
      y_[471*m+d,0] = pm25_list[m*480+d+9]
#  x = (x - np.mean(x,0))/np.std(x,0)
#  y_ = (y_ - np.mean(y_,0))/np.std(y_,0)
  return [x,y_]

def linearRegression(x,y_,num_i,rg,w):

#  x_mean = np.mean(x,0)
#  x_std  = np.std(x,0)
#  y_mean = np.mean(y_,0)
#  y_std  = np.std(y_,0)
#  x = (x-x_mean)
#  y_ = (y_-y_mean)
  [x_t,x_v] = np.array_split(x,[5000],axis = 0)
  [y_t,y_v] = np.array_split(y_,[5000],axis = 0)
  #w  = 1e-13*np.random.random((9,1))
  dw = np.zeros((9,1),dtype = np.float)
  dw_t = np.zeros((9,1),dtype = np.float)
  m1 = np.zeros((9,1),dtype = np.float)
  '''
  w2 = 1e-13*np.random.random((27,1))
  dw2= np.zeros((27,1),dtype = np.float)
  dw2_t = np.zeros((27,1),dtype = np.float)
  m2 = np.zeros((27,1),dtype = np.float)

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

  r = 0.01 
  rd = 0
  #rg = 2.0
  '''
  x_t2 = np.square(x_t)

  x_t3 = x_t * x_t2
  x_t4 = x_t * x_t3
  x_t5 = x_t * x_t4
  '''
  #x_v2 = np.square(x_v)
  '''
  x_v3 = x_v * x_v2
  x_v4 = x_v * x_v3
  x_v5 = x_v * x_v4
  '''
  #fp = open ('./data/record.csv','w')
  #fp.write('learning rate, %f ,ramda, %f\n' %(r,rg))
  for i in range(num_i):
    y  = b + x_t.dot(w) #+ x_t2.dot(w2)# + x_t3.dot(w3)# + x_t4.dot(w4)# + x_t5.dot(w5)
    e  = y_t - y
    c  = np.sum(np.square(e),axis = 0)/5000

    dw = np.sum(-x_t*(e),axis = 0).reshape(9,1) - rg * w
    dw_t = np.sqrt(np.square(dw_t) +  np.square(dw))
    m1 = rd * m1 - r * (dw/dw_t)
    '''
    dw2= np.sum(-x_t2*(e),axis =0).reshape(27,1) - rg * w2
    dw2_t = np.sqrt(np.square(dw2_t) +  np.square(dw2))
    m2 = rd * m2 - r * (dw2/dw2_t)

    dw3= np.sum(-x_t3*(e),axis =0).reshape(18,1) - rg * w3
    dw3_t = np.sqrt(np.square(dw3_t) + np.square(dw3))
    m3 = rd * m3 - r * (dw3/dw3_t)
    
    #dw4= np.sum(-x_t4*(e),axis =0).reshape(9,1) - rg * w4
    #dw4_t = np.sqrt(np.square(dw4_t) + np.square(dw4))
    #m4 = rd * m4 - r * (dw4/dw4_t)

    #dw5= np.sum(-x_t5*(e),axis =0).reshape(9,1) - rg * w5
    #dw5_t = np.sqrt(np.square(dw5_t) + np.square(dw5))
    #m5 = rd * m5 - r * (dw5/dw5_t)
    '''
    db = np.sum((e),axis = 0) - rg *b
    db_t = np.sqrt(np.square(db_t) + np.square(db))
    m0 = rd * m0 - r * (db/db_t)

    y  = b + x_v.dot(w) #+ x_v2.dot(w2)# + x_v3.dot(w3)# + x_v4.dot(w4)# + x_v5.dot(w5)
    c  = np.sum(np.square((y_v- y)),axis = 0)/652

    w  = w + m1
#    w2 = w2 + m2
    #w3 = w3 + m3
    #w4 = w4 + m4
    #w5 = w5 + m5
    b  = b + m0
    '''
    if i%1000 == 0:
      #fp.write('training cost, %f' %c)
    if i%1000 == 0:
      #fp.write(', valid cost, %f\n' %c)
    if i%3000 == 0:
      #print '%d%%'%(i/3000)
    '''
  return [w,b]

def test(w,b,nnn):
  x_te = np.zeros((240,9),dtype = np.float)
  with open('./data/test.csv') as fp:
    id_ = 0
    for line in fp:
      vector = line.strip().split(',')
      if vector[1] == 'PM2.5':
        x_te[id_,:9] = vector[2:]
        id_ += 1
#      elif vector[1] == 'AMB_TEMP':
#        x_te[id_,9:18] = vector[2:]
#      elif vector[1] == 'PM10':
#        x_te[id_,18:] = vector[2:]
  
  y_te = b + x_te.dot(w) #+ np.square(x_te).dot(w2)# + np.power(x_te,3).dot(w3)# + np.power(x_te,4).dot(w4) + np.power(x_te,5).dot(w5)
  with open('./lamda2/test9allf'+nnn+'.csv','w') as fp:
    fp.write('id,value\n')
    for i in range(240):
      fp.write('id_' + str(i) + ',' + str(y_te[i,0]) + '\n')

[x,y_] = gendata()
#[w,b] = linearRegression(x,y_,0.1)
w  = 1e-13*np.random.random((9,1))
num_i=10000
[w,b] = linearRegression(x,y_,num_i,0.1,w)
f = open("./lamda2/w_lamda01.dat", "wb")
pickle.dump(w,f)
f.close()
f2 = open("./lamda2/b_lamda01.dat", "wb")
pickle.dump(b,f2)
f2.close()
test(w,b,'01')

[w,b] = linearRegression(x,y_,num_i,0.01,w)
f = open("./lamda2/w_lamda001.dat", "wb")
pickle.dump(w,f)
f.close()
f2 = open("./lamda2/b_lamda001.dat", "wb")
pickle.dump(b,f2)
f2.close()
test(w,b,'001')

[w,b] = linearRegression(x,y_,num_i,0.001,w)
f = open("./lamda2/w_lamda0001.dat", "wb")
pickle.dump(w,f)
f.close()
f2 = open("./lamda2/b_lamda0001.dat", "wb")
pickle.dump(b,f2)
f2.close()
test(w,b,'0001')

[w,b] = linearRegression(x,y_,num_i,0.0001,w)
f = open("./lamda2/w_lamda00001.dat", "wb")
pickle.dump(w,f)
f.close()
f2 = open("./lamda2/b_lamda00001.dat", "wb")
pickle.dump(b,f2)
f2.close()
test(w,b,'00001')
'''
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
fig.savefig('./lamda2/9h_lamda.png')   # save the figure to file
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
fig.savefig('./lamda2/9h_lamda2.png')
test(w,b)
'''

