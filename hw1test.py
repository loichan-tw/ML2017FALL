import numpy as np

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
      elif vector[2] == 'AMB_TEMP':
        for v in vector[3:]:
          temp_list.append(float(v))
      elif vector[2] == 'PM10':
        for v in vector[3:]:
          pm10_list.append(float(v))
  for i in range(5760):
    if pm25_list[i] == -1:
      pm25_list[i] = tmp
    else:
      tmp = pm25_list[i]
  x  = np.zeros((5652,27),dtype = np.float)
  y_ = np.zeros((5652,1),dtype = np.float)
  for m in range(12):
    for d in range(471):
      x[471*m+d,:9] = pm25_list[m*480+d:m*480+d+9]
      x[471*m+d,9:18] = temp_list[m*480+d:m*480+d+9]
      x[471*m+d,18:27] = pm10_list[m*480+d:m*480+d+9]
      y_[471*m+d,0] = pm25_list[m*480+d+9]
  return [x,y_]

def linearRegression(x,y_):
  [x_t,x_v] = np.array_split(x,[5000],axis = 0)
  [y_t,y_v] = np.array_split(y_,[5000],axis = 0)
  w  = 1e-13*np.random.random((27,1))
  dw = np.zeros((27,1),dtype = np.float)
  dw_t = np.zeros((27,1),dtype = np.float)
  m1 = np.zeros((27,1),dtype = np.float)

  w2 = 1e-13*np.random.random((27,1))
  dw2= np.zeros((27,1),dtype = np.float)
  dw2_t = np.zeros((27,1),dtype = np.float)
  m2 = np.zeros((27,1),dtype = np.float)
  
  b  = 1e-13*np.random.random((1,1))
  db = np.zeros((1,1),dtype = np.float)
  db_t = np.zeros((1,1),dtype = np.float)
  m0 = np.zeros((1,1),dtype = np.float)

  r = 0.01 
  rd = 0
  rg = 2.0
  x_t2 = np.square(x_t)

  x_v2 = np.square(x_v)
  
  fp = open ('./data/record.csv','w')
  fp.write('learning rate, %f ,ramda, %f\n' %(r,rg))
  for i in range(300000):
    y  = b + x_t.dot(w) + x_t2.dot(w2)# + x_t3.dot(w3)# + x_t4.dot(w4)# + x_t5.dot(w5)
    e  = y_t - y
    c  = np.sum(np.square(e),axis = 0)/5000

    dw = np.sum(-x_t*(e),axis = 0).reshape(27,1) - rg * w
    dw_t = np.sqrt(np.square(dw_t) +  np.square(dw))
    m1 = rd * m1 - r * (dw/dw_t)

    dw2= np.sum(-x_t2*(e),axis =0).reshape(27,1) - rg * w2
    dw2_t = np.sqrt(np.square(dw2_t) +  np.square(dw2))
    m2 = rd * m2 - r * (dw2/dw2_t)

    db = np.sum((e),axis = 0) - rg *b
    db_t = np.sqrt(np.square(db_t) + np.square(db))
    m0 = rd * m0 - r * (db/db_t)

    y  = b + x_v.dot(w) + x_v2.dot(w2)# + x_v3.dot(w3)# + x_v4.dot(w4)# + x_v5.dot(w5)
    c  = np.sum(np.square((y_v- y)),axis = 0)/652

    w  = w + m1
    w2 = w2 + m2
    b  = b + m0
  return [w,w2,b]

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
  with open('./linear_regression.csv','w') as fp:
    fp.write('id,value\n')
    for i in range(240):
      fp.write('id_' + str(i) + ',' + str(y_te[i,0]) + '\n')

[x,y_] = gendata()
[w,w2,b] = linearRegression(x,y_)
test(w,w2,b)


