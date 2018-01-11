# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 15:27:04 2018

@author: carl
"""
from skimage import io 
import numpy as np
import scipy.misc
import sys
file_name =sys.argv[1] #'Aberdeen/'
k = 415
img_file = sys.argv[2]
img_ = io.imread(file_name+img_file)
num = 4
X = np.zeros((1080000,k))
for i in range(k):
    img = io.imread(file_name+str(i)+'.jpg')
    X[:,i] = img.flatten()
X_mean = np.mean(X,axis=1)
X_nor = (X.T-X_mean.T).T
U, s, V = np.linalg.svd(X_nor, full_matrices=False)
U_dot = np.dot(img_.flatten()-X_mean,U[:,:num])
eigen_var = np.sum(U[:,:num]*U_dot[:num],axis=1)
eigen_face = eigen_var+X_mean
eigen_face -= np.min(eigen_face)
eigen_face /= np.max(eigen_face)
eigen_face = ((eigen_face)*255).astype(np.uint8)
scipy.misc.toimage(eigen_face.reshape(600,600,3)).save('reconstruction.jpg')