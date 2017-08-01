#!/usr/bin/env python
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

N=100
D=2
K=3
X=np.zeros((N*K,D))
y=np.zeros(N*K,dtype='uint8')

for j in range(K):
  ix=range(N*j,N*(j+1))
  r=np.linspace(0.0,1,N)
  t=np.linspace(j*4,(j+1)*4,N)+np.random.randn(N)*0.2
  X[ix]=np.c_[r*np.sin(t),r*np.cos(t)]
  y[ix]=j

plt.scatter(X[:,0],X[:,1],c=y,s=40,cmap=plt.cm.Spectral)
plt.show()  
