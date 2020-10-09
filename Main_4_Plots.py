"""
This file contains the code to make the plots for the data obtined from Main4
we simply have to alter the indices being used in the loops in order to generate all different 
plots used in this project
"""
import numpy as np
import scipy as sc
import math as math
import matplotlib.pyplot as plt
import scipy.sparse.linalg as linalg
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds
import pickle



trials = 1000
max_i = 100
stepMethods = ['harmonic', 'binary', 'backsearch']
batchSizes = [64, 128, 256]
combos = [[a,b] for a in range(3) for b in range(3)]

xs = np.zeros((3, 3, trials, 4))
f_recs = np.zeros((3, 3, trials, max_i+1))
norm_recs = np.zeros((3,3, trials, max_i))
ks = np.zeros((3,3, trials))
ts = np.zeros((3,3, trials))

x0 = np.array([0,0,0,0])


pickle_in = open('res4_1.p',"rb")
example_dict = pickle.load(pickle_in)


frecs = example_dict["f"]




ts = example_dict["t"]

o = np.zeros(1)

fig = plt.figure()

for i in range (3):
    for j in range (3):
      lab = 'bsz_g = '+str(batchSizes[i]) + 'bsz_h = '+str(1+j)+ 'bsz_g'
      t = ts[i,j,:]
      t = np.append(o,t, axis = 0)
      plt.plot(t, frecs[i,j,:], label = lab) 
      plt.legend()
plt.xlabel('time')
plt.ylabel('function value')


pickle_in = open('res4_2_final.p',"rb")
example_dict = pickle.load(pickle_in)


frecs = example_dict["f"]




ts = example_dict["t"]

o = np.zeros(1)

fig1 = plt.figure()

for i in range (3):
    for j in range (3):
        lab = stepMethods[i] +' '+str(batchSizes[j])
        t = ts[i,j,:]
        t = np.append(o,t, axis = 0)
        plt.plot(t, frecs[i,j,:], label = lab) 
        plt.legend()
plt.xlabel('time')
plt.ylabel('function value')


pickle_in = open('res4_3.p',"rb")
example_dict = pickle.load(pickle_in)


frecs = example_dict["f"]




ts = example_dict["t"]

o = np.zeros(1)

fig2 = plt.figure()

d = np.array([1,5,10,20])
for j in range (4):
    lab ='M = ' +str(d[j])
    t = ts[j,:]
    t = np.append(o,t, axis = 0)
    plt.plot(t, frecs[j,:], label = lab) 
    plt.legend()
plt.xlabel('time')
plt.ylabel('function value')

plt.show()
