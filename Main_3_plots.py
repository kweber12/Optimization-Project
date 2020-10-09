"""
This file contains the code to make the plots for the data obtined from Main3
we simply have to alter the indices being used in the loops in order to generate all different 
plots used in this project
"""
import pandas as pn
import Functions as f
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
max_i = 300
stepMethods = ['harmonic', 'binary', 'backsearch']
batchSizes = [64, 128, 256]
combos = [[a,b] for a in range(3) for b in range(3)]

xs = np.zeros((3, 3, trials, 4))
f_recs = np.zeros((3, 3, trials, max_i+1))
norm_recs = np.zeros((3,3, trials, max_i))
ks = np.zeros((3,3, trials))
ts = np.zeros((3,3, trials))

x0 = np.array([0,0,0,0])


pickle_in = open('res3_new.p',"rb")
example_dict = pickle.load(pickle_in)




frecs = example_dict["f"]





ts = example_dict["t"]

o = np.zeros(1)


fig = plt.figure()
"""
for i in range (3):
    for j in range (3):
        lab = stepMethods[i] +' '+str(batchSizes[j])
        plt.scatter(ts[i,j], frecs[i,j], label = lab) 
        plt.legend()
"""

for j in range (3):
    lab = ' '+str(batchSizes[j])
    t = ts[j,:]
    t = np.append(o,t, axis = 0)
    plt.plot(t, frecs[j,:], label = lab) 
    plt.legend()
plt.xlabel('time')
plt.ylabel('function value')
"""
lab =str(batchSizes[0])
t = ts[0,:]
t = np.append(o,t, axis = 0)
plt.plot(t, frecs[0,:], label = lab) 
plt.legend()
"""
"""
for i in range (3):
      lab =str(batchSizes[i])
      t = ts[i,:]
      t = np.append(o,t, axis = 0)
      plt.plot(t, frecs[i,:], label = lab) 
      plt.legend()
"""

plt.show()