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

pickle_in = open('res4_2_final.p',"rb")
example_dict = pickle.load(pickle_in)

pickle_in_1 = open('res2_new.p',"rb")
example_dict_1 = pickle.load(pickle_in_1)

pickle_in_2 = open('res3_new.p',"rb")
example_dict_2 = pickle.load(pickle_in_2)

frecs_1 = example_dict_1["f"]

frecs_2 = example_dict_2["f"]

frecs = example_dict["f"]



ts_1 = example_dict_1["t"]

ts_2 = example_dict_2["t"]

ts = example_dict["t"]

o = np.zeros(1)

t_f = np.append(o,ts[2,2,:], axis = 0)
t_f1 = np.append(o,ts_1[2,2,:], axis = 0)
t_f2 = np.append(o,ts_2[2,:], axis = 0)

fig = plt.figure()
fig.subplots_adjust(top=0.8)
ax = fig.add_subplot(111)
lab = 'SLBFGS binary'
plt.plot(t_f, frecs[1,2,:], label = lab) 
lab = 'StochasticGradientDescent backsearch'
plt.plot(t_f1, frecs_1[2,2,:], label = lab) 
lab = 'SINewton'
plt.plot(t_f2, frecs_2[0,:], label = lab) 
ax.set_xlabel('time')
ax.set_ylabel('function value')
plt.legend()
