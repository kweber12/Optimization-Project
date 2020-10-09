"""
This file runs and stores the data for part 2
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

#Implimenting the Stochastic Gradient Dissent Method 

#first we take in the needed data
(total_dat, scales) = f.get_data('all',12,True)

# get the objective function and related function from project methods
function_1 = f.fun
function_1g = f.grad
function_1h = f.hessv
trials = 100
max_i = 1000
stepMethods = ['harmonic', 'binary', 'backsearch']
batchSizes = [64, 128, 256]
combos = [[a,b] for a in range(3) for b in range(3)]

xs = np.zeros((3, 3, trials, 4))
f_recs = np.zeros((3, 3, trials, max_i+1))
norm_recs = np.zeros((3,3, trials, max_i))
ks = np.zeros((3,3, trials))
ts = np.zeros((3,3, trials,max_i+1))

x0 = np.array([0,0,0,0])

for c in combos:
    for i in range(trials):
        (x, f_rec, norm_grad_rec,k,t) = f.StochGradDescent(function_1,total_dat,function_1g, x0, stepMethods[c[0]], batchSizes[c[1]], max_i)
        print(c)
        print(i)
        print(x)
        xs[c[0],c[1],i]=x
        f_recs[c[0],c[1],i] = f_rec
        norm_recs[c[0],c[1],i] = norm_grad_rec
        ks[c[0],c[1],i] = k
        ts[c[0],c[1],i] = t
       
xs_out = np.average(xs, axis=2)
f_recs_out = np.average(f_recs, axis=2)
norm_recs_out = np.average(norm_recs, axis=2)
ks_out = np.average(ks, axis=2)
ts_out = np.average(ts, axis=2)
frecs=f_recs_out[2,2,:]
t = ts_out[2,2,:]
plt.plot(t, frecs)
plt.show()
"""
#itters = np.arange(1,99)

plt.plot(frecs)
plt.show()

example_dict = {"xs":xs,"f_recs":f_recs,"norm_recs_out":norm_recs_out, "ks_out":ks_out,"ts_out":ts_out}
pickle_out = open("dict.pickle","wb")
pickle.dump(example_dict,pickle_out)
"""
