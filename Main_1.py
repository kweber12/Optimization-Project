"""
This file runs and stores the data for part 1
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


(cal_dat,scales) = f.get_data('cal', 12, True)
n = len(cal_dat[:,0])
d = 3
labels = cal_dat[:,3]



X0 = np.random.random(4)*2-1


# get the objective function and related function from project methods
function_1 = f.fun
function_1g = f.grad
function_1h = f.hessv

# use stochastic inexact newton's for initial guess
(result, f_rec, normgrad, k, t) = f.SINewton(function_1, function_1g, function_1h,cal_dat, X0,1000)
print(result)

# get expanded data set for pacific northwest
(pnw_dat, junk) = f.get_data('cal+', 12, False)
n2 = len(pnw_dat[:,0])
pnw_labels = pnw_dat[:,3]

#get Newton Plane for PNW
(result_pnw, f_rec_pnw, normgrad_pnw, k_pnw, t_pnw) = f.SINewton(function_1, function_1g, function_1h,cal_dat, X0, 1000)


# set up soft margin variables for consrained optimization
xi = np.array([max(0,1-np.dot(cal_dat[i,:],result)) for i in range(n)])+10**-15
x1 = np.append(result,xi)

#perform constrained minimization
hess = np.block([ [np.eye(d), np.zeros((d,n+1))] , [np.zeros((n+1,d)),np.zeros((n+1,n+1))]])
offset = np.block([np.zeros(d+1), np.ones(n)])

def fun(v):
    return np.dot(v,np.dot(hess,v))/2+1000*np.dot(offset,v)

def grad(v):
    return np.dot(hess,v)+1000*offset

mat = np.block([[cal_dat, np.eye(n)],[np.zeros((n,d+1)),np.eye(n)]])
lcon = np.append(np.ones(n),np.zeros(n))
ucon = np.inf*np.ones(2*n)
cons = LinearConstraint(mat, lcon, ucon, keep_feasible=True)
lb = np.append(-np.inf*np.ones(d+1),np.zeros(n))
ub = np.inf*np.ones(n+d+1)
bs = Bounds(lb,ub, keep_feasible=True)
out = minimize(fun, x1, method='SLSQP', jac=grad, constraints=cons, tol = 10**-16, bounds=bs)
x = out['x']

print('Constrained min '+str(x))

# test performance of the two planes
cal_test0 = np.greater(np.dot(cal_dat,result), np.zeros(n) ).astype(int)
cal_test0_result = np.sum(cal_test0)/n

cal_test1 = np.greater(np.dot(cal_dat,x[:4]), np.zeros(n) ).astype(int)
cal_test1_result = np.sum(cal_test1)/n

pnw_test0 = np.greater(np.dot(pnw_dat,result_pnw), np.zeros(n2) ).astype(int)
pnw_test0_result = np.sum(pnw_test0)/n2

pnw_test1 = np.greater(np.dot(pnw_dat,x[:4]), np.zeros(n2) ).astype(int)
pnw_test1_result = np.sum(pnw_test1)/n2

results = np.array([cal_test0_result, cal_test1_result, pnw_test0_result, pnw_test1_result])
print(results)

#Plotting planes from both methods as well as raw data 
a = np.linspace(0,1,100)
b = np.linspace(0,1,100)
a,b = np.meshgrid(a,b)


dem_indices = np.where(labels==1)[0]
rep_indices = np.where(labels==-1)[0]

dem_dat = cal_dat[dem_indices]
rep_dat = -1*cal_dat[rep_indices]


z = -1*(result[0]*a + result[1]*b + result[3])/(100*result[2])
z1 = -1*(x[0]*a + x[1]*b + x[3])/(10*x[2])

# plot cal_dat without plane
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter( dem_dat[:,0], dem_dat[:,1], dem_dat[:,2] )
ax.scatter( rep_dat[:,0], rep_dat[:,1], rep_dat[:,2] )
ax.set_xlabel('income')
ax.set_ylabel('Education')
ax.set_zlabel('migration rate')

#plot with Newton plane
fig1 = plt.figure()
ax1 = fig1.add_subplot(111,projection='3d')

ax1.scatter( dem_dat[:,0], dem_dat[:,1], dem_dat[:,2] )
ax1.scatter( rep_dat[:,0], rep_dat[:,1], rep_dat[:,2] )
ax1.plot_surface(a,b,z)
ax1.set_xlabel('income')
ax1.set_ylabel('Education')
ax1.set_zlabel('migration rate')


# plot cal_dat with Active Set plane
fig2 = plt.figure()
ax2 = fig2.add_subplot(111,projection='3d')
ax2.plot_surface(a,b,z1)

ax2.scatter( dem_dat[:,0], dem_dat[:,1], dem_dat[:,2] )
ax2.scatter( rep_dat[:,0], rep_dat[:,1], rep_dat[:,2] )
ax2.set_xlabel('Income')
ax2.set_ylabel('Education')
ax2.set_zlabel('Migration')

dem_indices = np.where(pnw_labels==1)[0]
rep_indices = np.where(pnw_labels==-1)[0]

dem_dat = pnw_dat[dem_indices]
rep_dat = -1*pnw_dat[rep_indices]

z = -1*(result_pnw[0]*a + result_pnw[1]*b + result_pnw[3])/(100*result_pnw[2])
z1 = -1*(x[0]*a + x[1]*b + x[3])/(10*x[2])

#plot pnw data
fig3 = plt.figure()
ax3 = fig1.add_subplot(111,projection='3d')
ax3.scatter(dem_dat[:,0]/scales['Income'], dem_dat[:,1]/scales['Education'], dem_dat[:,2]/scales['Migration'] )
ax3.scatter(rep_dat[:,0]/scales['Income'], rep_dat[:,1]/scales['Education'], rep_dat[:,2]/scales['Migration'] )
ax3.set_xlabel('Income')
ax3.set_ylabel('Education')
ax3.set_zlabel('Migration')

#plot with pnw Newton plane
fig4 = plt.figure()
ax4 = fig4.add_subplot(111,projection='3d')

ax4.scatter(dem_dat[:,0]/scales['Income'], dem_dat[:,1]/scales['Education'], dem_dat[:,2]/scales['Migration'] )
ax4.scatter(rep_dat[:,0]/scales['Income'], rep_dat[:,1]/scales['Education'], rep_dat[:,2]/scales['Migration'] )
ax4.plot_surface(a,b,z)
ax4.set_xlabel('income')
ax4.set_ylabel('Education')
ax4.set_zlabel('migration rate')


#plot with pnw Active Set plane
fig5 = plt.figure()
ax5 = fig5.add_subplot(111,projection='3d')
ax5.plot_surface(a,b,z1)

ax5.scatter(dem_dat[:,0]/scales['Income'], dem_dat[:,1]/scales['Education'], dem_dat[:,2]/scales['Migration'] )
ax5.scatter(rep_dat[:,0]/scales['Income'], rep_dat[:,1]/scales['Education'], rep_dat[:,2]/scales['Migration'] )
ax5.set_xlabel('Income')
ax5.set_ylabel('Education')
ax5.set_zlabel('Migration')

plt.show()
