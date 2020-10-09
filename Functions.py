"""
This file stores all the functions we need for the project
"""
import pandas
import numpy as np
import scipy.sparse.linalg as linalg
import time

# Define all the optimization methods needed for project

def SINewton(fun, grad, hess, dat, x0,max_iter):
    # fun, grad  and hess must each accept a vector x and data as input. Fun returns a real number, grad returns a vector, hess returns a matrix
    t0 = time.time()
    gam = 0.9
    jmax = int(np.ceil(np.log(10**-14)/np.log(gam)))
    eta = 0.5
    CGimax = 20 
    n = len(dat[:,0])
    max_batch=64
    bsz = min(n,max_batch)
    t = np.zeros(max_iter+1)
    
    
    f_rec = np.zeros(max_iter+1)
    f_rec[0]=fun(x0, dat)
    
    norm_grad_rec = np.zeros(max_iter)
    nfail = 0
    nfail_max = 5*np.ceil(n/bsz)

    x = x0
    for k in range(max_iter):
        ig = np.random.permutation(n)[0:bsz]
        ih = np.random.permutation(n)[0:bsz]
        datg = dat[ig,:]
        dath = dat[ih,:]
        
        hv = lambda v: hessv(v,x,dath)        
        app_hess = linalg.LinearOperator(shape=(len(x0),len(x0)), matvec = hv)
        app_grad = grad(x, datg)
        norm_grad_rec[k]=np.linalg.norm(app_grad)
        step = linalg.cg(app_hess, -1*app_grad, maxiter = CGimax)[0]
        a=1
        f0 = fun(x, datg)
        junk = eta*np.dot(app_grad,step)
        
        attempt = 0
        for j in range(jmax):
            attempt = j+1
            xtry = x+a*step
            f1 = fun(xtry, datg)
            if f1<f0+a*junk:
                break
            else:
                a=a*gam
        if attempt<jmax:
            x = xtry
        else:
            nfail=nfail+1
        
        f_rec[k+1]=fun(x,dat)
        t[k+1] = time.time()
        
                
        if nfail==nfail_max:            
            f_rec[k+2:]=None
            norm_grad_rec[k+1:]=None
            break
    

  
    return (x, f_rec, norm_grad_rec, k, t)

def StochGradDescent(fun, dat, grad, x0, stepMethod, bsz, max_i):
    t0 = time.time()
    n = len(dat[:,0])
    max_bsz = 64
    t = np.zeros(max_i+1)
    
    x = x0
    
    f_rec = np.zeros(max_i+1)
    f_rec[0]=fun(x0, dat)
    
    norm_grad_rec = np.zeros(max_i)
    
    convergence_tol = 10**-10
    
    for k in range(max_i):
        inds = np.random.permutation(np.arange(n))[:bsz]
        subdat=dat[inds,:]
        g = grad(x, subdat)
        a = SGStepsize(x,-g, g, fun, k, stepMethod, subdat)
        x = x-a*g
        
        f_rec[k+1] = fun(x,dat)
        norm_grad_rec[k] = np.linalg.norm(g)
        t[k+1] = time.time()
        
        
        if k>=4:
            m = np.max(norm_grad_rec[k-4:k])
            if m<convergence_tol:
                break
    
  

    return (x, f_rec, norm_grad_rec, k, t)
    

def SGStepsize(x, d, g, fun, k, stepMethod, dat ):
    if stepMethod =='backsearch':
        a = 1
        r = 0.9
        e = np.dot(d,g)
        f0 = fun(x,dat)
        for j in range(300):
            xtry = x+a*d
            f1 = fun(xtry, dat)
            if f1<f0+0.5*a*e:
                break
            else:
                a=a*r
        return a 
    elif stepMethod =='harmonic':
        return 1/(k+1)
    elif stepMethod =='binary':
        p = int(np.ceil(np.log2(k+1)))
        if p==0: return 1
        else: return 2**(-p+1)
    else: return 1        
            


# set up objective function and related functions   
def fun(x, dat):
    lam = 0.01
    d0 = len(dat[:,0])
    aux = -1*np.dot(dat,x)
    return np.sum(np.log(1+np.exp(aux)))/d0 + lam*np.dot(x,x)/2

def grad(x, dat):
    lam = 0.01
    d0 = len(dat[:,0])
    d1 = len(dat[0,:])
    aux = np.exp(-1*np.dot(dat,x))
    mat = np.outer(aux/(1+aux),np.ones(d1))
    return np.sum(-1*dat*mat,0)/d0 + lam*x

def hessv(v, x, dat):
    lam = 1/100    
    d1 = len(dat[0,:])
    aux = np.exp(-1*np.dot(dat,x))
    vec = (aux * np.dot(dat,v))/np.square(1+aux)
    mat = dat * np.outer(vec,np.ones(d1))
    return np.sum(mat,0) + lam*v


# set up data
def get_data(dset, year, rescale):
    # Read in the data for the correct year
    if year==16:
        raw = pandas.read_csv("C:\\Users\\Kevin\\Documents\\Machine_Learning\\Project1\\A2016.csv").values
    else:
        raw = pandas.read_csv("C:\\Users\\Kevin\\Documents\\Machine_Learning\\Project1\\A2012.csv").values
        
    # blabel is True if dem victory, False if repub victory
    # nlabel is 1 or -1
    blabels = np.greater(raw[:,3],raw[:,4])
    nlabels = 2*blabels.astype(int)-1 
    
    # pick the indices for the desired dataset
    if dset=='cal':
        indices = np.where( raw[:,2] ==' CA')[0]
    elif dset=='cal+':
        indices_ca = np.where( raw[:,2] ==' CA')[0]
        indices_or = np.where( raw[:,2] ==' OR')[0]
        indices_wa = np.where( raw[:,2] ==' WA')[0]
        indices = np.append(indices_ca,np.append(indices_or,indices_wa))
    else:
        dem_indices = np.where(blabels)[0]
        l = len(dem_indices)
        rep_indices = np.where(np.logical_not(blabels))[0]
        rep_indices = np.random.permutation(rep_indices)[0:l]
        indices = np.append(dem_indices,rep_indices)
        
    
    # take only the desired data from the raw
    raw = raw[indices]
    n = len(indices)
    
    blabels = blabels[indices]
    nlabels = nlabels[indices]
       

    votes = raw[:,3]+raw[:,4]
    log_votes = np.log( votes.astype(float) )
    vote_scale = np.max( log_votes )
    if rescale:
        log_votes = log_votes/vote_scale
    
    income = raw[:,5].astype(float)
    income_scale = np.max(income)
    if rescale:
        income = income/income_scale
    
    education = raw[:,9].astype(float)
    education_scale = np.max(education)
    if rescale:
        education = education/education_scale
    
    migra = raw[:,6].astype(float)
    migra_scale = np.max(np.abs(migra))
    if rescale:
        migra = migra/migra_scale
        

    data = np.transpose([ income, education, migra, np.ones(n) ])    
    for i in range(n):
        data[i] = data[i]*nlabels[i]   
        
    scales = {'Migration' : migra_scale, 'Income' : income_scale, 'Education' : education_scale}
    
    

    
    return (data,scales)

def Find_direction(g,s,y,k):
    m = np.size(s,1)
    a = np.zeros(m) 
    rho = np.zeros(m)
    if (k<m):
          q = s[(0,k),:]
          u = y[(0,k),:]
          m = k
           
    
          for i in range (m):
       
               rho[i] = 1/(np.dot(q[i,:],u[i,:]))
               l = np.dot(s[i,:],g)
    
               a[i] = np.dot(rho[i],l)
               g = g - a[i]*y[i,:]
               gam = np.dot(s[0,:],y[0,:])/np.dot(y[0,:],y[0,:]) # H0 = gam*eye(dim)
               g = g*gam
               for i in range (m-1):
                   aux = rho[i]*np.dot(y[i,:],g)
                   g = g + (a[i] - aux)*s[i,:]
     
          p = -g
    else:
         for i in range (m):
             q = s
             u = y
       
             rho[i] = 1/(np.dot(q[i,:],u[i,:]))
             l = np.dot(s[i,:],g)
    
             a[i] = np.dot(rho[i],l)
             g = g - a[i]*y[i,:]
             gam = np.dot(s[0,:],y[0,:])/np.dot(y[0,:],y[0,:]) # H0 = gam*eye(dim)
             g = g*gam
             for i in range (m-1):
                 aux = rho[i]*np.dot(y[i,:],g)
                 g = g + (a[i] - aux)*s[i,:]
     
         p = -g
    
    return(p)

def SLBFGS(fun, grad, hess, dat, x0, stepMethod, find_direction, max_itter):
    m = 5
    s = np.zeros((5,4))
    y = np.zeros((5,4))
    n = len(dat[:,0])
    max_batch=64
    bsz = min(n,max_batch)
    Ng = np.random.permutation(n)[0:bsz]
    Nh = np.random.permutation(n)[0:bsz]
    M=5
    x = x0
    t = np.array(max_itter)
    inds = np.random.permutation(np.arange(n))[:bsz]
    subdat=dat[inds,:]
    g = grad(x, subdat)
    s[0,:] = x
    y[0,:] = g
        
    
    f_rec = np.zeros(max_itter+1)
    f_rec[0]=fun(x0, dat)
    
    norm_grad_rec = np.zeros(max_itter)
    
    for k in range (max_itter):
        inds = np.random.permutation(np.arange(n))[:bsz]
        subdat=dat[inds,:]
        g = grad(x, subdat)
        

        p = find_direction(g,s,y,k)
        a = SGStepsize(x, g, fun, k, stepMethod, subdat)
        x = x+a*p
     
        
        if (k % M == 0 or k < m):
            np.roll(s,1, axis = 0)
            s[1,:] = x - s[1,:]
            r =  grad(x, subdat)
            np.roll(s,1, axis = 0)
            y = r-y[1,:]
            
            
        f_rec[k+1] = fun(x,dat)
        norm_grad_rec[k] = np.linalg.norm(g)
        t[k] = time.time()
        
        return(x,f_rec, norm_grad_rec, t)
        
        
        
    
    
    
