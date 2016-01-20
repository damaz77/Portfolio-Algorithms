
import mosek
import time
import numpy as np


def mosek_res (n, b, Q):
    env = mosek.Env ()
    task = env.Task()
    
    numvar = n
    numcon = 1

    task.appendcons(numcon)
    task.appendvars(numvar)
    
    bkx = [mosek.boundkey.lo] * n
    
    blx = [0] * n
    bux = [np.infty] * n
    
    for j in range(numvar):
        task.putbound(mosek.accmode.var,j,bkx[j],blx[j],bux[j])
   


     
    indexes = np.tril_indices(numvar)
    qsubi = indexes[0]
    qsubj = indexes[1]
    qval = np.tril(Q)[indexes]
    task.putqobj(qsubi,qsubj,qval)
    
    opro = [mosek.scopr.log] * n
    oprjo = np.array([i for i in range(n)])
    oprfo = -b
    oprgo = np.ones(n)
    oprho = np.zeros(n)
    
    task.putSCeval(opro, oprjo, oprfo, oprgo, oprho)
    
    task.putobjsense(mosek.objsense.minimize)
    
    init_time = time.clock()
    

    task.optimize()
    xx = np.zeros(numvar, float)
    task.getxx(mosek.soltype.itr,xx)

    xx = xx/sum(xx)
    
    time_mosek = time.clock()-init_time
    
    return time_mosek,xx

