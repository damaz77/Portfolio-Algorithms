import numpy           as np
from   optimize.snopt7 import SNOPT_solver

covMatrix_ = []

def compute_theta(x):
    sigma_dot_x = x.dot(covMatrix_)  
    old_A = x*sigma_dot_x
    return np.sum(old_A)/len(x)

def objFG(status,x,needF,needG,cu,iu,ru):
    x_ = np.array(x)
    theta = compute_theta(x_)
    aux = covMatrix_.dot(x_)
    A = x_*(aux)
    B = A - theta
    #G = gradfunc(x_)
    fun = B.dot(B)
    return status, np.array([fun,np.sum(x_)])
  
# def gradfunc(x):
#     aux = x.dot(covMatrix_)
#     B = covMatrix_*x
#     C = 2*(x*aux - theta)
#     acc = np.sum(C*B, 1)
#     g_obj = acc + C*aux
#     return g_obj

def solve(x_0, covMatrix, bounds):
    global covMatrix_
    covMatrix_ = covMatrix
    inf   = 1.0e20
    n = len(x_0)
    
    snopt = SNOPT_solver()
    snopt.setOption("Verbose", False)
    snopt.setOption("Specs file","/home/federico/workspace/TestMosek/algorithms/specs.spec")
    
    xlow    = np.array([bounds[0]]*n)
    xupp    = np.array([bounds[1]]*n)
    
    Flow    = np.array([0.0, 1.0])
    Fupp    = np.array([inf, 1.0])
    
    ObjRow = 1
    A = np.array([ [0]*n,
                   [1]*n
                 ])
    

    G = np.array([ [1]*n,
                   [0]*n
               ])

    [exe_time, iterations] = snopt.snopta(x0=x_0,xlow=xlow,xupp=xupp,
                 Flow=Flow,Fupp=Fupp, ObjRow=ObjRow, A=A, G=G,
                 usrfun=objFG)
    
    return exe_time, snopt.x, iterations
