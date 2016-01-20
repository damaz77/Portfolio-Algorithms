from knitro import *
import numpy
import time

b = []
covMatrix = []
theta = 0

def compute_theta(x):
    sigma_dot_x = x.dot(covMatrix)  
    old_A = x*sigma_dot_x/b
    return numpy.sum(old_A)/len(x)

def evaluateFC (x):
    global theta
    x_ = numpy.array(x)
    theta = compute_theta(x_)
    aux = covMatrix.dot(x_)
    A = x_*(aux)/b 
    B = A - theta
    return B.dot(B)
   
def evaluateGA(x, objGrad):
    x_ = numpy.array(x)
    n = len(x_)
    aux = x_.dot(covMatrix)/b
    D = x/b
    B = covMatrix*D
    C = 2*(x*aux - theta)
    acc = numpy.sum(C*B, 1)
    grad = acc + C*aux
    for i in range(n):
        objGrad[i]=grad[i]
        
def callbackEvalFC (evalRequestCode, n, m, nnzJ, nnzH, x, lambda_, obj, c, objGrad, jac, hessian, hessVector, userParams):
    if evalRequestCode == KTR_RC_EVALFC:
        obj[0] = evaluateFC(x)
        return 0
    else:
        return KTR_RC_CALLBACK_ERR
    
def callbackEvalGA (evalRequestCode, n, m, nnzJ, nnzH, x, lambda_, obj, c, objGrad, jac, hessian, hessVector, userParams):
    if evalRequestCode == KTR_RC_EVALGA:
        evaluateGA(x, objGrad)
        return 0
    else:
        return KTR_RC_CALLBACK_ERR    

def solve(xInit):
    n = len(xInit)
    objGoal = KTR_OBJGOAL_MINIMIZE
    objType = KTR_OBJTYPE_GENERAL
    bndsLo = [0]*n
    bndsUp = [1]*n
    jacIxConstr = [ ]
    jacIxVar    = [ ] 
    
    #---- SETUP AND RUN KNITRO TO SOLVE THE PROBLEM.
    
    #---- CREATE A NEW KNITRO SOLVER INSTANCE.
    kc = KTR_new()
    if KTR_set_func_callback(kc, callbackEvalFC):
        raise RuntimeError ("Error registering function callback.")
    if KTR_set_grad_callback(kc, callbackEvalGA):
        raise RuntimeError ("Error registering gradient callback.")
    KTR_set_int_param_by_name (kc, "hessopt", 6)
    KTR_set_int_param_by_name (kc, "outlev", 0)

    #---- INITIALIZE KNITRO WITH THE PROBLEM DEFINITION.
    KTR_init_problem (kc, n, objGoal, objType, bndsLo, bndsUp,
                                None, None, None,
                                jacIxVar, jacIxConstr,
                                None, None,
                                xInit, None)
   
    #---- SOLVE THE PROBLEM.
    #----
    #---- RETURN STATUS CODES ARE DEFINED IN "knitro.h" AND DESCRIBED
    #---- IN THE KNITRO MANUAL.
    x       = [0] * n
    lambda_ = [0] * n
    obj     = [0]
    
    KTR_solve (kc, x, lambda_, 0, obj,
                None, None, None, None, None, None)
    
    
#---- BE CERTAIN THE NATIVE OBJECT INSTANCE IS DESTROYED.
    sol = numpy.array(x)
    KTR_free(kc)
    return sol/numpy.sum(sol)
    

def standard_optimizer(x_0, covMatrix_, b_):
    global x, b, covMatrix
    x = x_0
    b = b_
    covMatrix = covMatrix_
    begin_time = time.clock()
    sol = solve(x)
    return time.clock()-begin_time, sol