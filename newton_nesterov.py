import numpy
import sys
import time

eps =sys.float_info.epsilon

lambda_star = 0.95* ((3-numpy.sqrt(5))/2)

def is_an_RP_solution(n, b, x, corrMatrix, epsilon):
    RC = numpy.zeros(n)
    volatility = x.dot(corrMatrix).dot(x)
    violation_found = False
    i=0
    aux = corrMatrix.dot(x)
    while(not violation_found) and (i<n):
        RC[i]=x[i]*aux[i]
        if numpy.abs(RC[i]/volatility-b[i])>epsilon:
            violation_found= True
        i = i+1
    return not violation_found

def compute_f_x_lambda(corrMatrix, x, b):
    A = corrMatrix.dot(x)
    B = b/(x)
    diffVector = A-B
    return diffVector

def compute_jacobian(corrMatrix, x, b):
    D = numpy.diag(1/((x*x)))
    A = corrMatrix + b*D
    return A
    
def compute_step(F,J):
    inverse_J = numpy.linalg.inv(J)
    A = inverse_J.dot(F)
    return A

def compute_delta_k(x,step):
    B = step/(x)
    return numpy.max(B)

def compute_lambda_k(F,step):
    return numpy.sqrt(F.dot(step))

def damped_newton(n, x_0, b, epsilon, corrMatrix):
    begin_time = time.clock()
    b = b*(1/min(b))
    k=0
    convergence = False
    x_k = x_0
    conv_time = 0
    while (not convergence):
        k = k+1  
        F = compute_f_x_lambda(corrMatrix, x_k, b)
        J = compute_jacobian(corrMatrix, x_k, b)
        step = compute_step(F, J)
        lambda_k = compute_lambda_k(F, step)
        old_x = x_k
        if(lambda_k>lambda_star):
            delta_k = compute_delta_k(x_k, step)
            x_k = old_x -(1/(1+delta_k))*step
        else:
            x_k = old_x -step 
        init_time = time.clock()   
        if (is_an_RP_solution(n, b/numpy.sum(b), x_k/sum(x_k), corrMatrix, epsilon)):
            conv_time = conv_time + time.clock()-init_time
            convergence = True 
        else:
            conv_time = conv_time + time.clock()-init_time      
    x_k = x_k/sum(x_k) 
    total_time = (time.clock() - begin_time)-conv_time
    return total_time,x_k,k