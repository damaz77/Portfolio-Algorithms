import numpy
import time
import optStep


#the dot product between the covariance matrix and x
old_sigma_dot_x = []


old_B = []

#returns True if the RC vector is volatility/b for all i (with an error epsilon)
def is_an_RP_solution(n, x, covMatrix, epsilon):
    RC = numpy.zeros(n)
    volatility = x.dot(covMatrix).dot(x)
    violation_found = False
    i=0
    aux = covMatrix.dot(x)
    while(not violation_found) and (i<n):
        RC[i]=x[i]*aux[i]
        if numpy.abs(RC[i]/volatility-(1.0/n))>epsilon:
            violation_found= True
        i = i+1
    return not violation_found

#compute theta as the zero of the first derivative
def compute_theta(n, x, old_x, alpha, i_k, j_k,  k, covMatrix):
    global old_sigma_dot_x
    if(len(alpha)==0):
        #the first time, the dot product is computed
        old_sigma_dot_x = x.dot(covMatrix)  
    else:
        #then it is updated using the fact that only 2 components of x were modified since last iteration
        old_sigma_dot_x = old_sigma_dot_x + covMatrix[i_k,:]*(alpha[k-1]) - covMatrix[j_k,:]*(alpha[k-1])

    return numpy.sum(x*old_sigma_dot_x)/n
      
#compute the gradient of the objective function with respect to x   
#NOTE: this is equal to the output of compute_derivative_with_finite_differences
def compute_grad_F_x(x, covMatrix, theta, i_k, j_k, k):
    global old_B
    if(k==0):
        old_B = covMatrix*x
    else:
        old_B[:,i_k] = covMatrix[:,i_k]*x[i_k]
        old_B[:,j_k] = covMatrix[:,j_k]*x[j_k]
    
    C = 2*(x*old_sigma_dot_x - theta)
    acc = numpy.sum(C*old_B, 1)
    grad = acc + C*old_sigma_dot_x
    return grad


#compute the most violating pair
def compute_most_violating_pair(x, grad, bounds): 
    copy_grad_j = numpy.copy(grad)
    copy_grad_i = numpy.copy(grad)
    #avoid to select i_k, j_k too near to the bounds (a bit hardcoded)
    copy_grad_i[numpy.where(bounds[1] - x<=1e-10)] = 1e+10
    copy_grad_j[numpy.where(x - bounds[0]<=1e-10)] = -1e+10
    actual_i_k = numpy.argmin(copy_grad_i)
    actual_j_k = numpy.argmax(copy_grad_j) 
    return actual_i_k, actual_j_k

###################### ARMIJO #######################################

#evaluate the objective function in the point x + alpha*d
def compute_F_x_theta(x, alpha, d, i_k, j_k, covMatrix, theta):
    x_new = x + alpha*d
    if(alpha != 0):
        aux = old_sigma_dot_x + covMatrix[i_k,:]*(alpha) - covMatrix[j_k,:]*(alpha)
    else:
        aux = old_sigma_dot_x
    A = x_new*(aux)
    B = A - theta
    return B.dot(B)


#perform the armijo line search along the direction d 
def armijo(x, grad, d, i_k, j_k, covMatrix, theta, alpha, k, bounds):
    delta = 0.5
    if(k==0):
        actual_alpha = 1.0
    else:
        if(alpha[k-1]<1.0):
            actual_alpha = alpha[k-1]/(delta*delta)
        else:
            actual_alpha = 1.0
    gamma = 0.1
    
    #A = value of F in x + alpha*d
    A = compute_F_x_theta(x, actual_alpha , d, i_k[k], j_k[k], covMatrix, theta)
    
    #B = value of F in x
    B = compute_F_x_theta(x, 0, d, i_k[k], j_k[k], covMatrix, theta)
    C = gamma*actual_alpha*grad.dot(d)
    while(A > B + C):
        actual_alpha  = delta*actual_alpha 
        A = compute_F_x_theta(x, actual_alpha , d, i_k[k], j_k[k], covMatrix, theta)
        C = delta*C
    return numpy.min([actual_alpha, bounds[1] - x[i_k[k]], x[j_k[k]] - bounds[0]])

################# END ARMIJO ####################################


def gaussSeidel(n, x_0, epsilon, covMatrix, bounds, line_search_method):
    begin_time = time.clock()
    convergence = False
    x_k = numpy.copy(x_0)
    k=0
    
    #some statistics
    armijo_time = 0
    calc_time = 0
    theta_time = 0
    viol_time = 0
    conv_time = 0
    alpha = []
    i_k = []
    j_k = []
    old_x = 0
    max_iter = 50*n
    while(not convergence):
        ######## COMPUTE THETA ##########
        init_time = time.clock()
        if(k>0):
            theta = compute_theta(n, x_k, old_x, alpha, i_k[k-1], j_k[k-1], k, covMatrix)
        else:
            theta = compute_theta(n, x_k, old_x, alpha, 0, 0, k, covMatrix)
        old_x = numpy.copy(x_k)
        theta_time = theta_time + time.clock() - init_time
        
        ######## COMPUTE GRADIENT ##########
        init_time = time.clock()
        if(k>0):
            grad_F_x = compute_grad_F_x(x_k, covMatrix, theta, i_k[k-1], j_k[k-1], k)
        else:
            grad_F_x = compute_grad_F_x(x_k, covMatrix, theta, 0, 0, k)   
        calc_time = calc_time + time.clock()- init_time
        
        ######## BUILD DESCENT DIRECTION ##########
        init_time = time.clock()
        [actual_i_k, actual_j_k] = compute_most_violating_pair(x_k, grad_F_x, bounds)
        d = numpy.zeros(n)
        d[actual_i_k] = 1
        d[actual_j_k] = -1
        viol_time = viol_time + time.clock()- init_time
        #APPEND actual_i_k TO THE LIST OF i_k
        i_k.append(actual_i_k)
        #APPEND actual_j_k TO THE LIST OF j_k
        j_k.append(actual_j_k)
        init_time = time.clock()
        if line_search_method is "armijo":
            actual_alpha = armijo(x_k, grad_F_x, d, i_k, j_k, covMatrix, theta, alpha, k, bounds)
        else:    
            if line_search_method is "exact":
                actual_alpha = optStep.computeOptStep(x_k, actual_i_k, actual_j_k, covMatrix, theta, old_sigma_dot_x, bounds)
            else:
                print "Selected line search is not valid."
                break
        
        armijo_time = armijo_time + time.clock()-init_time
        alpha.append(actual_alpha)
        
        if(actual_alpha == 0):
            convergence = True
        
        #compute next x_k
        x_k = x_k + actual_alpha*d
        
        ########## STOP CRITERION ############
        init_time = time.clock()
        if (is_an_RP_solution(n, x_k, covMatrix, epsilon)):
            conv_time = conv_time + time.clock()-init_time
            convergence = True    
        else:
            conv_time = conv_time + time.clock()-init_time
            
        ########## NON-CONVERGENCE TEST #########
        if(k>max_iter):
            convergence = True
        k=k+1
    
    total_time = (time.clock()-begin_time)-conv_time
    print "Line search time:", armijo_time/total_time * 100,"%"
    print "Total time", total_time
    print "Iterations:", k
    return total_time,x_k,k, compute_F_x_theta(x_k, 0, d, i_k, j_k, covMatrix, theta)


        
    
