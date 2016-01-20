import numpy as np

c = 0
lamb = 0
global_aux = 0
gamma = []
a = []
b = []

def computeOptStep(x,i,j,covMatrix,theta, sigma_dot_x, bounds):
    
    A = computeA(i,j,covMatrix)
    B = computeB(x,i,j,covMatrix, sigma_dot_x)
    C = computeC(x,i,j,covMatrix,theta)
    D = computeD(x,i,j,covMatrix,theta, sigma_dot_x)
    E = computeE(x,i,j,covMatrix,theta)

    coeff = [4*A,3*B,2*C,D]
    
    roots = np.roots(coeff)  
    #mettere a posto i bound
    roots = np.append(roots,bounds[0])

    min_fun = compute_function(x[j], A, B, C, D, E)
    best_root = x[j]

    for k in range(len(roots)):
        root = roots[k]
        if not isinstance(root, complex):
            #bound check
            if (root>=bounds[0]) and (root <= x[j]):
                act_fun = compute_function(root, A, B, C, D, E)
                if(act_fun<min_fun):
                    min_fun = act_fun
                    best_root = root  
        else:
            if (root.imag == 0.0):
                if(root.real>=bounds[0]) and (root.real <= x[j]):
                    act_fun = compute_function(root.real, A, B, C, D, E)
                    if(act_fun<min_fun):
                        min_fun = act_fun
                        best_root = root.real                  
    return x[j]- best_root
        
def computeA(i,j,covMatrix):
    left = (covMatrix[i,i] - covMatrix[i,j])**2
    right = (covMatrix[j,j] - covMatrix[i,j])**2
    return left+right

def computeB(x,i,j,covMatrix, sigma_dot_x):  
    global gamma, c, global_aux
    gamma = sigma_dot_x - covMatrix[i,:]*x[i] - covMatrix[j,:]*x[j]
    c = x[i] + x[j]
    global_aux = (-2*covMatrix[i,i]*c + covMatrix[i,j]*c - gamma[i])
    left = 2*(covMatrix[i,i]-covMatrix[i,j])*global_aux
    right = 2*(covMatrix[j,j] - covMatrix[i,j])*(covMatrix[i,j]*c + gamma[j])
    return left+right

def computeC(x,i,j,covMatrix,theta):
    global lamb
    lamb = covMatrix[i,i]*c*c + c*gamma[i] - theta
    left_sum = compute_left_sum_C(x,covMatrix,i,j)
    left = global_aux**2
    center = 2*lamb*(covMatrix[i,i]-covMatrix[i,j])
    right = (covMatrix[i,j]*c + gamma[j])**2 - 2*theta*(covMatrix[j,j]-covMatrix[i,j])
    return left_sum + left + center + right

def compute_left_sum_C(x,covMatrix,i,j):
    global a
    a = x * (covMatrix[:,j] - covMatrix[:,i]) 
    a[i] = 0
    a[j] = 0
    return np.sum(a*a)
        
def computeD(x,i,j,covMatrix,theta, sigma_dot_x):
    left_sum = compute_left_sum_D(x, covMatrix, i, j, theta)
    center = 2*lamb*global_aux
    right = -2*theta*(covMatrix[i,j]*c + gamma[j])
    return left_sum + center + right

def compute_left_sum_D(x, covMatrix, i, j, theta):
    global b
    b = x*(gamma + covMatrix[:,i]*c) - theta
    b[i] = 0
    b[j] = 0
    return np.sum(2*a*b)

def computeE(x,i,j,covMatrix,theta):
    left_sum = compute_left_sum_E()
    return left_sum + lamb*lamb + theta*theta

def compute_left_sum_E():
    return np.sum(b*b)

def compute_function(root,A,B,C,D,E):
    return A*np.power(root,4) + B*np.power(root,3) + C*np.power(root,2) + D*root + E
    
    
  
