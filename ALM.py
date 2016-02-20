import numpy
from scipy.optimize import minimize
import time

global_cov_matrix = []
theta = 0
x_const = []
y_const = []
mu_1 = 0
mu_2 = 0
n = 0


def compute_theta(x, y):
    global global_cov_matrix
    aux = x.dot(global_cov_matrix).dot(y)  
    return aux/len(x)

def compute_F(x,y):
    global theta,global_cov_matrix
    n = len(x)
    squared_sum = 0
    for i in range(n):
        M_i = numpy.zeros([n,n])
        M_i[i,:] = global_cov_matrix[i,:]
        aux = x.dot(M_i).dot(y)
        squared_sum = squared_sum + (aux - theta)*(aux-theta)
    return squared_sum

def compute_F_y_const(x):
    global y_const,theta,global_cov_matrix
    n = len(x)
    squared_sum = 0
    for i in range(n):
        M_i = numpy.zeros([n,n])
        M_i[i,:] = global_cov_matrix[i,:]
        aux = x.dot(M_i).dot(y_const)
        squared_sum = squared_sum + (aux - theta)*(aux-theta)
    return squared_sum

def compute_F_x_const(y):
    global x_const,theta,global_cov_matrix
    n = len(y)
    squared_sum = 0
    for i in range(n):
        M_i = numpy.zeros([n,n])
        M_i[i,:] = global_cov_matrix[i,:]
        aux = x_const.dot(M_i).dot(y)
        squared_sum = squared_sum + (aux - theta)*(aux-theta)
    return squared_sum
 
def compute_grad_1(x,y):
    global theta,global_cov_matrix
    n = len(x)
    sum_ = numpy.zeros(n)
    for i in range(n):
        M_i = numpy.zeros([n,n])
        M_i[i,:] = global_cov_matrix[i,:]
        dot_1 = M_i.dot(y)
        aux = x.dot(dot_1)
        sum_ = sum_ + (aux - theta)*dot_1
    return 2*sum_   
 
def compute_grad_1_x_const():
    global x_const,theta,global_cov_matrix
    n = len(x_const)
    sum_ = numpy.zeros(n)
    for i in range(n):
        M_i = numpy.zeros([n,n])
        M_i[i,:] = global_cov_matrix[i,:]
        dot_1 = M_i.dot(x_const)
        aux = x_const.dot(dot_1)
        sum_ = sum_ + (aux - theta)*dot_1
    return 2*sum_   

def compute_grad_2(x,y):
    global theta,global_cov_matrix
    n = len(x)
    sum_ = numpy.zeros(n)
    for i in range(n):
        M_i = numpy.zeros([n,n])
        M_i[i,:] = global_cov_matrix[i,:]
        dot_1 = M_i.dot(y)
        aux = x.dot(dot_1)
        sum_ = sum_ + (aux - theta)*M_i.transpose().dot(x)
    return 2*sum_  

def compute_grad_2_y_const():
    global y_const,theta,global_cov_matrix
    n = len(y_const)
    sum_ = numpy.zeros(n)
    for i in range(n):
        M_i = numpy.zeros([n,n])
        M_i[i,:] = global_cov_matrix[i,:]
        dot_1 = M_i.dot(y_const)
        aux = y_const.dot(dot_1)
        sum_ = sum_ + (aux - theta)*M_i.transpose().dot(y_const)
    return 2*sum_  


cons = ({'type': 'ineq','fun' : lambda x: x},
        {'type': 'eq','fun' : lambda x: numpy.array(1-numpy.sum(x))})

bnds = []

def f_to_minimize_1(x):
    global y_const,mu_1
    return compute_F_y_const(x) + compute_grad_2_y_const().dot(x-y_const) + 1/(2*mu_1)*numpy.power(numpy.linalg.norm(x-y_const),2)

def f_to_minimize_2(y):
    global x_const,mu_2
    return compute_F_x_const(y) + compute_grad_1_x_const().dot(y-x_const) + 1/(2*mu_2)*numpy.power(numpy.linalg.norm(x_const-y),2)

def compute_min_of_Q_1(x):
    return minimize(f_to_minimize_1, x, constraints=cons, bounds=bnds)

def compute_min_of_Q_2(y):
    return minimize(f_to_minimize_2, y, constraints=cons, bounds=bnds)

def compute_Q_1(x, y):
    global mu_1
    F = compute_F(x, y)
    grad_2 = compute_grad_2(y, y)
    penalization = 1/(2*mu_1)*numpy.power(numpy.linalg.norm(x-y),2)
    return F+grad_2.dot(x-y)+penalization

def compute_Q_2(x, y):
    global mu_2
    F = compute_F(x, y)
    grad_1 = compute_grad_1(x, x)
    penalization = 1/(2*mu_2)*numpy.power(numpy.linalg.norm(x-y),2)
    return F+grad_1.dot(y-x)+penalization


def alm_bktr(x_0, covMatrix, epsilon):
    global n,x_const,y_const,global_cov_matrix,theta, mu_1, mu_2
    begin_time = time.clock()
    global_cov_matrix = covMatrix
    n = len(x_0)
    x_k = x_0
    y_k = x_0
    mu_1 = 10
    beta = 0.9
    mu_2 = 10
    convergence = False
    k=0
    QP = 0
    conv_time = 0
    while (not convergence):
        theta = compute_theta(x_k, y_k)
        y_const = y_k
        x_k_plus_1 = numpy.array(compute_min_of_Q_1(x_k).x)
        QP = QP + 1
        
        #BACKTRACKING
        if compute_F(x_k_plus_1,x_k_plus_1) <= compute_Q_1(x_k_plus_1, y_k):
            mu_1 = mu_1/beta
        else:
            mu_1_found = False
            while (not mu_1_found):
                mu_1 = mu_1*beta
                x_k_plus_1 = numpy.array(compute_min_of_Q_1(x_k_plus_1).x)
                QP = QP + 1
                if compute_F(x_k_plus_1, x_k_plus_1) <= compute_Q_1(x_k_plus_1, y_k):
                    mu_1_found = True
                    mu_1 = mu_1/beta
        x_const = x_k_plus_1
        theta = compute_theta(x_k_plus_1, y_k)
        y_k_plus_1 = numpy.array(compute_min_of_Q_2(y_k).x)
        QP = QP + 1
        if compute_F(y_k_plus_1, y_k_plus_1) <= compute_Q_2(x_k_plus_1, y_k_plus_1):
            mu_2 = mu_2/beta
        else:
            mu_2_found = False
            while (not mu_2_found):
                mu_2 = mu_2*beta
                y_k_plus_1 = numpy.array(compute_min_of_Q_2(y_k_plus_1).x)
                QP = QP + 1
                if compute_F(y_k_plus_1, y_k_plus_1) <= compute_Q_2(x_k_plus_1, y_k_plus_1):
                    mu_2_found = True
                    mu_2 = mu_2/beta
        #fine BACKTRACKING  
        init_time = time.clock()
        if(numpy.linalg.norm(x_k_plus_1-x_k)<1e-07) and (numpy.linalg.norm(y_k_plus_1-y_k)<1e-07):
            convergence = True 
            conv_time = conv_time + time.clock()-init_time  
        else:    
            conv_time = conv_time + time.clock()-init_time

        x_k = x_k_plus_1
        y_k = y_k_plus_1
        k = k+1
    total_time = (time.clock()-begin_time)-conv_time
    print 'QP solved:',QP
    return total_time,y_k,k,compute_F(y_k, y_k)


