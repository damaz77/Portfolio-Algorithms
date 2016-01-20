import numpy
import time

old_sigma_dot_x = []
old_volatility = 0

def is_an_RP_solution(n, b, x, covMatrix, epsilon):
    RC = numpy.zeros(n)
    volatility = x.dot(covMatrix).dot(x)
    violation_found = False
    i=0
    aux = covMatrix.dot(x)
    while(not violation_found) and (i<n):
        RC[i]=x[i]*aux[i]
        if numpy.abs(RC[i]/volatility-b[i])>epsilon:
            violation_found= True
        i = i+1
    return not violation_found

def compute_volatility(old_x, input_x, i, covMatrix, sigma):
    global old_volatility
    if((old_volatility == 0) or (i==0)):
        old_volatility = numpy.sqrt(input_x.dot(covMatrix).dot(input_x))
        return old_volatility
    else:
        A = old_volatility * old_volatility
        B = 2*old_x[i-1]*covMatrix[i-1,:].dot(old_x)
        C = old_x[i-1] * old_x[i-1] * sigma[i-1]
        D = 2*input_x[i-1]*covMatrix[i-1,:].dot(input_x)
        E = input_x[i-1] * input_x[i-1] * sigma[i-1]
        old_volatility = numpy.sqrt(A-B+C+D-E)
        return old_volatility
  
                                                     
def compute_x_i_star(n, old_x, input_x, i, b, covMatrix, sigma):
    global old_sigma_dot_x
    if(len(old_sigma_dot_x)==0 or i==0):
        A = covMatrix.dot(input_x) 
    else:
        A = old_sigma_dot_x + covMatrix[i-1,:].dot(input_x[i-1]- old_x[i-1])
    old_sigma_dot_x = A
        
    B = input_x[i]*sigma[i]
    C = -A[i] + B
    D = -C*-C
    E = 4*sigma[i]*b[i]*compute_volatility(old_x, input_x, i, covMatrix, sigma)
    num = C + numpy.sqrt(D+E)
    den = 2*sigma[i]
    input_x[i] = num/den 
    if(i>0):
        old_x[i-1] = input_x[i-1] 
    return input_x[i]

         
def ccd(n, x_0, b , covMatrix, epsilon):
    begin_time = time.clock()
    convergence = False
    k=0
    x_k = numpy.copy(x_0)
    conv_time = 0
    sigma_square = numpy.diag(covMatrix)
    while (not convergence):
        old_x = numpy.zeros(n)
        for i in range(n):
            old_x[i] = x_k[i]
        input_x = numpy.zeros(n)
        for i in range(n):
            input_x[i] = x_k[i]
            
        for i in range(n):
            k = k+1
            x_k[i] = compute_x_i_star(n, old_x, input_x, i, b, covMatrix, sigma_square)    
        init_time = time.clock()
        if (is_an_RP_solution(n, b, x_k/sum(x_k), covMatrix, epsilon)):
            conv_time = conv_time + time.clock()-init_time
            convergence = True  
        else:
            conv_time = conv_time + time.clock()-init_time    
            
    x_star = x_k/sum(x_k)
    total_time = (time.clock()-begin_time)-conv_time
    return total_time,x_star,k
