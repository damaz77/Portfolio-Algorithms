
import numpy
import scipy
from mosek.fusion import *
import time
    
def generateCuts(t, x, b, n, epsilon):
    max_violation = 0
 
    for i in range(n):
        t_i = t[i]
        x_i = x[i]
        log_x = numpy.log(x_i)
        if(t_i > b[i]*log_x):
            if(t_i - b[i]*log_x > max_violation):
                max_violation = t_i - b[i]*log_x
                  
    if(abs(max_violation)<epsilon):
        return numpy.zeros(n), numpy.zeros(n)
    else:
        m = b/x 
        c = b*(- 1 + numpy.log(x))
        return m,c    
 

def fusion_mosek(n, b, Q):
    
    chol = scipy.linalg.cholesky(Q)
    V = DenseMatrix(chol)

    with Model("Log-Barrier") as M:
        x = M.variable("x", n, Domain.greaterThan(0.0))
        z = M.variable("z", 1, Domain.greaterThan(0.0))
        t = M.variable("t", n, Domain.lessThan(0.0))
        

        M.constraint(Expr.vstack( Expr.constTerm(1, 1.0),
                                  z.asExpr(),
                                  Expr.mul(V,x)),
                     Domain.inRotatedQCone())
        M.objective(ObjectiveSense.Minimize, Expr.sub(z, Expr.sum(t)))
        start_time = time.clock()
        
        M.solve()

        n_cut_added = 0
        convergence = False
        epsilon = 0.0000000001

        while (not convergence):       
            [m, c] = generateCuts(t.level(), x.level(), b, n, epsilon)
            if not numpy.equal(m.all(),numpy.zeros(n).all()):
                m_coeff = numpy.diag(m)
                M.constraint(Expr.sub(t, Expr.mul(m_coeff,x)), Domain.lessThan(c))          
                n_cut_added = n_cut_added + 1         
                M.solve()                
            else:
                convergence = True
        return time.clock()-start_time, x.level()/numpy.sum(x.level())
