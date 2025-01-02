import sys
import numpy as np
import pyqcstrc.qnclass.qnmath
import pyqcstrc.dode2.qnmath12

from numpy.typing import NDArray
#sys.path.append('.')
#from numericalc import coplanar_check_numeric_tau
#from pyqcstrc.qnclass.qnmath import mul
#from pyqcstrc.dode2.numericalc import coplanar_check_numeric_tau

class Qnmat:
    def __init__(self,vt: Qnmat,shape:np.array):
        #for i in range(shape):
        #    self.a[i]=a[i]
        self.vt=vt
        self.shape = shape  #dimension of a vector a
        
    def matmul(ma1: Qnmat, ma2: Qnmat):
        n1=ma1.ndim
        n2=ma2.ndim
        if(n1==1 & n2==1): # vectors
            sum=0
            for i in range(len(ma1)):
                sum=sum+ma1[i]*ma2[i]
            return sum
        elif(n1==2 & n2==1): # matrix*vector
            ma3=np.zeros(len(ma1))
            for i in range(len(ma1)):
                for j in range(len(ma1[0])):
                    ma3[i]=ma3[i][j]+ma1[i,j]*ma2[j]
            return ma3
        elif(n1==1 & n2==2): # vectpr*matrix
            ma3=np.zeros(len(ma1))
            for i in range(len(ma1)):
                for j in range(len(ma1[0])):
                    ma3[i]=ma3[i]+ma1[j]*ma2[j,i]
            return ma3
                
        
	def matrixpow(ma: Qnmat, n: int) -> Qnmat:
	    """
	    """
	    (mx,my)=ma.shape
	    if mx==my:
	        if n==0:
	            return np.identity(mx)
	        elif n<0:
	            tmp=np.identity(mx)
	            inva = np.linalg.inv(ma) # matrix inversion
	            for i in range(-n):
	                tmp=np.dot(tmp,inva)
	            return tmp
	        else:
	            tmp=np.identity(mx)
	            for i in range(n):
	                tmp=np.dot(tmp,ma)
	            return tmp
	    else:
	        print('matrix has not regular shape')
	        return 

	def det_matrix(mtx: Qnmat) -> Qnmat:
	    """Determinant of 3x3 matrix, mtx, in TAU style
	    
	    Parameters
	    ----------
	    mtx: array
	        3x3 matrix in SQRT3-style

	    Returns
	    -------
	    6d vectors projected onto Eperp in SQRT3-style.
	    """
	    
	    t3=mtx[0][0]*mtx[1][1]  #mul(mtx[0][0],mtx[1][1])
	    t1=t3*mtx[2][2]         #mul(t3,mtx[2][2])
	    #
	    t3=mtx[0][2]*mtx[1][0]  #mul(mtx[0][2],mtx[1][0])
	    t2=t3*c[1]              #mul(t3,c[1])
	    #
	    t1=t1+t2                #add(t1,t2)
	    
	    t3=mtx[0][1]*mtx[1][2]  #mul(mtx[0][1],mtx[1][2])
	    t3=t3*mtx[2][0]         #mul(t3,mtx[2][0])
	    #
	    t1=t1+t3                #add(t1,t3)
	    
	    t3=mtx[0][2]*mtx[1][1]  #mul(mtx[0][2],mtx[1][1])
	    t2=t3*mtx[2][0]         #mul(t3,mtx[2][0])
	    #
	    t1=t1-t2                #sub(t1,t2)
	    
	    t3=mtx[0][1]*mtx[1][0]  #mul(mtx[0][1],mtx[1][0])
	    t2=t3*mtx[2][2]         #mul(t3,mtx[2][2])
	    #
	    t1=t1-t2                #sub(t1,t2)
	    
	    t3=mtx[0][0]*mtx[1][2]  #mul(mtx[0][0],mtx[1][2])
	    t2=t3*mtx[2][1]         #mul(t3,mtx[2][1])
	    #
	    t1=t1-t2                #sub(t1,t2)
	    #
	    return t1
    
    def __matmul__(ma1:Qnmat, ma2:Qnmat) -> Qnmat:  #  for ma1@ma2
        return matmul(ma1,ma2)
    
    def qnm2npa(a):
        # Qnmatrix to np.array converter
        la=len(a)
        b=np.zeros(la,la,3) #la x la qnnum matrix 
        for i in range(la):
            for j in range(la):
                b[i][j]=[a[i][j].n[0],a[i][j].n[1],a[i][j].n[2]]
        return b
    
    def qnm2flt(a):
        la=len(a)
        b=np.zeros(la,la,3)  #la x la qnnum matrix 
        N=a[0].N
        for i in range(la):
            for j in range(la):
                b[i][j]=(a[i][j].n[0]+a[i][j].n[1]*np.sqrt(N))/a[i][j].n[2]
        return b
    
        

	if __name__ == '__main__':
	    # test
	    import random
	    sys.path.append('.')
	    import numericalc

	    ncycle=20
	    eps=1e-3
	    