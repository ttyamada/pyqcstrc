import sys
import numpy as np
from numpy.typing import NDArray
#from pyqcstrc.qnvec import qnvec
    
class Qnvec:
    def __init__(self,vt: Qnvec,shape:np.int64):
        self.vt=vt
        self.shape = shape  #dimension of a vector a

    def add(vt1:Qnvec, vt2:Qnvec) -> Qnvec:
	    a=np.zeros(vt1.shape,dtype=np.int64)
	    for i in range(len(vt1)):
	        a[i]=vt1[i]+vt2[i]  #add(vt1[i],vt2[i])
	    return a

    def sub(vt1: Qnvec, vt2:Qnvec) -> Qnvec:
	    if vt1.ndim==2 and vt2.ndim==2:
	        return vt1-vt2  #add_vectors(vt1,vt2)
	    else:
	        print('incorrect shape')
	        return

    def mul_vector(vt: Qnvec, coeff:Qnvec) -> Qnvec:
	    if vt.ndim==2:
	        a=np.zeros(vt.shape,dtype=np.int64)
	        for i,v in enumerate(vt):
	            a[i]=v+coeff  #mul(v,coeff)
	        return a
	    else:
	        print('incorrect shape')
	        return

    def mul_vectors(vts: Qnvec, coeff:Qnvec) -> Qnvec:
	    if vts.ndim==3:
	        a=np.zeros(vts.shape,dtype=np.int64)
	        for i,vt in enumerate(vts):
	            a[i]=mul_vector(vt,coeff)
	        return a
	    elif vts.ndim==4:
	        a=np.zeros(vts.shape,dtype=np.int64)
	        for i1,vt in enumerate(vts):
	            for i2,v in enumerate(vt):
	                a[i1][i2]=mul_vector(v,coeff)
	    else:
	        print('incorrect shape')
	        return

    def shift_vectors(vts: Qnvec, vt: Qnvec) -> Qnvec:
	    if vts.ndim==3:
	        a=np.zeros(vts.shape,dtype=np.int64)
	        for i,vt1 in enumerate(vts):
	            a[i]=add_vectors(vt1,vt)
	        return a
	    elif vts.ndim==4:
	        a=np.zeros(vts.shape,dtype=np.int64)
	        for i1,vt1 in enumerate(vts):
	            for i2,vt2 in enumerate(vt1):
	                a[i1][i2]=add_vectors(vt2,vt)
	    else:
	        print('incorrect shape')
	        return

    def outer_product(vt1: Qnvec, vt2:Qnvec) -> Qnvec:
	    a=vt1[1]*vt2[2] #mul(vt1[1],vt2[2])
	    b=vt1[2]*vt2[1] #mul(vt1[2],vt2[1])
	    c1=a-b          #sub(a,b)
	    #
	    a=vt1[2]*vt2[0] #mul(vt1[2],vt2[0])
	    b=vt1[0]*vt2[2] #mul(vt1[0],vt2[2])
	    c2=a-b          #sub(a,b)
	    #
	    a=vt1[0]*vt2[1] #mul(vt1[0],vt2[1])
	    b=vt1[1]*vt2[0] #mul(vt1[1],vt2[0])
	    c3=a-b          #sub(a,b)
	    #
	    return np.array([c1,c2,c3],dtype=np.int64)

    def inner_product(vt1: Qnvec, vt2:Qnvec) -> Qnvec:
	    s1,_=vt1.shape
	    s2,_=vt2.shape
	    if s1!=s2:
	        print('matrices have not a proper shape.')
	        return 
	    else:
	        a=np.array([0,0,1])
	        for i in range(s1):
	            b=vt1[i]*vt2[i]  #mul(vt1[i],vt2[i])
	            a=a+b            #add(a,b)
	        return a

    def dot_product(mat1: Qnvec, mat2:Qnvec) -> Qnvec:
	    ndim1=mat1.ndim
	    ndim2=mat2.ndim
	    
	    if ndim1==2 and ndim2==2:
	        return inner_product(mat1,mat2)
	    
	    elif ndim1==3 and ndim2==2:
	        s,t1,_=mat1.shape
	        t2,_=mat2.shape
	        if t1!=t2:
	            print('incorrect shape found in dot_product')
	            return 
	        else:
	            mat_new=np.zeros((s,3),dtype=np.int64)
	            for k in range(s):
	                a=np.array([0,0,1])
	                for j in range(t1):
	                    b=mat1[k][j]*mat2[j]  #mul(mat1[k][j],mat2[j])
	                    a=a+b                 #add(a,b)
	                mat_new[k]=a
	            return mat_new
	            
	    elif ndim1==3 and ndim2==3:
	        s,t1,_=mat1.shape
	        t2,u,_=mat2.shape
	        if t1!=t2:
	            print('incorrect shape found in dot_product')
	            return 
	        else:
	            mat_new=np.zeros((s,u,3),dtype=np.int64)
	            for k in range(s):
	                for j in range(u):
	                    a=np.array([0,0,1])
	                    for i in range(t1):
	                        b=mat1[k][i]*mat2[i][j]  #mul(mat1[k][i],mat2[i][j])
	                        a=a+b                    #add(a,b)
	                    mat_new[k][j]=a
	            return mat_new
	    else:
	        print('incorrect shape found in dot_product')
	        return 

    def dot_product_1(mat1: Qnvec, mat2:Qnvec) -> Qnvec:
	    ndim1=mat1.ndim
	    ndim2=mat2.ndim
	    
	    if ndim1==2 and ndim2==2:
	        s,t1,=mat1.shape
	        t2,_=mat2.shape
	        if t1!=t2:
	            print('incorrect shape found in dot_product')
	            return 
	        else:
	            mat_new=np.zeros((s,3),dtype=np.int64)
	            for k in range(s):
	                a=np.array([0,0,1])
	                for j in range(t1):
	                    val=np.array([mat1[k][j],0,1])
	                    b=val*mat2[j]  #mul(val,mat2[j])
	                    a=a+b          #add(a,b)
	                mat_new[k]=a
	            return mat_new
	            
	    elif ndim1==2 and ndim2==3:
	        s,t1,=mat1.shape
	        t2,u,_=mat2.shape
	        if t1!=t2:
	            print('incorrect shape found in dot_product')
	            return 
	        else:
	            mat_new=np.zeros((s,u,3),dtype=np.int64)
	            for k in range(s):
	                for j in range(u):
	                    a=np.array([0,0,1])
	                    for i in range(t1):
	                        val=np.array([mat1[k][j],0,1])
	                        b=val*mat2[i][j]  #mul(val,mat2[i][j])
	                        a=a+b             #add(a,b)
	                    mat_new[k][j]=a
	            return mat_new
	    else:
	        print('incorrect shape found in dot_product')
	        return 

    def __add__(a, b):
        return add(a,b)
    
    def __sub__(a, b):
        return sub(a,b)
  
    