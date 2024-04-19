#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
import numpy as np

TAU=(1+np.sqrt(5))/2.0

def add(a,b):
    # a+b in TAU-style
    c1=a[0]*b[2]+b[0]*a[2]
    c2=a[1]*b[2]+b[1]*a[2]
    c3=a[2]*b[2]
    x=np.array([c1,c2,c3])
    gcd=np.gcd.reduce(x)
    c1=c1/gcd
    c2=c2/gcd
    c3=c3/gcd
    if c3<0:
        return np.array([int(-c1),int(-c2),int(-c3)])
    else:
        return np.array([int(c1),int(c2),int(c3)])

def mul(a,b): 
    # A*B in TAU-style
    c1=a[0]*b[0]+a[1]*b[1]
    c2=a[0]*b[1]+a[1]*b[0]+a[1]*b[1]
    c3=a[2]*b[2]
    x=np.array([c1,c2,c3])
    gcd=np.gcd.reduce(x)
    c1=c1/gcd
    c2=c2/gcd
    c3=c3/gcd
    if c3<0:
        return np.array([int(-c1),int(-c2),int(-c3)])
    else:
        return np.array([int(c1),int(c2),int(c3)])

def sub(a,b):
    # a-b in TAU-style
    c=np.array([-1,0,1])
    b=mul(c,b)
    return add(a,b)

def div(a,b):
    # a/b in TAU-style
    if np.all(b[:2]==0):
        print('ERROR_1:division error')
        return 
    else:
        if np.all(a[:2]==0):
            return np.array([0,0,1])
        else:
            if b[1]!=0:
                if b[0]!=0:
                    c1=(a[0]*b[0] + a[0]*b[1] - a[1]*b[1])*b[2]
                    c2=(a[1]*b[0] - a[0]*b[1]            )*b[2]
                    c3=(b[0]*b[0] - b[1]*b[1] + b[0]*b[1])*a[2]
                else:
                    c1=(-a[0]+a[1])*b[2]
                    c2=        a[0]*b[2]
                    c3=        b[1]*a[2]
            else:
                #if b[0]!=0:
                c1=a[0]*b[2]
                c2=a[1]*b[2]
                c3=b[0]*b[2]
            x=np.array([c1,c2,c3])
            gcd=np.gcd.reduce(x)
            if gcd!=0:
                c1=c1/gcd
                c2=c2/gcd
                c3=c3/gcd
                if c3<0:
                    return np.array([int(-c1),int(-c2),int(-c3)])
                else:
                    return np.array([int(c1),int(c2),int(c3)])
            else:
                print('ERROR_2:division error')
                return 

def numeric_value(a):
    return (a[0]+a[1]*TAU)/a[2]

def centroid(tetrahedron):
    # geometric center, centroid of tetrahedron
    v0=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
    for i2 in range(6):
        v2=v0[i2]
        for i1 in range(4):
            v1=add(v1,tetrahedron[i1][i2])
        v2[i2]=mul(v1,np.array([1,0,4]))
    return v2

def outer_product(v1,v2):
    
    a=mul(v1[1],v2[2])
    b=mul(v1[2],v2[1])
    c1=sub(a,b)
    #
    a=mul(v1[2],v2[0])
    b=mul(v1[0],v2[2])
    [c2=sub(a,b)
    #
    a=mul(v1[0],v2[1])
    b=mul(v1[1],v2[0])
    c3=sub(a,b)
    #
    return np.array([c1,c2,c3])

def inner_product(v1,v2):
    a=mul(v1[0],v2[0])
    b=mul(v1[1],v2[1])
    c=mul(v1[2],v2[2])
    a=add(a,b)
    return add(a,c)

def int coplanar_check(p):
    # coplanar check
    # p: points to be cheked.
    if len(p)>3:
        xyz0i=projection3(p[0])
        xyz1i=projection3(p[1])
        xyz2i=projection3(p[2])
        v1=sub_vectors(xyz1i,xyz0i)
        v2=sub_vectors(xyz2i,xyz0i)
        v3=outer_product(v1,v2)
        flag=0
        for i1 in range(3,len(p)):
            xyz3i=projection3(p[i1])
            v4=sub_vectors(xyz3i,xyz0i)
            
            d=inner_product(v3,v4)
            if np.all(d[:2])==0:
                flag+=0
            else:
                flag+=1
                break
        if flag==0:
            return 1 # coplanar
        else:
            return 0
    else:
        return 1 # coplanar

def projection(v):
    # projection of a 6d vector onto Epar and Eperp, using "TAU-style"
    #
    # NOTE: coefficient (alpha) of the projection matrix is set to be 1.
    # alpha = a/np.sqrt(2.0+TAU)
    # see Yamamoto ActaCrystal (1997)
    m0=np.array([ 0, 0, 1]) #  0 in 'TAU-style'
    m1=np.array([ 1, 0, 1]) #  1
    m2=np.array([-1, 0, 1]) # -1
    m3=np.array([ 0, 1, 1]) #  tau
    m4=np.array([ 0,-1, 1]) # -tau
    v1e=mtrixcal(m1,m3,m3,m0,m2,m0,v) # 1,tau,tau,0,-1,0
    v2e=mtrixcal(m3,m0,m0,m1,m3,m1,v) # tau,0,0,1,TAU,1
    v3e=mtrixcal(m0,m1,m2,m4,m0,m3,v) # 0,1,-1,-tau,0,tau
    v1i=mtrixcal(m3,m2,m2,m0,m4,m0,v) # tau,-1,-1,0,-tau,0
    v2i=mtrixcal(m2,m0,m0,m3,m2,m3,v) # -1,0,0,tau,-1,tau
    v3i=mtrixcal(m0,m3,m4,m1,m0,m2,v) # 0,tau,-tau,1,0,-1
    return np.array([v1e,v2e,v3e,v1i,v2i,v3i])

def list projection3(v):
    # projection of a 6d vector onto Eperp, using "TAU-style"
    #
    # NOTE: coefficient (alpha) of the projection matrix is set to be 1.
    # alpha = a/np.sqrt(2.0+TAU)
    # see Yamamoto ActaCrystal (1997)
    m0=np.array([ 0, 0, 1]) #  0 in 'TAU-style'
    m1=np.array([ 1, 0, 1]) #  1
    m2=np.array([-1, 0, 1]) # -1
    m3=np.array([ 0, 1, 1]) #  tau
    m4=np.array([ 0,-1, 1]) # -tau
    #v1e=mtrixcal(m1,m3,m3,m0,m2,m0,h1,h2,h3,h4,h5,h6) # 1,tau,tau,0,-1,0
    #v2e=mtrixcal(m3,m0,m0,m1,m3,m1,h1,h2,h3,h4,h5,h6) # tau,0,0,1,TAU,1
    #v3e=mtrixcal(m0,m1,m2,m4,m0,m3,h1,h2,h3,h4,h5,h6) # 0,1,-1,-tau,0,tau
    v1i=mtrixcal(m3,m2,m2,m0,m4,m0,v) # tau,-1,-1,0,-tau,0
    v2i=mtrixcal(m2,m0,m0,m3,m2,m3,v) # -1,0,0,tau,-1,tau
    v3i=mtrixcal(m0,m3,m4,m1,m0,m2,v) # 0,tau,-tau,1,0,-1
    return np.array([v1i,v2i,v3i])

def projection_perp(v):
    # This returns 6D indeces of a projection of 6D vector (v) onto Eperp
    # Direct lattice vector
    m1=np.array([ 1, 0, 2]) #  (TAU+2)/2/(2+TAU)=1/2 in 'TAU-style'
    m2=np.array([-1, 2,10]) #  TAU/2/(2+TAU)
    m3=np.array([ 1,-2,10]) # -TAU/2/(2+TAU)
    h1=mtrixcal(m1,m3,m3,m3,m3,m3,v) # (tau+2,-tau,-tau,-tau,-tau,-tau)/2
    h2=mtrixcal(m3,m1,m3,m2,m2,m3,v) # (-tau,tau+2,-tau,tau,tau,-tau)/2
    h3=mtrixcal(m3,m3,m1,m3,m2,m2,v) # (-tau,-tau,tau+2,-tau,tau,tau)/2
    h4=mtrixcal(m3,m2,m3,m1,m3,m2,v) # (-tau,tau,-tau,tau+2,-tau,tau)/2
    h5=mtrixcal(m3,m2,m2,m3,m1,m3,v) # (-tau,tau,tau,-tau,tau+2,-tau)/2
    h6=mtrixcal(m3,m3,m2,m2,m3,m1,v) # (-tau,-tau,tau,tau,-tau,tau+2)/2
    return np.array([h1,h2,h3,h4,h5,h6])
    #const=1/(2.0+TAU)
    #m1 =((TAU+2)*n1 -     TAU*n2 -     TAU*n3 -     TAU*n4 -     TAU*n5 -     TAU*n6)/2.0*const
    #m2 = ( - TAU*n1 + (TAU+2)*n2 -     TAU*n3 +     TAU*n4 +     TAU*n5 -     TAU*n6)/2.0*const
    #m3 = ( - TAU*n1 -     TAU*n2 + (TAU+2)*n3 -     TAU*n4 +     TAU*n5 +     TAU*n6)/2.0*const
    #m4 = ( - TAU*n1 +     TAU*n2 -     TAU*n3 + (TAU+2)*n4 -     TAU*n5 +     TAU*n6)/2.0*const
    #m5 = ( - TAU*n1 +     TAU*n2 +     TAU*n3 -     TAU*n4 + (TAU+2)*n5 -     TAU*n6)/2.0*const
    #m6 = ( - TAU*n1 -     TAU*n2 +     TAU*n3 +     TAU*n4 -     TAU*n5 + (TAU+2)*n6)/2.0*const
    #return m1,m2,m3,m4,m5,m6

def mtrixcal(m1,m2,m3,m4,m5,m6,v2):
    a1=mul(m1,v2[0])
    a2=mul(m2,v2[1])
    a3=mul(m3,v2[2])
    a4=mul(m4,v2[3])
    a5=mul(m5,v2[4])
    a6=mul(m6,v2[5])
    a1=add(a1,a2)
    a1=add(a1,a3)
    a1=add(a1,a4)
    a1=add(a1,a5)
    a1=add(a1,a6)
    return a1
    
def dot_product(v1,v2):
    # product of vectors A and B
    #
    # vector A
    # Ax=(a1[0]+a1[1]*tau)/a1[2]
    # Ay=(a2[0]+a2[1]*tau)/a2[2]
    # Az=(a3[0]+a3[1]*tau)/a3[2]
    #
    # vector B
    # Bx=(b1[0]+b1[1]*tau)/b1[2]
    # By=(b2[0]+b2[1]*tau)/b2[2]
    # Bz=(b3[0]+b3[1]*tau)/b3[2]
    #    
    # return:
    # A*B = Ax*Bx + Ay*By + Az*Bz
    #     = (t1+t2*TAU)/t3
    t1=mul(a[0],b[0])
    t2=mul(a[1],b[1])
    t3=mul(a[2],b[2])
    t4=add(t1,t2)
    return add(t3,t4)

# これは必要か？　行列式？
def det_matrix(np.ndarray[DTYPE_int_t, ndim=2] a,
                        np.ndarray[DTYPE_int_t, ndim=2] b,
                        np.ndarray[DTYPE_int_t, ndim=2] c):
    # determinant of 3x3 matrix in TAU style
    #
    [t1,t2,t3]=mul(a[0][0],a[0][1],a[0][2],b[1][0],b[1][1],b[1][2])
    [t1,t2,t3]=mul(t1,t2,t3,c[2][0],c[2][1],c[2][2])
    #        
    [t4,t5,t6]=mul(a[2][0],a[2][1],a[2][2],b[0][0],b[0][1],b[0][2])
    [t4,t5,t6]=mul(t4,t5,t6,c[1][0],c[1][1],c[1][2])
    #
    [t7,t8,t9]=add(t1,t2,t3,t4,t5,t6)
    #
    [t1,t2,t3]=mul(a[1][0],a[1][1],a[1][2],b[2][0],b[2][1],b[2][2])
    [t1,t2,t3]=mul(t1,t2,t3,c[0][0],c[0][1],c[0][2])
    #
    [t7,t8,t9]=add(t7,t8,t9,t1,t2,t3)
    #
    [t1,t2,t3]=mul(a[2][0],a[2][1],a[2][2],b[1][0],b[1][1],b[1][2])
    [t1,t2,t3]=mul(t1,t2,t3,c[0][0],c[0][1],c[0][2])
    #
    [t7,t8,t9]=sub(t7,t8,t9,t1,t2,t3)
    #
    [t1,t2,t3]=mul(a[1][0],a[1][1],a[1][2],b[0][0],b[0][1],b[0][2])
    [t1,t2,t3]=mul(t1,t2,t3,c[2][0],c[2][1],c[2][2])
    #
    [t7,t8,t9]=sub(t7,t8,t9,t1,t2,t3)
    #
    [t1,t2,t3]=mul(a[0][0],a[0][1],a[0][2],b[2][0],b[2][1],b[2][2])
    [t1,t2,t3]=mul(t1,t2,t3,c[1][0],c[1][1],c[1][2])

    [t7,t8,t9]=sub(t7,t8,t9,t1,t2,t3)
    #
    return [t7,t8,t9]

def mul_vectors(vecs,a):
    
    for i in range(vecs.shape[0]):
        vecs[i]=mul_vector(vecs[i],a)
    return vecs
    
def mul_vector(v,a):
    
    for i in range(len(v)):
        v[i]=mul(v[i],a)
    return v

def add_vectors(v1,v2):
    """
    Vector composition, v1+b2
    """
    for i1 in range(len(v1)):
        v1[i1]=add(v1[i1],v2[i1])
    return v1

def sub_vectors(v1,v2):
    """
    Vector composition, v1-b2
    """
    v2=mul_vector(v2,np.array(-1,0,1))
    return add_vectors(v1,v2)
    
def triangle_area(a):
    
    vx1=numeric_value(a[1][0])-numeric_value(a[0][0])
    vy1=numeric_value(a[1][1])-numeric_value(a[0][1])
    vz1=numeric_value(a[1][2])-numeric_value(a[0][2])

    vx2=numeric_value(a[2][0])-numeric_value(a[0][0])
    vy2=numeric_value(a[2][1])-numeric_value(a[0][1])
    vz2=numeric_value(a[2][2])-numeric_value(a[0][2])
    
    v1=np.array([vx1,vy1,vz1])
    v2=np.array([vx2,vy2,vz2])
    
    v3=np.cross(v2,v1) # cross product
    return np.sqrt(np.sum(np.abs(v3**2)))/2.0

if __name__ == '__main__':
    
    
    def math_check(a0,b0):
        
        flg=0
        
        c=add(a,b)
        c1=numeric_value(c)
        a1=numeric_value(a)
        b1=numeric_value(b)
        c2=a1+b1
        if abs(c1-c2)<1e-5:
            flg+=1
            
        c=sub(a,b)
        c1=numeric_value(c)
        a1=numeric_value(a)
        b1=numeric_value(b)
        c2=a1-b1
        if abs(c1-c2)<1e-5:
            flg+=1
            
        c=mul(a,b)
        c1=numeric_value(c)
        a1=numeric_value(a)
        b1=numeric_value(b)
        c2=a1*b1
        if abs(c1-c2)<1e-5:
            flg+=1
            
        c=div(a,b)
        c1=numeric_value(c)
        a1=numeric_value(a)
        b1=numeric_value(b)
        c2=a1/b1
        if abs(c1-c2)<1e-5:
            flg+=1
            
        if flg==4:
            return 0
        else:
            return 1
    
    import random
    flg=0
    ntimes=20
    for _ in range(ntimes):
        a1=random.randrange(1,100)
        a2=random.randrange(1,100)
        a3=random.randrange(1,100)
        b1=random.randrange(1,100)
        b2=random.randrange(1,100)
        b3=random.randrange(1,100)
        a=np.array([a1,a2,a3])
        b=np.array([b1,b2,b3])
        flg+=math_check(a,b)
    if flg==0:
        print('Math_check: Correct!')
    else:
        print('Math_check: Wrong')
    