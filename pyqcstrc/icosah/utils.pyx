import sys
import numpy as np
cimport numpy as np
cimport cython

from pyqcstrc.icosah.math1 cimport det_matrix, projection, add, sub, mul, div
from pyqcstrc.icosah.numericalc cimport obj_volume_6d_numerical

DTYPE_double = np.float64
DTYPE_int = int

cdef np.float64_t TAU=(1+np.sqrt(5))/2.0
cdef np.float64_t TOL=1e-6 # tolerance

#cpdef np.ndarray full_sym_obj_edge(np.ndarray[DTYPE_int_t, ndim=4] od):
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray generator_edge(np.ndarray[DTYPE_int_t, ndim=4] obj, int verbose):
    # remove doubling edges in a set of triangles of the surface of OD
    # parameter: object (dim4)
    # return: doubling edges in a set of triangles (dim4)
    cdef int i,j,num1,num2,counter_ab,counter_ac,counter_bc
    cdef list edge
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,tmp1b,tmp1c,tmp1d
    cdef np.ndarray[DTYPE_int_t,ndim=2] tmp2a,tmp2b,tmp2c
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a,tmp4b#,obj
        
    if verbose>0:
        print('      generator_edge()')
    else:
        pass
    # three edges of 1st triangle
    tmp1a=np.append(obj[0][0],obj[0][1]) #edge 0-1
    tmp1a=np.append(tmp1a,obj[0][0]) 
    tmp1a=np.append(tmp1a,obj[0][2])     #edge 0-2
    tmp1a=np.append(tmp1a,obj[0][1]) 
    tmp1a=np.append(tmp1a,obj[0][2])     #edge 1-2    
    num2=int(len(tmp1a)/2/6/3)
    tmp4a=np.array(tmp1a).reshape(num2,2,6,3)
    num1=len(obj)
    for i in range(1,num1):
        tmp2a=obj[i][0]
        tmp2b=obj[i][1]
        tmp2c=obj[i][2]
        tmp1d=np.array([],dtype=int)
        counter_ab=0
        counter_ac=0
        counter_bc=0
        for j in range(0,num2):
            tmp1a=np.array([],dtype=int)
            if np.all(tmp2a==tmp4a[j][0]) and np.all(tmp2b==tmp4a[j][1]):
                counter_ab+=1
                break
            elif np.all(tmp2b==tmp4a[j][0]) and np.all(tmp2a==tmp4a[j][1]):
                counter_ab+=1
                break
            else:
                counter_ab+=0
        if counter_ab==0:
            tmp1a=np.append(tmp2a,tmp2b)
        else:
            tmp1a=np.array([],dtype=int)
        for j in range(0,num2):
            tmp1b=np.array([],dtype=int)
            if np.all(tmp2a==tmp4a[j][0]) and np.all(tmp2c==tmp4a[j][1]):
                counter_ac+=1
                break
            elif np.all(tmp2c==tmp4a[j][0]) and np.all(tmp2a==tmp4a[j][1]):
                counter_ac+=1
                break
            else:
                counter_ac+=0
        if counter_ac==0:
            tmp1b=np.append(tmp2a,tmp2c)
        else:
            tmp1b=np.array([],dtype=int)
        for j in range(0,num2):
            tmp1c=np.array([],dtype=int)
            if np.all(tmp2b==tmp4a[j][0]) and np.all(tmp2c==tmp4a[j][1]):
                counter_bc+=1
                break
            elif np.all(tmp2c==tmp4a[j][0]) and np.all(tmp2b==tmp4a[j][1]):
                counter_bc+=1
                break
            else:
                counter_bc+=0
        if counter_bc==0:
            tmp1c=np.append(tmp2b,tmp2c)
        else:
            tmp1c=np.array([],dtype=int)
        tmp1d=np.append(tmp1d,tmp1a)
        tmp1d=np.append(tmp1d,tmp1b)
        tmp1d=np.append(tmp1d,tmp1c)
        tmp1d=np.append(tmp4a,tmp1d)
        num2=int(len(tmp1d)/36) # 36=2*6*3
        tmp4a=tmp1d.reshape(num2,2,6,3)
    if verbose>0:
        print('       Number of edges: %d'%(len(tmp4a)))
    else:
        pass
    return tmp4a

# new version (much faster)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray generator_surface_1(np.ndarray[DTYPE_int_t, ndim=4] obj,
                                        int verbose):
    #
    # remove doubling surface in a set of tetrahedra in the OD (dim4)
    #
    cdef int i,j,k,val,num1,counter1,counter2,counter3
    cdef list edge,comb,list1
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1,tmp1a,tmp1k,tmp1j
    cdef np.ndarray[DTYPE_int_t,ndim=2] tmp2a
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a
    
    if verbose>0:
        print('      generator_surface_1()')
    else:
        pass
    #
    # (1) preparing a list of triangle surfaces without doubling (tmp2)
    #
    if verbose>1:
        print('       Number of triangles in POD: %d'%(len(obj)*4))
    else:
        pass
    
    #
    # initialization of tmp2 by 1st tetrahedron
    #
    # three triangles of tetrahedron: 1-2-3, 1-2-4, 1-3-4, 2-3-4
    comb=[[0,1,2],[0,1,3],[0,2,3],[1,2,3]] 
    #
    # Four triangles of 1-th tetrahedron
    for k in range(len(comb)):
        if k==0:
            tmp1=np.append(obj[0][comb[k][0]],obj[0][comb[k][1]])
        else:
            tmp1=np.append(tmp1,obj[0][comb[k][0]])
            tmp1=np.append(tmp1,obj[0][comb[k][1]])
        tmp1=np.append(tmp1,obj[0][comb[k][2]])
    num1=int(len(tmp1)/len(comb))
    tmp2a=tmp1.reshape(len(comb),num1)
    #

    list1=[]
    for i in range(1,len(obj)): # i-th tetrahedron
        tmp1=np.array([0])
        for k in range(len(comb)): # k-th triangle of i-th tetrahedron
            tmp1k=np.append(obj[i][comb[k][0]],obj[i][comb[k][1]])
            tmp1k=np.append(tmp1k,obj[i][comb[k][2]])
            counter1=0
            for j in range(len(tmp2a)): # j-th triangle in list 'tmp2'
                tmp1j=tmp2a[j]
                #val=equivalent_triangle(tmp1k,tmp1j) # val=0 (equivalent), 1 (non equivalent)
                val=equivalent_triangle_1(tmp1k,tmp1j) # val=0 (equivalent), 1 (non equivalent)
                if val==0:
                    counter1+=1
                    list1.append(j)
                    break
                else:
                    pass
            if counter1==0:
                if len(tmp1)==1:
                    tmp1=tmp1k
                else:
                    tmp1=np.append(tmp1,tmp1k)
        if len(tmp1)==1:
            pass
        else:
            tmp1=np.append(tmp2a,tmp1)
            tmp2a=tmp1.reshape(int(len(tmp1)/num1),num1)

    if verbose>0:
        print('       Number of unique triangles: %d'%(len(tmp2a)))
    else:
        pass

    tmp1=np.array([0])
    for i in range(len(tmp2a)):
        counter1=0
        for j in list1:
            if i==j:
                counter1+=1
                #break
            else:
                pass
        if counter1==0:
            if len(tmp1)==1:
                tmp1=tmp2a[i]
            else:
                tmp1=np.append(tmp1,tmp2a[i])
        elif counter1==1:
            pass
        else:
            if verbose>0:
                print('      ERROR_001 %d: check your model.'%(counter1))
            else:
                pass
                
    if verbose>0:
        print('       Number of triangles on POD surface:%d'%(int(len(tmp1)/54)))
    else:
        pass
    
    return tmp1.reshape(int(len(tmp1)/54),3,6,3)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray generator_surface(np.ndarray[DTYPE_int_t, ndim=4] obj,
                                    int verbose):
    #
    # remove doubling surface in a set of tetrahedra in the OD (dim4)
    #
    cdef int i,j,k,val,num1,counter1,counter2,counter3
    cdef list edge,comb
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1,tmp1a,tmp1k,tmp1j
    cdef np.ndarray[DTYPE_int_t,ndim=2] tmp2
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a
    
    if verbose>0:
        print('      generator_surface()')
    else:
        pass
    #
    # (1) preparing a list of triangle surfaces without doubling (tmp2)
    #
    if verbose>0:
        print('       Number of triangles in POD: %d'%(len(obj)*4))
    else:
        pass
    
    #
    # initialization of tmp2 by 1st tetrahedron
    #
    # three triangles of tetrahedron: 1-2-3, 1-2-4, 1-3-4, 2-3-4
    comb=[[0,1,2],[0,1,3],[0,2,3],[1,2,3]] 
    #
    # Four triangles of 1-th tetrahedron
    for k in range(len(comb)):
        if k==0:
            tmp1=np.append(obj[0][comb[k][0]],obj[0][comb[k][1]])
        else:
            tmp1=np.append(tmp1,obj[0][comb[k][0]])
            tmp1=np.append(tmp1,obj[0][comb[k][1]])
        tmp1=np.append(tmp1,obj[0][comb[k][2]])
    num1=int(len(tmp1)/len(comb))
    tmp2=tmp1.reshape(len(comb),num1)
    #
    for i in range(1,len(obj)): # i-th tetrahedron
        for k in range(len(comb)): # k-th triangle of i-th tetrahedron
            tmp1k=np.append(obj[i][comb[k][0]],obj[i][comb[k][1]])
            tmp1k=np.append(tmp1k,obj[i][comb[k][2]])
            counter1=0
            for j in range(len(tmp2)): # j-th triangle in list 'tmp2'
                tmp1j=tmp2[j]
                val=equivalent_triangle(tmp1k,tmp1j) # val=0 (equivalent), 1 (non equivalent)
                if val==0:
                    counter1+=1
                    break
                else:
                    pass
            if counter1==0:
                tmp1=np.append(tmp2,tmp1k)
                tmp2=tmp1.reshape(int(len(tmp1)/num1),num1)
            else:
                pass
    if verbose>0:
        print('       Number of unique triangles: %d'%(len(tmp2)))
    else:
        pass
    #
    # (2) how many each unique triangle in "tmp2" is found in "obj", and count.
    #     if a triangle is NOT part of surface of OD, we can find the triangle in "obj" two times.
    counter2=0
    counter3=0
    for j in range(len(tmp2)): # j-th triangle in the unique triangle list 'tmp2'
        tmp1j=tmp2[j]
        counter1=0
        for i in range(0,len(obj)): # i-th tetrahedron in 'obj'
            for k in range(len(comb)): # k-th triangke of i-th tetrahedron
                tmp1i=np.append(obj[i][comb[k][0]],obj[i][comb[k][1]])
                tmp1i=np.append(tmp1i,obj[i][comb[k][2]])
                val=equivalent_triangle(tmp1i,tmp1j) # val=0 (equivalent), 1 (non equivalent)
                #val=equivalent_triangle_1(tmp1i,tmp1j) # val=0 (equivalent), 1 (non equivalent)
                if val==0:
                    counter1+=1
                    if counter1==2:
                        break
                    else:
                        pass
                else:
                    pass
        if counter1==1:
            if counter2==0:
                tmp1a=tmp1j
            else:
                tmp1a=np.append(tmp1a,tmp1j)
            counter2+=1
        elif counter1==2:
            pass
        elif counter1>2:
            counter3+=1
        else:
            pass
    if counter3>0:
        if verbose>0:
            print('      ERROR_001 %d: check your model.'%(counter3))
        else:
            pass
        return np.array([[[[0]]]])
    else:
        tmp4a=tmp1a.reshape(int(len(tmp1a)/54),3,6,3) # 54=3*6*3
        if verbose>0:
            print('       Number of triangles on POD surface:%d'%(counter2))
        else:
            pass
        return tmp4a

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int equivalent_triangle(np.ndarray[DTYPE_int_t, ndim=1] triangle1,
                            np.ndarray[DTYPE_int_t, ndim=1] triangle2):
                            
    cdef int i,i1,i2,i3,i4,i5,i6,counter
    cdef np.ndarray[DTYPE_int_t,ndim=1] t1,t2,t3,a1,a2,a3,b1,b2,b3,c1,c2,c3,d1,d2,d3,e1,e2,e3,f1,f2,f3
    cdef np.ndarray[DTYPE_int_t,ndim=2] tmp2a,tmp2b,tmp2c,tmp2d,tmp2e,tmp2f
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a,tmp3b
    cdef list comb
    tmp3a=triangle1.reshape(int(len(triangle1)/18),6,3) # 18=6*3
    tmp3b=triangle2.reshape(int(len(triangle2)/18),6,3)
    #triangle 1: a1,a2,a3
    #triangle 2: b1,b2,b3
    # all combinations, 6
    # a1=b1,a2=b2,a3=b3    1 2 3
    # a1=b2,a2=b3,a3=b1    2 3 1
    # a1=b3,a2=b1,a3=b2    3 1 2
    # a1=b1,a2=b3,a3=b2    1 3 2
    # a1=b3,a2=b2,a3=b1    3 2 1
    # a1=b2,a2=b1,a3=b3    2 1 3
    comb=[[0,1,2,0,1,2],\
        [0,1,2,1,2,0],\
        [0,1,2,2,0,1],\
        [0,1,2,0,2,1],\
        [0,1,2,2,1,0],\
        [0,1,2,1,0,2]]
    counter=0
    for i in range(len(comb)):
        [i1,i2,i3,i4,i5,i6]=comb[i]
        #if np.all(tmp3a[i1]==tmp3b[i4]) and np.all(tmp3a[i2]==tmp3b[i5]) and np.all(tmp3a[i3]==tmp3b[i6]):
        t1,t2,t3,a1,a2,a3=projection(tmp3a[i1][0],tmp3a[i1][1],tmp3a[i1][2],tmp3a[i1][3],tmp3a[i1][4],tmp3a[i1][5])
        t1,t2,t3,b1,b2,b3=projection(tmp3b[i4][0],tmp3b[i4][1],tmp3b[i4][2],tmp3b[i4][3],tmp3b[i4][4],tmp3b[i4][5])
        t1,t2,t3,c1,c2,c3=projection(tmp3a[i2][0],tmp3a[i2][1],tmp3a[i2][2],tmp3a[i2][3],tmp3a[i2][4],tmp3a[i2][5])
        t1,t2,t3,d1,d2,d3=projection(tmp3b[i5][0],tmp3b[i5][1],tmp3b[i5][2],tmp3b[i5][3],tmp3b[i5][4],tmp3b[i5][5])
        t1,t2,t3,e1,e2,e3=projection(tmp3a[i3][0],tmp3a[i3][1],tmp3a[i3][2],tmp3a[i3][3],tmp3a[i3][4],tmp3a[i3][5])
        t1,t2,t3,f1,f2,f3=projection(tmp3b[i6][0],tmp3b[i6][1],tmp3b[i6][2],tmp3b[i6][3],tmp3b[i6][4],tmp3b[i6][5])
        tmp2a=np.array([a1,a2,a3])
        tmp2b=np.array([b1,b2,b3])
        tmp2c=np.array([c1,c2,c3])
        tmp2d=np.array([d1,d2,d3])
        tmp2e=np.array([e1,e2,e3])
        tmp2f=np.array([f1,f2,f3])
        if np.all(tmp2a==tmp2b) and np.all(tmp2c==tmp2d) and np.all(tmp2e==tmp2f):
            counter+=1
            break
        else:
            pass
    if counter==0:
        return 1 # not equivalent traiangles
    else:
        return 0 # equivalent traiangle

# another version of equivalent_triangle()
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int equivalent_triangle_1(np.ndarray[DTYPE_int_t, ndim=1] triangle1,\
                            np.ndarray[DTYPE_int_t, ndim=1] triangle2):
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3
    tmp1=np.append(triangle1,triangle2)
    tmp3=remove_doubling_dim3_in_perp_space(tmp1.reshape(int(len(tmp1)/18),6,3))
    if len(tmp3)!=3:
        return 1 # not equivalent traiangles
    else:
        return 0 # equivalent traiangle

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray shift_object(np.ndarray[DTYPE_int_t, ndim=4] obj,
                            np.ndarray[DTYPE_int_t, ndim=2] shift,
                            int vorbose):
    cdef int i1,i2,i3
    cdef long n1,n2,n3
    cdef long v0,v1,v2,v3,v4,v5DTYPE_int_t
    cdef double vol1,vol2
    cdef list a
    cdef np.ndarray[DTYPE_int_t, ndim=4] obj_new
    
    a=[]
    v0,v1,v2=obj_volume_6d(obj)
    vol1=obj_volume_6d_numerical(obj)
    
    if vorbose>0:
        print('        shift_object()')
        if vorbose>1:
            if vorbose>0:
                print('         volume: %d %d %d (%10.8f)'%(v0,v1,v2,vol1))
            else:
                pass
    else:
        pass
    
    for i1 in range(len(obj)):
        for i2 in range(4):
            for i3 in range(6):
                n1,n2,n3=add(obj[i1][i2][i3][0],obj[i1][i2][i3][1],obj[i1][i2][i3][2],shift[i3][0],shift[i3][1],shift[i3][2])
                a.append(n1)
                a.append(n2)
                a.append(n3)
    obj_new=np.array(a).reshape(len(obj),4,6,3)
    
    v3,v4,v5=obj_volume_6d(obj_new)
    vol2=obj_volume_6d_numerical(obj_new)
    if v0==v3 and v1==v4 and v2==v5 or abs(vol1-vol2)<vol1*TOL:
        if vorbose>0:
            print(' succeeded')
        else:
            pass
        return obj_new
    else:
        print(' fail')
        return np.array([[[[0]]]])

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list obj_volume_6d(np.ndarray[DTYPE_int_t, ndim=4] obj):
    cdef int i
    cdef long v1,v2,v3,w1,w2,w3
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3
    w1,w2,w3=0,0,1
    for i in range(len(obj)):
        tmp3=obj[i]
        [v1,v2,v3]=tetrahedron_volume_6d(tmp3)
        w1,w2,w3=add(w1,w2,w3,v1,v2,v3)
    return [w1,w2,w3]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list tetrahedron_volume_6d(np.ndarray[DTYPE_int_t, ndim=3] tetrahedron):
    cdef long v1,v2,v3
    cdef np.ndarray[DTYPE_int_t,ndim=1] x1e,y1e,z1e,x1i,y1i,z1i,x2i,y2i,z2i,x3i,y3i,z3i,x4i,y4i,z4i
    x1e,y1e,z1e,x1i,y1i,z1i=projection(tetrahedron[0][0],tetrahedron[0][1],tetrahedron[0][2],tetrahedron[0][3],tetrahedron[0][4],tetrahedron[0][5])
    x1e,y1e,z1e,x2i,y2i,z2i=projection(tetrahedron[1][0],tetrahedron[1][1],tetrahedron[1][2],tetrahedron[1][3],tetrahedron[1][4],tetrahedron[1][5])
    x1e,y1e,z1e,x3i,y3i,z3i=projection(tetrahedron[2][0],tetrahedron[2][1],tetrahedron[2][2],tetrahedron[2][3],tetrahedron[2][4],tetrahedron[2][5])
    x1e,y1e,z1e,x4i,y4i,z4i=projection(tetrahedron[3][0],tetrahedron[3][1],tetrahedron[3][2],tetrahedron[3][3],tetrahedron[3][4],tetrahedron[3][5])
    [v1,v2,v3]=tetrahedron_volume(np.array([x1i,y1i,z1i]),np.array([x2i,y2i,z2i]),np.array([x3i,y3i,z3i]),np.array([x4i,y4i,z4i]))
    return [v1,v2,v3]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef list tetrahedron_volume(np.ndarray[DTYPE_int_t, ndim=2] v1,
                            np.ndarray[DTYPE_int_t, ndim=2] v2,
                            np.ndarray[DTYPE_int_t, ndim=2] v3,
                            np.ndarray[DTYPE_int_t, ndim=2] v4):
    # This function returns volume of a tetrahedron
    # input: vertex coordinates of the tetrahedron (x0,y0,z0),(x1,y1,z1),(x2,y2,z2),(x3,y3,z3)
    cdef long a1,a2,a3,b1,b2,b3,c1,c2,c3
    cdef np.ndarray[DTYPE_int_t, ndim=1] x0,x1,x2,x3,y0,y1,y2,y3,z0,z1,z2,z3
    cdef np.ndarray[DTYPE_int_t, ndim=2] a,b,c
    #
    x0=v1[0]
    y0=v1[1]
    z0=v1[2]
    #
    x1=v2[0]
    y1=v2[1]
    z1=v2[2]
    #
    x2=v3[0]
    y2=v3[1]
    z2=v3[2]
    #
    x3=v4[0]
    y3=v4[1]
    z3=v4[2]
    #
    [a1,a2,a3]=sub(x1[0],x1[1],x1[2],x0[0],x0[1],x0[2])
    [b1,b2,b3]=sub(y1[0],y1[1],y1[2],y0[0],y0[1],y0[2])
    [c1,c2,c3]=sub(z1[0],z1[1],z1[2],z0[0],z0[1],z0[2])
    a=np.array([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]])
    #
    [a1,a2,a3]=sub(x2[0],x2[1],x2[2],x0[0],x0[1],x0[2])
    [b1,b2,b3]=sub(y2[0],y2[1],y2[2],y0[0],y0[1],y0[2])
    [c1,c2,c3]=sub(z2[0],z2[1],z2[2],z0[0],z0[1],z0[2])
    b=np.array([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]])
    #
    [a1,a2,a3]=sub(x3[0],x3[1],x3[2],x0[0],x0[1],x0[2])
    [b1,b2,b3]=sub(y3[0],y3[1],y3[2],y0[0],y0[1],y0[2])
    [c1,c2,c3]=sub(z3[0],z3[1],z3[2],z0[0],z0[1],z0[2])
    c=np.array([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]])
    #
    [a1,a2,a3]=det_matrix(a,b,c) # determinant of 3x3 matrix
    #
    # avoid a negative value
    #
    if a1+a2*TAU<0.0: # to avoid negative volume...
        [a1,a2,a3]=mul(a1,a2,a3,-1,0,6)
    else:
        [a1,a2,a3]=mul(a1,a2,a3,1,0,6)
    return [a1,a2,a3]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray remove_doubling_dim4(np.ndarray[DTYPE_int_t, ndim=4] obj):
    cdef int i1,i2,j,counter,num
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,tmp1b
    cdef np.ndarray[DTYPE_int_t,ndim=2] tmp2a
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a
    cdef np.ndarray[DTYPE_int_t,ndim=4] tmp4a,tmp4b    
    num=len(obj[0])
    tmp1a=np.append(obj[0][0],obj[0][1])
    for i1 in range(2,num):
        tmp1a=np.append(tmp1a,obj[0][i1])
    tmp3a=tmp1a.reshape(int(len(tmp1a)/18),6,3) # 18=6*3
    for i1 in range(1,len(obj)):
        for i2 in range(num):
            tmp2a=obj[i1][i2]
            counter=0
            for j in range(0,len(tmp3a)):
                if np.all(tmp2a==tmp3a[j]):
                    counter+=1
                    break
                else:
                    counter+=0
            if counter==0:
                tmp1b=np.append(tmp3a,tmp2a)
                tmp3a=tmp1b.reshape(int(len(tmp1b)/18),6,3) # 18=6*3
            else:
                pass
    return tmp3a

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray remove_doubling_dim3(np.ndarray[DTYPE_int_t, ndim=3] obj):
    cdef int i,j,counter
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a
    cdef np.ndarray[DTYPE_int_t,ndim=2] tmp2a
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a
    tmp2a=obj[0]
    tmp3a=tmp2a.reshape(1,6,3)
    for i in range(1,len(obj)):
        tmp2a=obj[i]
        counter=0
        for j in range(0,len(tmp3a)):
            if np.all(tmp2a==tmp3a[j]):
                counter+=1
                break
            else:
                counter+=0
        if counter==0:
            tmp1a=np.append(tmp3a,tmp2a)
            tmp3a=tmp1a.reshape(int(len(tmp1a)/18),6,3) # 18=6*3
        else:
            pass
    return tmp3a

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray remove_doubling_dim4_in_perp_space(np.ndarray[DTYPE_int_t, ndim=4] obj):
    # remove 6d coordinates which is doubled in perpendicular space
    cdef int num
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a,tmp3b
    num=len(obj[0])
    tmp3a=obj.reshape(len(obj)*num,6,3)
    tmp3b=remove_doubling_dim3_in_perp_space(tmp3a)
    return tmp3b

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray remove_doubling_dim3_in_perp_space(np.ndarray[DTYPE_int_t, ndim=3] obj):
    # remove 6d coordinates which is doubled in perpendicular space
    cdef int i1,i2,counter1,counter2
    cdef np.ndarray[DTYPE_int_t,ndim=1] v1,v2,v3,v4,v5,v6
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,tmp1b
    cdef np.ndarray[DTYPE_int_t,ndim=2] tmp2b
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3a,tmp3b
    counter2=0
    if len(obj)>1:
        tmp2b=obj[0]
        v1,v2,v3,v4,v5,v6=projection(tmp2b[0],tmp2b[1],tmp2b[2],tmp2b[3],tmp2b[4],tmp2b[5])
        tmp1a=np.array([v4,v5,v6]).reshape(9)
        tmp3a=tmp1a.reshape(1,3,3) # perpendicular components
        #tmp3a=np.array([v4,v5,v6]).reshape(1,3,3) # perpendicular components
        tmp1b=tmp2b.reshape(18)
        tmp3b=tmp1b.reshape(1,6,3) # 6d
        #tmp3b=tmp2b.reshape(1,6,3) # 6d
        for i1 in range(1,len(obj)):
            tmp2b=obj[i1]
            v1,v2,v3,v4,v5,v6=projection(tmp2b[0],tmp2b[1],tmp2b[2],tmp2b[3],tmp2b[4],tmp2b[5])
            counter1=0
            for i2 in range(len(tmp3a)):
                if np.all(v4==tmp3a[i2][0]) and np.all(v5==tmp3a[i2][1]) and np.all(v6==tmp3a[i2][2]):
                    counter1+=1
                    break
                else:
                    counter1+=0
            if counter1==0:
                tmp1a=np.append(tmp1a,[v4,v5,v6])
                tmp1b=np.append(tmp1b,[tmp2b[0],tmp2b[1],tmp2b[2],tmp2b[3],tmp2b[4],tmp2b[5]])
            else:
                pass
            tmp3a=tmp1a.reshape(int(len(tmp1a)/9),3,3)
        return tmp1b.reshape(int(len(tmp1b)/18),6,3)
    else:
        return obj

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int generator_xyz_dim4(np.ndarray[DTYPE_int_t, ndim=4] obj, char filename):
    cdef int i1
    cdef np.ndarray[DTYPE_int_t,ndim=1] a1,a2,a3,a4,a5,a6
    cdef np.ndarray[DTYPE_int_t,ndim=2] tmp2
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3
    f=open('%s'%(filename),'w', encoding="utf-8", errors="ignore")
    tmp3=remove_doubling_dim4(obj)
    f.write('%d\n'%(len(tmp3)))
    f.write('%s\n'%(filename))
    for i1 in range(len(tmp3)):
        a1,a2,a3,a4,a5,a6=projection(tmp3[i1][0],tmp3[i1][1],tmp3[i1][2],tmp3[i1][3],tmp3[i1][4],tmp3[i1][5])
        f.write('Xx %8.6f %8.6f %8.6f # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
        ((a4[0]+a4[1]*TAU)/float(a4[2]),(a5[0]+a5[1]*TAU)/float(a5[2]),(a6[0]+a6[1]*TAU)/float(a6[2]),\
        tmp3[i1][0][0],tmp3[i1][0][1],tmp3[i1][0][2],\
        tmp3[i1][1][0],tmp3[i1][1][1],tmp3[i1][1][2],\
        tmp3[i1][2][0],tmp3[i1][2][1],tmp3[i1][2][2],\
        tmp3[i1][3][0],tmp3[i1][3][1],tmp3[i1][3][2],\
        tmp3[i1][4][0],tmp3[i1][4][1],tmp3[i1][4][2],\
        tmp3[i1][5][0],tmp3[i1][5][1],tmp3[i1][5][2]))
    f.closed
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int generator_xyz_dim4_triangle(np.ndarray[DTYPE_int_t, ndim=4] obj, char filename):
    cdef int i1,i2
    cdef np.ndarray[DTYPE_int_t,ndim=1] a1,a2,a3,a4,a5,a6
    f=open('%s'%(filename),'w', encoding="utf-8", errors="ignore")
    f.write('%d\n'%(len(obj)*3))
    f.write('%s\n'%(filename))
    for i1 in range(len(obj)):
        for i2 in range(3):
            a1,a2,a3,a4,a5,a6=projection(obj[i1][i2][0],obj[i1][i2][1],obj[i1][i2][2],obj[i1][i2][3],obj[i1][i2][4],obj[i1][i2][5])
            f.write('Xx %8.6f %8.6f %8.6f # %3d-the tetrahedron %d-th vertex # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
            ((a4[0]+a4[1]*TAU)/float(a4[2]),\
            (a5[0]+a5[1]*TAU)/float(a5[2]),\
            (a6[0]+a6[1]*TAU)/float(a6[2]),\
            i1,i2,\
            obj[i1][i2][0][0],obj[i1][i2][0][1],obj[i1][i2][0][2],\
            obj[i1][i2][1][0],obj[i1][i2][1][1],obj[i1][i2][1][2],\
            obj[i1][i2][2][0],obj[i1][i2][2][1],obj[i1][i2][2][2],\
            obj[i1][i2][3][0],obj[i1][i2][3][1],obj[i1][i2][3][2],\
            obj[i1][i2][4][0],obj[i1][i2][4][1],obj[i1][i2][4][2],\
            obj[i1][i2][5][0],obj[i1][i2][5][1],obj[i1][i2][5][2]))
    f.closed
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int generator_xyz_dim4_tetrahedron(np.ndarray[DTYPE_int_t, ndim=4] obj, char basename, int option):
    cdef int i1,i2
    cdef long w1,w2,w3
    cdef np.ndarray[DTYPE_int_t,ndim=1] a1,a2,a3,a4,a5,a6
    
    if option==0:
        f=open('%s.xyz'%(basename),'w', encoding="utf-8", errors="ignore")
        f.write('%d\n'%(len(obj)*4))
        f.write('%s.xyz\n'%(basename))
        for i1 in range(len(obj)):
            for i2 in range(4):
                a1,a2,a3,a4,a5,a6=projection(obj[i1][i2][0],obj[i1][i2][1],obj[i1][i2][2],obj[i1][i2][3],obj[i1][i2][4],obj[i1][i2][5])
                f.write('Xx %8.6f %8.6f %8.6f # %3d-the tetrahedron %d-th vertex # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
                ((a4[0]+a4[1]*TAU)/float(a4[2]),\
                (a5[0]+a5[1]*TAU)/float(a5[2]),\
                (a6[0]+a6[1]*TAU)/float(a6[2]),\
                i1,i2,\
                obj[i1][i2][0][0],obj[i1][i2][0][1],obj[i1][i2][0][2],\
                obj[i1][i2][1][0],obj[i1][i2][1][1],obj[i1][i2][1][2],\
                obj[i1][i2][2][0],obj[i1][i2][2][1],obj[i1][i2][2][2],\
                obj[i1][i2][3][0],obj[i1][i2][3][1],obj[i1][i2][3][2],\
                obj[i1][i2][4][0],obj[i1][i2][4][1],obj[i1][i2][4][2],\
                obj[i1][i2][5][0],obj[i1][i2][5][1],obj[i1][i2][5][2]))
        w1,w2,w3=obj_volume_6d(obj)
        f.write('volume = %d %d %d (%8.6f)\n'%(w1,w2,w3,(w1+TAU*w2)/float(w3)))
        for i1 in range(len(obj)):
            [v1,v2,v3]=tetrahedron_volume_6d(obj[i1])
            f.write('%3d-the tetrahedron, %d %d %d (%8.6f)\n'%(i1,v1,v2,v3,(v1+TAU*v2)/float(v3)))
        f.closed
        return 0
    elif option==1:
        for i1 in range(len(obj)):
            f=open('%s_%d.xyz'%(basename,i1),'w', encoding="utf-8", errors="ignore")
            f.write('%d\n'%(4))
            f.write('%s_%d.xyz\n'%(basename,i1))
            for i2 in range(4):
                a1,a2,a3,a4,a5,a6=projection(obj[i1][i2][0],obj[i1][i2][1],obj[i1][i2][2],obj[i1][i2][3],obj[i1][i2][4],obj[i1][i2][5])
                f.write('Xx %8.6f %8.6f %8.6f # %3d-the tetrahedron %d-th vertex # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
                ((a4[0]+a4[1]*TAU)/float(a4[2]),\
                (a5[0]+a5[1]*TAU)/float(a5[2]),\
                (a6[0]+a6[1]*TAU)/float(a6[2]),\
                i1,i2,\
                obj[i1][i2][0][0],obj[i1][i2][0][1],obj[i1][i2][0][2],\
                obj[i1][i2][1][0],obj[i1][i2][1][1],obj[i1][i2][1][2],\
                obj[i1][i2][2][0],obj[i1][i2][2][1],obj[i1][i2][2][2],\
                obj[i1][i2][3][0],obj[i1][i2][3][1],obj[i1][i2][3][2],\
                obj[i1][i2][4][0],obj[i1][i2][4][1],obj[i1][i2][4][2],\
                obj[i1][i2][5][0],obj[i1][i2][5][1],obj[i1][i2][5][2]))
                [v1,v2,v3]=tetrahedron_volume_6d(obj[i1])
            f.write('%3d-the tetrahedron, %d %d %d (%8.6f)\n'%(i1,v1,v2,v3,(v1+TAU*v2)/float(v3)))
        f.closed
        return 0
    else:
        return 1

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int generator_xyz_dim4_tetrahedron_select(np.ndarray[DTYPE_int_t, ndim=4] obj, char  filename, int num):
    cdef int i2
    cdef long w1,w2,w3
    cdef np.ndarray[DTYPE_int_t,ndim=1] a1,a2,a3,a4,a5,a6
    # option=0 # generate single .xyz file
    # option=1 # generate single .xyz file
    f=open('%s_%d.xyz'%(filename,num),'w', encoding="utf-8", errors="ignore")
    f.write('%d\n'%(4))
    f.write('%s_%d\n'%(filename,num))
    for i2 in range(4):
        a1,a2,a3,a4,a5,a6=projection(obj[num][i2][0],obj[num][i2][1],obj[num][i2][2],obj[num][i2][3],obj[num][i2][4],obj[num][i2][5])
        print('Xx %8.6f %8.6f %8.6f # %3d-the tetrahedron %d-th vertex # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
        ((a4[0]+a4[1]*TAU)/float(a4[2]),\
        (a5[0]+a5[1]*TAU)/float(a5[2]),\
        (a6[0]+a6[1]*TAU)/float(a6[2]),\
        num,i2,\
        obj[num][i2][0][0],obj[num][i2][0][1],obj[num][i2][0][2],\
        obj[num][i2][1][0],obj[num][i2][1][1],obj[num][i2][1][2],\
        obj[num][i2][2][0],obj[num][i2][2][1],obj[num][i2][2][2],\
        obj[num][i2][3][0],obj[num][i2][3][1],obj[num][i2][3][2],\
        obj[num][i2][4][0],obj[num][i2][4][1],obj[num][i2][4][2],\
        obj[num][i2][5][0],obj[num][i2][5][1],obj[num][i2][5][2]))
        [v1,v2,v3]=tetrahedron_volume_6d(obj[num])
    f.write('%3d-the tetrahedron, %d %d %d (%8.6f)\n'%(num,v1,v2,v3,(v1+TAU*v2)/float(v3)))
    f.close
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int generator_xyz_dim3(np.ndarray[DTYPE_int_t, ndim=3] obj, char filename):
    cdef int i1
    cdef np.ndarray[DTYPE_int_t,ndim=1] a1,a2,a3,a4,a5,a6
    cdef np.ndarray[DTYPE_int_t,ndim=2] tmp2
    cdef np.ndarray[DTYPE_int_t,ndim=3] tmp3
    f=open('%s'%(filename),'w', encoding="utf-8", errors="ignore")
    tmp3=remove_doubling_dim3(obj)
    f.write('%d\n'%(len(tmp3)))
    f.write('%s\n'%(filename))
    for i1 in range(len(tmp3)):
        a1,a2,a3,a4,a5,a6=projection(tmp3[i1][0],tmp3[i1][1],tmp3[i1][2],tmp3[i1][3],tmp3[i1][4],tmp3[i1][5])
        f.write('Xx %8.6f %8.6f %8.6f # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
        ((a4[0]+a4[1]*TAU)/float(a4[2]),(a5[0]+a5[1]*TAU)/float(a5[2]),(a6[0]+a6[1]*TAU)/float(a6[2]),\
        tmp3[i1][0][0],tmp3[i1][0][1],tmp3[i1][0][2],\
        tmp3[i1][1][0],tmp3[i1][1][1],tmp3[i1][1][2],\
        tmp3[i1][2][0],tmp3[i1][2][1],tmp3[i1][2][2],\
        tmp3[i1][3][0],tmp3[i1][3][1],tmp3[i1][3][2],\
        tmp3[i1][4][0],tmp3[i1][4][1],tmp3[i1][4][2],\
        tmp3[i1][5][0],tmp3[i1][5][1],tmp3[i1][5][2]))
    f.close
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray middle_position(np.ndarray[DTYPE_int_t, ndim=2] pos1,
                                 np.ndarray[DTYPE_int_t, ndim=2] pos2):
    cdef int i1
    cdef long w1,w2,w3
    cdef np.ndarray[DTYPE_int_t,ndim=1] tmp1a,tmp1b
    
    for i1 in range(6):
        w1,w2,w3=add(pos1[i1][0],pos1[i1][1],pos1[i1][2],pos2[i1][0],pos2[i1][1],pos2[i1][2])
        w1,w2,w3=mul(w1,w2,w3,1,0,2)
        tmp1b=np.array([w1,w2,w3])
        if i1!=0:
            tmp1a=np.append(tmp1a,tmp1b)
        else:
            tmp1a=tmp1b
    return tmp1a.reshape(6,3)
