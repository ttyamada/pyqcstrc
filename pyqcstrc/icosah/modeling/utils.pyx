import sys
import numpy as np
cimport numpy as np
cimport cython

from pyqcstrc.icosah.modeling.math1 cimport det_matrix, projection, add, sub, mul, div
from pyqcstrc.icosah.modeling.numericalc cimport obj_volume_6d_numerical

DTYPE_double = np.float64
DTYPE_int = int

cdef np.float64_t TAU=(1+np.sqrt(5))/2.0
cdef np.float64_t TOL=1e-6 # tolerance

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
        print('shift_object()')
        if vorbose>1:
            if vorbose>0:
                print(' volume: %d %d %d (%10.8f)'%(v0,v1,v2,vol1))
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

