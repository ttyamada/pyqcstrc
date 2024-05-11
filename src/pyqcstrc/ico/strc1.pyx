#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
import numpy as np
cimport numpy as np
cimport cython

from pyqcstrc.ico.numericalc cimport inside_outside_obj

from pyqcstrc.ico.numericalc cimport projection_numerical,inside_outside_obj,projection_numerical_phason
from pyqcstrc.ico.numericalc cimport get_internal_component_numerical



DTYPE_double = np.float64
DTYPE_int = int

cdef atomic_structure_factor(rlv,Z):
    """
    VESTA manual, p112
    D. Waasmaier and A. Kirfel, Acta Crystallogr., Sect. A: Found. Crystallogr., 51, 416 (1995).
    """
    if Z==21:
        """
               a1         a2         a3         a4         a5         c         
               b1         b2         b3         b4         b5         bc        
        Sc:  1.476566   1.487278   1.600187   9.177463   7.099750   0.157765
            53.131022   0.035325 137.319495   9.098031   0.602102  12.290000
        """
        (a1,a2,a3,a4,a5,c)=(1.476566,1.487278,1.600187,9.177463,7.099750,0.157765)
        (b1,b2,b3,b4,b5)=(53.131022,0.035325,137.319495,9.098031,0.602102)

    if Z==30:
        """
            a1         a2         a3         a4         a5         c         
            b1         b2         b3         b4         b5         bc
        Zn: 14.741002   6.907748   4.642337   2.191766  38.424042 -36.915828
             3.388232   0.243315  11.903689  63.312130   0.000397   5.600000
        """
        (a1,a2,a3,a4,a5,c)=(14.741002,   6.907748,   4.642337,   2.191766,  38.424042, -36.915828)
        (b1,b2,b3,b4,b5)=(3.388232,   0.243315,  11.903689,  63.312130,   0.000397)
    
    afac=a1*math.exp(-b1*rlv**2)
    afac+=a2*math.exp(-b2*rlv**2)
    afac+=a3*math.exp(-b3*rlv**2)
    afac+=a4*math.exp(-b4*rlv**2)
    afac+=a5*math.exp(-b5*rlv**2)
    afac+=c
    
    return afac
    
cpdef np.ndarray strcfac(list obj,
                            float lat
                            int h1,
                            int h2,
                            int h3,
                            int h4,
                            int h5,
                            int h6,
                            int verbose):
    
    q=1
    return 0

cpdef fourier_integral_tetrahedron(h):
    """
    Fourier integral of a tetrahedron
    """
    
    return 0
    
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray strc(list objlst,
                        list poslst,
                        np.ndarray[DTYPE_int_t, ndim=3] pmatrx,
                        int nmax,
                        list eshift,
                        list shift,
                        int verbose):
    
    cdef int i1,i2
    cdef int h1,h2,h3,h4,h5,h6,flg
    cdef double v1,v2,v3,v4,v5,v6
    cdef np.ndarray[DTYPE_int_t, ndim=2] hindex,pos
    cdef list lst,v,w,shft
    cdef np.ndarray[DTYPE_int_t, ndim=4] obj
    lst=[]
    if pmatrx.tolist()==[[[0]]]:
        flg=0
    else:
        flg=1
    
    shft=projection_numerical(shift[0],shift[1],shift[2],shift[3],shift[4],shift[5])
    for h1 in range(-nmax,nmax+1):
        if verbose>0:
            print('%d'%(h1))
        else:
            pass
        for h2 in range(-nmax,nmax+1):
            for h3 in range(-nmax,nmax+1):
                for h4 in range(-nmax,nmax+1):
                    for h5 in range(-nmax,nmax+1):
                        for h6 in range(-nmax,nmax+1):
                            for i1 in range(len(objlst)):
                                
                                obj=objlst[i1]
                                pos=poslst[i1]
                                xe=eshift[i1]
                                
                                # symmetry
                                for i2 in range(120):

                                if flg==0:
                                        w=projection_numerical(pos)
                                        shfte=projection_numerical(xe)
                                        v=projection_numerical(float(h1),float(h2),float(h3),float(h4),float(h5),float(h6))
                                    else:
                                        #v1,v2,v3,v4,v5,v6=projection_numerical_phason(hindex,pmatrx)
                                        #v=projection_numerical_phason(hindex,pmatrx)
                                        pass

                                    if inside_outside_obj(np.array(v[3:6])-np.array([shft[3],shft[4],shft[5]]),obj)==0: # inside
                                        lst.append([v[0]-w[0]+shfte[0],v[1]-w[1]+shfte[1],v[2]-w[2]+shfte[2],i1])
                                    else:
                                        pass
                                    
                                    
                                    
                                
                                
    return np.array(lst)
    
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray get_points_inside_obj(np.ndarray[DTYPE_int_t, ndim=4] obj,
                                        double step,
                                        int nstep):
    """
    """
    cdef int i1,i2,i3
    cdef double x,y,z
    cdef np.ndarray[DTYPE_int_t, ndim=2] hindex
    cdef list lst, v
    lst=[]

    if pmatrx.tolist()==[[[0]]]:
        flg=0
    else:
        flg=1
    for i1 in range(0,nstep+1):
        x=0.0+i1*step
        for i2 in range(0,nstep+1):
            y=0.0+i2*step
            for i3 in range(0,nstep+1):
                z=0.0+i3*step
                if inside_outside_obj([x,y,z],obj)==0:
                    print('%8.6f %8.6f %8.6f'%(x,z,y))
                else:
                    pass
    return 0

