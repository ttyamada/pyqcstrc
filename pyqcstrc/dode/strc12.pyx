#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
import numpy as np
cimport numpy as np
cimport cython

from pyqcstrc.dode.numericalc12 cimport projection_numerical,inout_occupation_domain_numerical,inout_occupation_domain_phason_numerical
from pyqcstrc.dode.numericalc12 cimport projection_1_numerical, get_internal_component_numerical, projection_numerical_phason_1
from pyqcstrc.dode.numericalc12 cimport projection_2_numerical

DTYPE_double = np.float64
#DTYPE_int = int
DTYPE_int = np.int64
#ctypedef np.int64_t DTYPE_int_t
#ctypedef np.float64_t DTYPE_double_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray strc(list objlst,
                        list poslst,
                        np.ndarray[double, ndim=2] pmatrx,
                        int nmax,
                        list eshift,
                        list shift,
                        int verbose):
    
    cdef int h1,h2,h3,h4,flg
    cdef double v1,v2,v3,v4,v5,v6
    cdef np.ndarray[DTYPE_int_t, ndim=2] hindex,pos
    cdef list lst,v,w,shft
    cdef np.ndarray[DTYPE_int_t, ndim=4] obj
    lst=[]
    if pmatrx.tolist()==[[0.0]]:
        flg=0
        shft=projection_1_numerical(shift[0],shift[1],shift[2],shift[3],shift[4],shift[5])
    else:
        flg=1
        shft=projection_2_numerical(shift[0],shift[1],shift[2],shift[3],shift[4],shift[5],pmatrx[0][0],pmatrx[0][1],pmatrx[1][0],pmatrx[1][1])
    
    for h1 in range(-nmax,nmax+1):
        if verbose>0:
            print('%d'%(h1))
        else:
            pass
        for h2 in range(-nmax,nmax+1):
            for h3 in range(-nmax,nmax+1):
                for h4 in range(-nmax,nmax+1):
                    for i1 in range(len(objlst)):
                        obj=objlst[i1]
                        pos=poslst[i1]
                        xe=eshift[i1]
                        #hindex=np.array([[h1,0,1],[h2,0,1],[h3,0,1],[h4,0,1],[0,0,1],[0,0,1]])
                        if flg==0:
                            #v1,v2,v3,v4,v5,v6=projection_numerical(hindex)
                            w=projection_numerical(pos)
                            shfte=projection_numerical(xe)
                            v=projection_1_numerical(float(h1),float(h2),float(h3),float(h4),0.0,0.0)
                            #
                            if inout_occupation_domain_numerical(obj,np.array(v[3:6])-np.array([shft[3],shft[4],shft[5]]),verbose-1)==0: # inside
                                #lst.append([v[0]+w[0]+shfte[0],v[1]+w[1]+shfte[1],v[2]+w[2]+shfte[2],i1,v[3]+w[3]+shfte[3],v[4]+w[4]+shfte[4],v[5]+w[5]+shfte[5]])
                                #lst.append([v[0]-w[0]-shfte[0],v[1]-w[1]-shfte[1],v[2]-w[2]-shfte[2],i1,v[3]-w[3]-shfte[3],v[4]-w[4]-shfte[4],v[5]-w[5]-shfte[5]])
                                lst.append([v[0]-w[0]-shfte[0],v[1]-w[1]-shfte[1],v[2]-w[2]-shfte[2],i1,h1,h2,h3,h4])
                            else:
                                pass
                        else:
                            #v1,v2,v3,v4,v5,v6=projection_numerical_phason(hindex,pmatrx)
                            #w=projection_numerical(pos)
                            w=projection_numerical_phason_1(pos,pmatrx)
                            #shfte=projection_numerical(xe)
                            shfte=projection_numerical_phason_1(xe,pmatrx)
                            v=projection_2_numerical(float(h1),float(h2),float(h3),float(h4),0.0,0.0,pmatrx[0][0],pmatrx[0][1],pmatrx[1][0],pmatrx[1][1])
                            #
                            # Occupation domain is not deformed by phason strain matrix,
                            if inout_occupation_domain_numerical(obj,np.array(v[3:6])-np.array([shft[3],shft[4],shft[5]]),verbose-1)==0: # inside 
                            #
                            # Occupation domain is alos deformed by phason strain matrix.
                            #if inout_occupation_domain_phason_numerical(obj, np.array(v[3:6])-np.array([shft[3],shft[4],shft[5]]), pmatrx, verbose-1)==0: # inside
                                #lst.append([v[0]+w[0]+shfte[0],v[1]+w[1]+shfte[1],v[2]+w[2]+shfte[2],i1,v[3]+w[3]+shfte[3],v[4]+w[4]+shfte[4],v[5]+w[5]+shfte[5]])
                                #lst.append([v[0]-w[0]-shfte[0],v[1]-w[1]-shfte[1],v[2]-w[2]-shfte[2],i1,v[3]-w[3]-shfte[3],v[4]-w[4]-shfte[4],v[5]-w[5]-shfte[5]])
                                lst.append([v[0]-w[0]-shfte[0],v[1]-w[1]-shfte[1],v[2]-w[2]-shfte[2],i1,h1,h2,h3,h4])
                            else:
                                pass
                        #if inout_occupation_domain_numerical(obj,np.array(v[3:6])+np.array(w[3:6])+np.array([shft[3],shft[4],shft[5]]),verbose-1)==0: # inside
                        #if inout_occupation_domain_numerical(obj,np.array(v[3:6])-np.array([shft[3],shft[4],shft[5]]),verbose-1)==0: # inside
                            #lst.append([v[0]+w[0]+shft[0],v[1]+w[1]+shft[1],v[2]+w[2]+shft[2],i1])
                            #lst.append([v[0]-w[0]+shfte[0],v[1]-w[1]+shfte[1],v[2]-w[2]+shfte[2],i1])
                            #lst.append([v[0]-w[0]-shfte[0],v[1]-w[1]-shfte[1],v[2]-w[2]-shfte[2],i1,v[3]-w[3]-shfte[3],v[4]-w[4]-shfte[4],v[5]-w[5]-shfte[5]])
                            #break
                        #else:
                        #    pass
    return np.array(lst)
"""
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray strc(np.ndarray[DTYPE_int_t, ndim=4] obj,
                        np.ndarray[DTYPE_int_t, ndim=3] pmatrx,
                        int nmax,
                        list ishift,
                        int verbose):
    
    cdef int h1,h2,h3,h4,flg
    cdef double v1,v2,v3,v4,v5,v6
    cdef np.ndarray[DTYPE_int_t, ndim=2] hindex
    cdef list lst, v
    lst=[]
    if pmatrx.tolist()==[[[0]]]:
        flg=0
    else:
        flg=1
    for h1 in range(-nmax,nmax+1):
        for h2 in range(-nmax,nmax+1):
            for h3 in range(-nmax,nmax+1):
                for h4 in range(-nmax,nmax+1):
                    #hindex=np.array([[h1,0,1],[h2,0,1],[h3,0,1],[h4,0,1],[0,0,1],[0,0,1]])
                    if flg==0:
                        #v1,v2,v3,v4,v5,v6=projection_numerical(hindex)
                        #v=projection_numerical(hindex)
                        v=projection_1_numerical(float(h1),float(h2),float(h3),float(h4),0.0,0.0)
                    else:
                        #v1,v2,v3,v4,v5,v6=projection_numerical_phason(hindex,pmatrx)
                        #v=projection_numerical_phason(hindex,pmatrx)
                        pass
                    if inout_occupation_domain_numerical(obj,np.array(v[3:6])-np.array([ishift[0],ishift[1],0]),verbose-1)==0: # inside
                        lst.append(v[0:3])
                    else:
                        pass
    return np.array(lst)
"""