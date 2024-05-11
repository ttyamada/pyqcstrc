#
# PyQCdiff - Python library for Quasi-Crystal diffraction
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#

import timeit
import os
import sys
import numpy as np

try:
    import pyqcdiff.genref as genref

except ImportError:
    print('import error\n')

TAU=(1+np.sqrt(5))/2.0

def read_ref_data(file):
    try:
        f=open(file,'r')
    except IOError, e:
        print e
        sys.exit(0)
    
    line=[]
    while 1:
        a=f.readline()
        # print a
        if not a:
            break

        line.append(a[:-1])

    return line
    

def list generate_refection_list_ico(hmax, qlimit, qimax, mode, psmatrix, tmpf):
    """
    generate 6d reflection list in Qpar up to qlimit in Qperp up to qimax in r.l.u.
    mode: 0 in Elser's setting (ref: Yamamoto-1997)
          1 in Cahn Shechtman Gratias's setting (1986)
    lphason: phason strain matrix
    """
    if lphason.size==4:
        flag=1
    else:
        flag==0
    for h1 in range(-hmax,hmax+1):
        for h2 in range(-hmax,hmax+1):
            for h3 in range(-hmax,hmax+1):
                for h4 in range(-hmax,hmax+1):
                    for h5 in range(-hmax,hmax+1):
                        for h6 in range(-hmax,hmax+1):
                            if mode==0:
                                qperp=projection_qperp_ico_elser([h1,h2,h3,h4,h5,h6])
                                qpar=projection_qpar_ico_elser([h1,h2,h3,h4,h5,h6])
                                if flag==0:
                                    pass
                                else:
                                    dq=lphason(psmatrix,qperp)
                                    qpar=[qpar[0]+dq[0],qpar[1]+dq[1],qpar[2]+dq[2]]
                            else:
                                qpar=projection_qpar_ico_csg([h1,h2,h3,h4,h5,h6])
                                qperp=projection_qperp_ico_csg([h1,h2,h3,h4,h5,h6])
                                if flag==0:
                                    pass
                                else:
                                    dq=lphason(psmatrix,qperp)
                                    qpar=[qpar[0]+dq[0],qpar[1]+dq[1],qpar[2]+dq[2]]
                            else:
                                qpar=[0.,0.,0.] # dummy
                                qperp=[0.,0.,0.]
                            q1=norm(qpar)
                            q2=norm(qperp)
                            if q1<=qlimit and q2<=qimax:
                                print('%4d%4d%4d%4d%4d%4d'%(h1,h2,h3,h4,h5,h6))
                            else:
                                pass
    return 0
    
genref.generate_refection_list_ico
    
if __name__ == '__main__':

