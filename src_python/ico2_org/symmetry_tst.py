#if __name__ == '__main__':
import sys
#sys.path.append('.')
from ico2.math1 import (add, 
                                matrixpow, 
                                dot_product, 
                                dot_product_1, 
                                sub_vectors, 
                                add_vectors,
                                mul_vectors,
                                )
from ico2.utils import (remove_doubling_in_perp_space, 
                                remove_doubling,
                                )
from ico2.numericalc import (length_numerical,
                                     numerical_vector,
                                     numeric_value,
                                     )
import numpy as np

import random
from pyqcstrc.ico2.numericalc import (projection_par_numerical,
                                        numerical_vectors,
                                        numerical_vector,
                                        numeric_value,
                                        )
from pyqcstrc.ico2.math1 import (projection,
                                )
# test

def generate_random_value():
    """ generate value in TAU-style
    """
    nmax=10
    v=np.zeros((3),dtype=np.int64)
    for i1 in range(2):
        v[i1]=random.randrange(-nmax,nmax) # a and b in (a+b*TAU)/c.
    v[2]=random.randrange(1,nmax) # c in (a+b*TAU)/c.
    return v
    
def generate_random_vector(ndim=6):
    """ generate ndim vector in TAU-style
    ndim: dimension of vectors
    """
    nmax=10
    v=np.zeros((ndim,3), dtype=np.int64)
    for i1 in range(ndim):
        v[i1]=generate_random_value()
    return v
    
def generate_random_vectors(n,ndim=6):
    """
    num: number of generated vectors.
    ndim: dimension of vectors
    """
    v=np.zeros((n,ndim,3), dtype=np.int64)
    for i1 in range(n):
        v[i1]=generate_random_vector(ndim)
    return v

def generate_random_tetrahedron():
    return generate_random_vectors(4)

cen0=V0

#-----------------------------------------------
# TEST: symop_vec()
#-----------------------------------------------
"""
print("TEST: symop_vec()")
symop=icosasymop()
vt=generate_random_vector()
counter=0
for sop in symop:
    #
    # calc using symop_vec
    svt=symop_vec(sop,vt,cen0)
    svn1=numerical_vector(svt)
    #print(svn1)
    #
    # calc using no.dot with float values
    vn=numerical_vector(vt)
    svn2=np.dot(sop,vn)
    #print(svn2)
    if np.allclose(svn1,svn2):
        pass
    else:
        counter+=1
if counter==0:
    print('symop_vec: correct')
else:
    print('symop_vec: worng')
    
nset=4
vts=generate_random_vectors(nset)
print(vts)
svts=symop_vecs(symop[1],vts,cen0)
print(svts)
"""

#-----------------------------------------------
# TEST: icosasymop_array and icosasymop3_array
##-----------------------------------------------
"""
print('TEST: icosasymop_array and icosasymop3_array')

flag=None
#flag='axial'
op3 = icosasymop3_array(flag)
op6 = icosasymop_array()
vt=np.array([[1,0,3],[0,1,4],[1,0,5],[0,1,6],[1,0,7],[0,1,8]],dtype=np.int64)
vn=numerical_vector(vt)
vn=projection_par_numerical(vn)
counter=0
for i1 in range(120):
    a=np.dot(op3[i1],vn)
    b=symop_vec(op6[i1],vt,cen0)
    bn=numerical_vector(b)
    bn=projection_par_numerical(bn)
    #print('%d'%(i1))
    #print(' a:',a)
    #print(' b:',bn)
    if np.allclose(a,bn):
        pass
    else:
        counter+=1
if counter==0:
    print('ok')
else:
    print('wrong!')
"""

##-----------------------------------------------
# TEST: equivalent_positions()
##-----------------------------------------------
"""
brv='p'
#brv='f'
#brv='i'
#site=POS_V
#site=POS_C1
#site=POS_C2
site=POS_EC

eqpos=equivalent_positions(site,brv)
for i,p in enumerate(eqpos):
    print(i,numerical_vector(p))
print('len(eqpos):',len(eqpos))
"""



##-----------------------------------------------
# TEST: site_symmetry_and_coset()
##-----------------------------------------------
"""

symop=icosasymop_array()

print('TEST: site_symmetry_and_coset()')
#site=POS_V
#site=POS_C1
#site=POS_C2
site=POS_EC

print('site:')
print(site)
"""
"""
print('P-type icosahedral lattice')
brv='p'
idx_ssym,idx_coset=site_symmetry_and_coset(site,brv,verbose=1)
print('idx_ssym:',idx_ssym)
print('idx_coset:',idx_coset)
"""

"""
flag=0
eq_sites=equivalent_sites_unit_cell(site,idx_coset,brv,flag)
print(len(eq_sites))
for i1,vt in enumerate(eq_sites):
    vn=numerical_vector(vt)
    print(' %d site: %3.2f %3.2f %3.2f %3.2f %3.2f %3.2f'%(i1,vn[0],vn[1],vn[2],vn[3],vn[4],vn[5]))
print('number of euivalent site in unit cell:',len(eq_sites))
"""

"""
print('F-type icosahedral lattice')
brv='f'
idx_ssym,idx_coset=site_symmetry_and_coset(site,brv,verbose=1)
print('idx_ssym:',idx_ssym)
print('idx_coset:',idx_coset)

print('I-type icosahedral lattice')
brv='i'
idx_ssym,idx_coset=site_symmetry_and_coset(site,brv,verbose=1)
print('idx_ssym:',idx_ssym)
print('idx_coset:',idx_coset)

"""

"""
vts=generator_equivalent_vec(site,V0)
for i1,vt in enumerate(vts):
    vn=numerical_vector(vt)
    print(' %d site: %3.2f %3.2f %3.2f %3.2f %3.2f %3.2f'%(i1,vn[0],vn[1],vn[2],vn[3],vn[4],vn[5]))
    

a=get_index_of_symmetry_operation_for_equivalent_vectors(site)
print(a)
"""

"""
brv='p'
idx_ssym,idx_coset=site_symmetry_and_coset(site,brv,verbose=1)
print('idx_ssym:',idx_ssym)
print('idx_coset:',idx_coset)
"""

##-----------------------------------------------
# TEST: icosasymop3_array()
##-----------------------------------------------

"""
vt=np.array([[1,0,1],[0,1,2],[-1,0,3],[0,-1,4],[1,0,5],[0,1,6]]) 
tmp=projection(vt)[0]
vn=numerical_vector(tmp)
lst=[]
for i in range(120):
    lst.append(i)
"""

#"""
sop=icosasymop3_array()
counter=0
for m in range(2): # 2, inversion
    for l in range(3): # 3, c3
        for k in range(2): # 2, c2
            for j in range(2): # 2, c2'
                for i in range(5): # 5, c5
                    print('(%d)'%counter,m,l,k,j,i)
                    counter+=1

lst=[0,108,22,92,119,16,14,111,98,28,102,5]
for i in lst:
    print(np.linalg.det(sop[i]))

#"""

"""
vt=np.array([[-1,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]]) # along 5fold axis
tmp=projection(vt)[0]
vn=numerical_vector(tmp)
lst=[0,108,22,92,119,16,14,111,98,28,102,5]
"""
"""
# symmetry operation on normal vectors
flag='normal'
ops=icosasymop3_array(flag)
ops6=icosasymop_array_1(flag)
print(flag)
for i,j in enumerate(lst):
    vn1=ops[j]@vn
    vt1=symop_vec(ops6[j],vt,V0)
    vt2=projection(vt1)[0]
    vn2=numerical_vector(vt2)
    if np.allclose(vn1,vn2):
        print('%d %8.6f %8.6f %8.6f'%(i,vn1[0],vn1[1],vn1[2]))
    else:
        print('%d %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f'%(i,vn1[0],vn1[1],vn1[2],vn2[0],vn2[1],vn2[2]))
    

# symmetry operation on axial vectors
flag='axial'
ops=icosasymop3_array(flag)
ops6=icosasymop_array_1(flag)
print(flag)
for i,j in enumerate(lst):
    vn1=ops[j]@vn
    vt1=symop_vec(ops6[j],vt,V0)
    vt2=projection(vt1)[0]
    vn2=numerical_vector(vt2)
    if np.allclose(vn1,vn2):
        print('%d %8.6f %8.6f %8.6f'%(i,vn1[0],vn1[1],vn1[2]))
    else:
        print('%d %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f'%(i,vn1[0],vn1[1],vn1[2],vn2[0],vn2[1],vn2[2]))

vts=symmetry_operations_axial_vector(vt,V0,lst)
for i,v in enumerate(vts):
    tmp=projection(v)[0]
    vn1=numerical_vector(tmp)
    print('%d %8.6f %8.6f %8.6f'%(i,vn1[0],vn1[1],vn1[2]))

lst=[0,1]
vtss=symmetry_operations_axial_vectors(vts,V0,lst)
for i1,vts in enumerate(vtss):
    for i2,v in enumerate(vts):
        tmp=projection(v)[0]
        vn1=numerical_vector(tmp)
        print('%d %d %8.6f %8.6f %8.6f'%(i1,i2,vn1[0],vn1[1],vn1[2]))
"""
"""
#-----------------------------------------------
# TEST: icosasymop_array_2 and icosasymop3_array_2
##-----------------------------------------------

print('TEST: icosasymop_array and icosasymop3_array')

flag=None
#flag='axial'
op3 = icosasymop3_array_2(flag)
op6 = icosasymop_array_2()
vt=np.array([[1,0,3],[0,1,4],[1,0,5],[0,1,6],[1,0,7],[0,1,8]],dtype=np.int64)
vn=numerical_vector(vt)
vn=projection_par_numerical(vn)
counter=0
for i1 in range(120):
    a=op3[i1]@vn
    b=symop_vec(op6[i1],vt,cen0)
    bn=numerical_vector(b)
    bn=projection_par_numerical(bn)
    print('%d'%(i1))
    print(' a:',a/np.linalg.norm(a))
    print(' b:',bn/np.linalg.norm(bn))
    if np.allclose(a,bn):
        pass
    else:
        counter+=1
if counter==0:
    print('ok')
else:
    print('wrong!')
"""

"""
counter=0
for i in range(5):
    for j in range(2):
        for k in range(2):
            for l in range(3):
                for m in range(2):
                    print(counter,i,j,l,m)
                    counter+=1

vt=np.array([[1,0,1],[1,0,2],[1,0,3],[1,0,4],[1,0,5],[1,0,6]],dtype=np.int64) # general 6d vector

symop=icosasymop_array()
eqpos=np.zeros((len(symop),6,3),dtype=np.int64)
for i,op in enumerate(symop):
    v=symop_vec(op,vt,centre=V0)
    eqpos[i]=v
    print(i,numerical_vector(v))
eqpos1=remove_doubling(eqpos) 
print(len(eqpos1))
for v in eqpos1:
    print(v)
"""