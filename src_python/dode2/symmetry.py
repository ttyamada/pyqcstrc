#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
import itertools
from dode2.math1 import (add, 
                        matrixpow, 
                        dot_product, 
                        dot_product_1, 
                        sub_vectors, 
                        add_vectors,
                        )
from dode2.utils import (remove_doubling_in_perp_space, 
                        remove_doubling,
                        )
from dode2.numericalc import (projection_numerical,
                            projection3_numerical,
                            numerical_vector,
                            length_numerical,
                            )
import numpy as np

EPS=1e-6
DTYPE_int = int
#DTYPE_int = DTYPE_int

V0=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=DTYPE_int)

def symop_obj(symop,obj,centre=V0):
    """ Apply a symmetric operation on an object around given centre. in TAU-style
    
    """
    ndim=obj.ndim
    if ndim==3:
        return symop_vecs(symop,obj,centre)
    elif ndim==4:
        obj1=np.zeros(obj.shape,dtype=DTYPE_int)
        i=0
        for vts in obj:
            obj1[i]=symop_vecs(symop,vts,centre)
            i+=1
        return obj1
    elif ndim==2:
        return symop_vec(symop,obj,centre)
    else:
        print('object has an incorrect shape!')
        return 

def symop_vecs(symop,vts,centre=V0):
    """ Apply a symmetric operation on set of vectors around given centre. in TAU-style
    
    """
    out=np.zeros(vts.shape,dtype=DTYPE_int)
    i=0
    for vt in vts:
        out[i]=symop_vec(symop,vt,centre)
        i+=1
    return out

def symop_vec(symop,vt,centre=V0):
    """ Apply a symmetric operation on a vector around given centre. in TAU-style
    """
    vt=sub_vectors(vt,centre)
    vt=dot_product_1(symop,vt)
    return add_vectors(vt,centre)

def generator_obj_symmetric_obj(obj,centre=V0,pg='-12m2'):
    """
    """
    if obj.ndim==3 or obj.ndim==4:
        if np.all(centre==V0):
            mop=dodesymop_array(pg)
        else:
            lst_site_symmetry,__=site_symmetry_and_coset(centre,'p',pg,verbose=0)
            mop=[]
            tmp=dodesymop_array(pg)
            for i in lst_site_symmetry:
                mop.append(tmp[i])
        num=len(mop)
        shape=tuple([num])
        a=np.zeros(shape+obj.shape,dtype=DTYPE_int)
        for i,op in enumerate(mop):
            a[i]=symop_obj(op,obj,centre)
        if obj.ndim==4:
            n1,n2,_,_=obj.shape
            a=a.reshape(num*n1,n2,6,3)
        return a
    else:
        print('object has an incorrect shape!')
        return

def generator_obj_symmetric_triangle(obj,centre=V0,pg='-12m2'):
    """
    """
    return generator_obj_symmetric_obj(obj,centre,pg)
    
#def generator_obj_symmetric_tetrahedron(obj,centre):
#    return generator_obj_symmetric_obj(obj,centre)

def generator_obj_symmetric_vector_specific_symop(obj,centre,index_of_symmetry_operation,pg='-12m2'):
    """
    vector: triangles
    (6,3)
    """
    # using specific symmetry operations
    if obj.ndim==2:
        #mop=dodesymop(pg)
        mop=dodesymop_array(pg)
        shape=tuple([len(index_of_symmetry_operation)])
        a=np.zeros(shape+obj.shape,dtype=DTYPE_int)
        j=0
        for i1 in index_of_symmetry_operation:
            a[j]=symop_obj(mop[i1],obj,centre)
            j+=1
        return a
    else:
        print('object has an incorrect shape!')
        return

def generator_obj_symmetric_triangle_specific_symop(obj,centre,index_of_symmetry_operation,pg='-12m2'):
    """
    triangle: triangles
    (3,6,3)
    """
    # using specific symmetry operations
    if obj.ndim==3:
        #mop=dodesymop(pg)
        mop=dodesymop_array(pg)
        shape=tuple([len(index_of_symmetry_operation)])
        a=np.zeros(shape+obj.shape,dtype=DTYPE_int)
        j=0
        for i1 in index_of_symmetry_operation:
            a[j]=symop_obj(mop[i1],obj,centre)
            j+=1
        return a
    else:
        print('object has an incorrect shape!')
        return

def generator_obj_symmetric_obj_specific_symop(obj,centre,index_of_symmetry_operation,pg='-12m2'):
    """
    obj: a set of triangles
    (n,3,6,3)
    """
    # using specific symmetry operations
    if obj.ndim==4:
        #mop=dodesymop(pg)
        mop=dodesymop_array(pg)
        shape=tuple([len(index_of_symmetry_operation)])
        a=np.zeros(shape+obj.shape,dtype=DTYPE_int)
        j=0
        for i1 in index_of_symmetry_operation:
            a[j]=symop_obj(mop[i1],obj,centre)
            j+=1
        n1,n2,_,_,_=a.shape
        #return a.reshape(int(len(a)/54),3,6,3)
        return a.reshape(n1*n2,3,6,3)
    else:
        print('object has an incorrect shape!')
        return
    
def generator_obj_symmetric_triangle_0(obj,centre,symmetry_operation_index,pg='-12m2'):
    """
    """
    #mop=dodesymop(pg)
    mop=dodesymop_array(pg)
    return symop_obj(mop[symmetry_operation_index],obj,centre)

def generator_obj_symmetric_vec(vectors,centre,pg='-12m2'):
    """
    """
    return generator_obj_symmetric_obj(vectors,centre,pg)

def generator_equivalent_vectors(vectors,centre,pg='-12m2'):
    """
    """
    a=generator_obj_symmetric_obj(vectors,centre,pg)
    return remove_doubling_in_perp_space(a)

def generator_equivalent_vec(vector,centre,pg='-12m2'):
    """
    """
    a=generator_obj_symmetric_obj(vector,centre,pg)
    return remove_doubling(a)

def dodesymop(pg='-12m2'):
    """
    pg: point group, '12/mmm', '-12m2'
    """
    
    #"""
    # dodecagonal symmetry operations
    # c12
    # y, z, u, −x + z, v,
    m1=np.array([[0, 1, 0, 0, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 1, 0, 0],\
                [-1, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 0, 1, 0],\
                [ 0, 0, 0, 0, 0, 1]],dtype=DTYPE_int)
    # mirror normal to y-axis
    # x,y − u,x − z,−u,−v
    m2=np.array([[1, 0, 0, 0, 0, 0],\
                [ 0, 1, 0,-1, 0, 0],\
                [ 1, 0,-1, 0, 0, 0],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0, 0, 0,-1, 0],\
                [ 0, 0, 0, 0, 0, 1]],dtype=DTYPE_int)
    # mirror normal to z-axis
    # x,y,z,u,−v
    m3=np.array([[1, 0, 0, 0, 0, 0],\
                [ 0, 1, 0, 0, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 1, 0, 0],\
                [ 0, 0, 0, 0,-1, 0],\
                [ 0, 0, 0, 0, 0, 1]],dtype=DTYPE_int)
    # inversion
    # -x,-y,-z,-u,−v
    m4=np.array([[-1, 0, 0, 0, 0, 0],\
                [ 0,-1, 0, 0, 0, 0],\
                [ 0, 0,-1, 0, 0, 0],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0, 0, 0,-1, 0],\
                [ 0, 0, 0, 0, 0, 1]],dtype=DTYPE_int)
    # s12
    # IR12: -y, -z, -u, x-z, -v
    m5=np.dot(m4,m1)
    
    symop=[]
    if pg=='12/mmm':
        for i1 in range(2):
            for i2 in range(2):
                for i3 in range(2):
                    for i4 in range(12):
                        s1=matrixpow(m1,i4) # c12
                        s2=matrixpow(m2,i3) # mirror,y
                        s3=matrixpow(m3,i2) # mirror,z
                        s4=matrixpow(m4,i1) # inversion
                        tmp=np.dot(s2,s1)
                        tmp=np.dot(s3,tmp)
                        tmp=np.dot(s4,tmp)
                        symop.append(tmp)
    elif pg=='-12m2':
        for i1 in range(2):
            for i2 in range(12):
                s1=matrixpow(m5,i2) # s12
                s2=matrixpow(m2,i1) # mirror,y
                tmp=np.dot(s2,s1)
                symop.append(tmp)
    elif pg=='-12':
        for i1 in range(12):
            s1=matrixpow(m5,i1) # s12
            symop.append(s1)
    elif pg=='12':
        for i1 in range(12):
            s1=matrixpow(m1,i1) # c12
            symop.append(s1)
    return symop
    
def dodesymop_array(pg='-12m2'):
    """
    """
    ops=matrix_dode_sym()
    num=0
    if pg=='12/mmm':
        m1=ops[0] # c12
        m2=ops[1] # mirror,y
        m3=ops[2] # mirror,z
        m4=ops[3] # inversion
        symop=np.zeros((96,6,6),dtype=DTYPE_int)
        for i1 in range(2):
            s4=matrixpow(m4,i1)
            for i2 in range(2):
                s3=matrixpow(m3,i2)
                for i3 in range(2):
                    s2=matrixpow(m2,i3)
                    for i4 in range(12):
                        s1=matrixpow(m1,i4)
                        tmp=np.dot(s2,s1)
                        tmp=np.dot(s3,tmp)
                        tmp=np.dot(s4,tmp)
                        symop[num]=tmp
                        num+=1
    elif pg=='12mm':
        m1=ops[0] # C12
        m2=ops[1] # mirror,y
        m3=ops[3] # inversion
        symop=np.zeros((48,6,6),dtype=DTYPE_int)
        for i1 in range(2):
            s3=matrixpow(m3,i1)
            for i2 in range(2):
                s2=matrixpow(m2,i2)
                for i3 in range(12):
                    s1=matrixpow(m1,i3)
                    tmp=np.dot(s2,s1)
                    tmp=np.dot(s3,tmp)
                    symop[num]=tmp
                    num+=1
    elif pg=='-12m2':
        m1=ops[4] # s12
        m2=ops[1] # mirror,y
        symop=np.zeros((24,6,6),dtype=DTYPE_int)
        for i1 in range(2):
            s2=matrixpow(m2,i1)
            for i2 in range(12):
                s1=matrixpow(m1,i2)
                tmp=np.dot(s2,s1)
                symop[num]=tmp
                num+=1
    elif pg=='-12':
        m1=ops[4] # s12
        symop=np.zeros((12,6,6),dtype=DTYPE_int)
        for i1 in range(12):
            s1=matrixpow(m1,i1)
            symop[num]=s1
            num+=1
    elif pg=='12':
        m1=ops[0] # c12
        symop=np.zeros((12,6,6),dtype=DTYPE_int)
        for i1 in range(12):
            s1=matrixpow(m1,i1)
            symop[num]=s1
            num+=1
    return symop
    
def matrix_dode_sym():
    # c12
    # y, z, u, −x + z, v,
    m1=np.array([[0, 1, 0, 0, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 1, 0, 0],\
                [-1, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 0, 1, 0],\
                [ 0, 0, 0, 0, 0, 1]],dtype=DTYPE_int)
    # mirror normal to y-axis
    # x,y − u,x − z,−u,−v
    m2=np.array([[1, 0, 0, 0, 0, 0],\
                [ 0, 1, 0,-1, 0, 0],\
                [ 1, 0,-1, 0, 0, 0],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0, 0, 0,-1, 0],\
                [ 0, 0, 0, 0, 0, 1]],dtype=DTYPE_int)
    # mirror normal to z-axis
    # x,y,z,u,−v
    m3=np.array([[1, 0, 0, 0, 0, 0],\
                [ 0, 1, 0, 0, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 1, 0, 0],\
                [ 0, 0, 0, 0,-1, 0],\
                [ 0, 0, 0, 0, 0, 1]],dtype=DTYPE_int)
    # inversion
    # -x,-y,-z,-u,−v
    m4=np.array([[-1, 0, 0, 0, 0, 0],\
                [ 0,-1, 0, 0, 0, 0],\
                [ 0, 0,-1, 0, 0, 0],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0, 0, 0,-1, 0],\
                [ 0, 0, 0, 0, 0, 1]],dtype=DTYPE_int)
    # s12
    # IR12: -y, -z, -u, x-z, -v
    #m5=np.array([[0,-1, 0, 0, 0, 0],\
    #            [ 0, 0,-1, 0, 0, 0],\
    #            [ 0, 0, 0,-1, 0, 0],\
    #            [ 1, 0,-1, 0, 0, 0],\
    #            [ 0, 0, 0, 0,-1, 0],\
    #            [ 0, 0, 0, 0, 0, 1]],dtype=DTYPE_int)
    m5=np.dot(m4,m1)
    symop=np.zeros((5,6,6),dtype=DTYPE_int)
    symop[0]=m1
    symop[1]=m2
    symop[2]=m3
    symop[3]=m4
    symop[4]=m5
    return symop

"""
def mattrix_dode_sym():
    # dodecagonal symmetry operations
    # c12
    m1=np.array([[ 0, 1, 0, 0, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 1, 0, 0],\
                [-1, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 0, 1, 0],\
                [ 0, 0, 0, 0, 0, 1]],dtype=DTYPE_int)
    # mirror
    m2=np.array([[ 0, 0, 0, 1, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 1, 0, 0, 0, 0],\
                [ 1, 0, 0, 0, 0, 0],\
                [ 0, 0, 0, 0, 1, 0],\
                [ 0, 0, 0, 0, 0, 1]],dtype=DTYPE_int)
    # inversion
    m3=np.array([[-1, 0, 0, 0, 0, 0],\
                [ 0,-1, 0, 0, 0, 0],\
                [ 0, 0,-1, 0, 0, 0],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0, 0, 0,-1, 0],\
                [ 0, 0, 0, 0, 0, 1]],dtype=DTYPE_int)
    symop=np.zeros((3,6,6),dtype=DTYPE_int)
    symop[0]=m1
    symop[1]=m2
    symop[2]=m3
    return symop
"""
    
def generator_symmetric_vec_specific_symop(vector,centre,index_of_symmetry_operation,pg='-12m2'):
    #mop=dodesymop(pg)
    mop=dodesymop_array(pg)
    #a=np.zeros(len(index_of_symmetry_operation))
    #return symop_vec(mop[symmetry_operation_index],vector,centre)
    return symop_vec(mop[index_of_symmetry_operation],vector,centre)
    
def translation(ndim):
    """translational symmetry
    """
    symop=[]
    lst=[-1,0,1]
    tmp=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
    symop.append(tmp)
    if ndim==4:
        for i1 in lst:
            for i2 in lst:
                for i3 in lst:
                    for i4 in lst:
                        tmp=np.array([[i1,0,1],[i2,0,1],[i3,0,1],[i4,0,1],[0,0,1],[0,0,1]])
                        symop.append(tmp)
    else:
        for i1 in lst:
            for i2 in lst:
                for i3 in lst:
                    for i4 in lst:
                        for i5 in lst:
                            tmp=np.array([[i1,0,1],[i2,0,1],[i3,0,1],[i4,0,1],[i5,0,1],[0,0,1]])
                            symop.append(tmp)
    return np.array(symop)

# new ver
def translation_new(brv,flag=0):
    """translational symmetry
    
    brv : bravais lattce p,
            s : superlattice for decagonal quasicrystal
    """
    if flag==1:
        lst=[0,1]
    elif flag==-1:
        lst=[-1,0]
    elif flag==-2:
        lst=[0]
    else:
        lst=[-1,0,1]
        #lst=[-2,0,2]
        
    tr=np.zeros((len(lst)**5,6,3),dtype=DTYPE_int)
    j1=0
    for i1 in lst:
        for i2 in lst:
            for i3 in lst:
                for i4 in lst:
                    for i5 in lst:
                        tr[j1]=np.array([[i1,0,1],[i2,0,1],[i3,0,1],[i4,0,1],[i5,0,1],[0,0,1]])
                        j1+=1
    
    if brv=='p':
        return tr
    else:
        print('no lattice type selected.')
        return 
    
################ 
# numeric
################
def generator_equivalent_numeric_vector_specific_symop(vn,index_of_symmetry_operation,pg='-12m2'):
    mop=dodesymop_array(pg)
    out=np.zeros((len(index_of_symmetry_operation),6),dtype=np.float64)
    for i1 in index_of_symmetry_operation:
        out[i1]=mop[i1]@vn
    return out
    
def generator_equivalent_numeric_vectors_specific_symop(vns,index_of_symmetry_operation,pg='-12m2'):
    mop=dodesymop_array(pg)
    num1=len(index_of_symmetry_operation)
    out=np.zeros((num1,len(vns),6),dtype=np.float64)
    for i1 in range(num1):
        for vn in vns:
            out[i1]=mop[i1]@vn
    return out
    
################ 
# site symmetry
################

def site_symmetry_and_coset(site,brv,pg,verbose=0):
    #symmetry operators in the site symmetry group G and its left coset decomposition.
    #
    #Args:
    #    site (numpy.ndarray):
    #        xyz coordinate of the site.
    #        The shape is (6,3).
    #
    #Returns:
    #    List of index of symmetry operators of the site symmetry group G (list):
    #        The symmetry operators leaves xyz identical.
    #    
    #    List of index of symmetry operators in the left coset representatives of the poibt group G (list):
    #        The symmetry operators generates equivalent positions of the site xyz.
    
    def site_symmetry(site,symop,brv):
        """symmetry operators in the site symmetry group G.
        
        Args:
            site (numpy.ndarray):
                xyz coordinate of the site.
                The shape is (6,3).
            
        Returns:
            List of index of symmetry operators of the site symmetry group G (list):
                The symmetry operators leaves xyz identical.
        """
        # サイト周りでvtgに対して対称操作を施す。
        vtg=np.array([[1,0,3],[0,1,4],[1,0,5],[0,1,6],[1,0,7],[0,0,1]],dtype=DTYPE_int)
        #vtg=add_vectors(vtg,site)
        a=np.zeros((len(symop),6,3),dtype=DTYPE_int)
        for i1,op in enumerate(symop):
            a[i1]=symop_vec(op,vtg,site)
            
        if brv=='p':
            flag=1
        else:
            pass
        traop=translation_new(brv,flag)
        lst=[]
        for i1,a1 in enumerate(a):
            # vtgに対して並進を含む全ての対称操作を施す。
            #print('%3d     a1:'%(i1),numerical_vector(a1))
            counter1=0
            for op in symop:
                tmp1=symop_vec(op,vtg,V0)
                if np.all(a1==tmp1):
                    counter1=1
                    #print('      tmp1:',numerical_vector(tmp1))
                    break
                else:
                    flag1=0
                    for tr in traop:
                        b=add_vectors(tmp1,tr)
                        if np.all(a1==b):
                            flag1=1
                            #print('         b:',numerical_vector(b))
                            break
                        else:
                            pass
                    if flag1==1:
                        counter1=1
                        break
                    else:
                        pass
            if counter1==1:
                lst.append(i1)
            else:
                pass
        print('lst:',lst)
        return lst
        
    def coset(site,symop,brv,pg,idx_site):
        """
        """
        # coordinate of equivalent sites
        pos_equiv=equivalent_positions(site,brv,pg)
        #print('  number of equivalent positions:',len(pos_equiv))
        #for xyz in pos_equiv:
        #    print(xyz)
        
        # site symmetry以外の対称操作のインデックスを収納したリストを作成（idx_else）
        a=set(range(len(symop)))
        b=set(idx_site)-{0}
        idx_else=list(a-b)
        
        # idx_elseの対称操作のうち、各等価サイトを作る対称操作を調べる
        tmp1=[]
        for pos in pos_equiv:
            #print('  pos:',pos)
            tmp=[]
            for idx in idx_else:
                #pos1=symop[idx]@site
                pos1=symop_vec(symop[idx],site,V0)
                if np.all(pos==pos1):
                    tmp.append(idx)
                    #break
                else:
                    pass
            tmp1.append(tmp)
            #print('  idx_coset',tmp)
            
        # いくつかある組み合わせのうち最初のものを選ぶ。
        idx_coset=[]
        for i in range(len(tmp1)):
            idx_coset.append((tmp1[i][0]))
            
        if check_coset(site,idx_coset,symop,idx_site):
            return idx_coset
        else:
            return 
            
    def check_coset(site,comb,symop,idx_site):
        """
        """
        #symop=symop_array()
        #list1=site_symmetry(site)
        
        list4=[]
        for i2 in comb:
            for i1 in idx_site: # i1-th symmetry operation of the site symmetry (point group, H)
                op1=symop[i2]@symop[i1]
                for i3,op in enumerate(symop):
                    if np.all(op==op1):
                        num=i3
                        break
                    else:
                        pass
                list4.append(num)
        c=remove_overlaps(list4)
        if len(c)==len(list4):
            return True
        else:
            return False
            
    def equivalent_positions(site,brv,pg):
        """
        siteに対して点群の対称性を施したサイトのうち、並進操作のみで結ばれない位置を求める。
        適切な名前を決める必要がある！！！
        """
        print('equivalent_positions()')
        print('  site:',numerical_vector(site))
        symop=dodesymop_array(pg)
        eqpos=np.zeros((len(symop),6,3),dtype=DTYPE_int)
        for i,op in enumerate(symop):
            eqpos[i]=symop_vec(op,site,centre=V0)
        eqpos1=remove_doubling(eqpos)
        print('  len(eqpos1)=',len(eqpos1))
        for _eqpos1 in eqpos1:
            print('  _eqpos1:',numerical_vector(_eqpos1))
        
        # 求めた等価なサイトのうち、並進操作を施して同一なのであれば、どちらか片方を選ぶようにする。
        if len(eqpos1)==1:
            lst_saved=[site]
        else:
            lst_saved=[site]
            translation=translation_new(brv,flag=0)
            for pos in eqpos1:
                counter=0
                for tr in translation:
                    pos1=add_vectors(pos,tr)
                    if np.all(site==pos1):
                        counter+=1
                        break
                    else:
                        pass
                if counter==0:
                    lst_saved.append(pos)
        print('  len(lst_saved)=',len(lst_saved))
        for _lst_saved in lst_saved:
            print('  _lst_saved:',numerical_vector(_lst_saved))
        
        
        # 求めたサイトのうち単位胞内にあるサイトを選ぶ
        out=np.zeros((len(lst_saved),6,3),dtype=DTYPE_int)
        num=0
        for vt in lst_saved:
            vn=numerical_vector(vt)
            #if np.all(vn>=0.0):# and np.all(vn<1.0):
            if np.all(vn>=0.0) and np.all(vn<=1.0):
                out[num]=vt
                print('  vt:',numerical_vector(vt))
                num+=1
            else:
                pass
        return out[:num]
        
        
    symop=dodesymop_array(pg)
    
    if verbose>0:
        vn=numerical_vector(site)
        print(' site: %4.3f %4.3f %4.3f %4.3f %4.3f %4.3f'%(vn[0],vn[1],vn[2],vn[3],vn[4],vn[5]))
    if np.all(site==V0):
        a=[]
        for i in range(len(symop)):
            a.append(i)
        idx_site=a
        idx_coset=[0]
    else:
        idx_site=site_symmetry(site,symop,brv)
        idx_coset=coset(site,symop,brv,pg,idx_site)
    if verbose>0:
        print('  order of site symmetry:',len(idx_site))
        print('  number of equivalent positions:',len(idx_coset))
    return idx_site,idx_coset

def equivalent_positions_in_unit_cell(site,brv,pg,vervose=0):
    """
    単位胞内にある等価なサイトを得る。
    """
    if vervose>0:
        print(' equivalent_positions_in_unit_cell()')
        print('  site:',numerical_vector(site))
    symop=dodesymop_array(pg)
    eqpos=np.zeros((len(symop),6,3),dtype=DTYPE_int)
    for i,op in enumerate(symop):
        eqpos[i]=symop_vec(op,site,centre=V0)
    eqpos1=remove_doubling(eqpos)
    if vervose>1:
        print('  len(eqpos1)=',len(eqpos1))
        for _eqpos1 in eqpos1:
            print('  _eqpos1:',numerical_vector(_eqpos1))
    
    # 求めた等価なサイトに、並進操作を施し、単位胞ないにあるもののみ得る。
    if np.all(site==V0):
        #return site.reshape(1,1,6,3),[0]
        return site.reshape(1,6,3),[0]
    else:
        translation=translation_new(brv,flag=0)
        out=np.zeros((len(translation)*len(eqpos1),6,3),dtype=DTYPE_int)
        num=0
        for pos in eqpos1:
            for tr in translation:
                _pos=add_vectors(pos,tr)
                _pos=numerical_vector(_pos)
                if np.all(_pos>=0.0) and np.all(_pos<1.0):
                    if vervose>1:
                        print('  _pos:',numerical_vector(_pos))
                    out[num]=_pos
                    num+=1
        out1=remove_doubling(out[:num])
        if vervose>0:
            for out1_ in out1:
                if vervose>1:
                    print('  equivalent site:',numerical_vector(out1_))
            
        # 得られたサイトが元のサイトとどのような対称操作で結ばれているのかを調べる。
        out_idx_symop=[]
        flag=0
        for out_ in out1:
            counter=0
            for i,op in enumerate(symop):
                vt=symop_vec(op,site,centre=V0)
                for tr in translation:
                    _vt=add_vectors(vt,tr)
                    if np.all(_vt==out_):
                        out_idx_symop.append(i)
                        counter=1
                        break
                if counter!=0:
                    break
            if counter==0:
                flag+=1
                break
            else:
                pass
        if flag==0:
            if vervose>1:
                print('out1.shape:',out1.shape)
            return out1,out_idx_symop
        else:
            return

def site_symmetry(site,brv,pg,vervose=0):
    """symmetry operators in the site symmetry group G.
        
    Args:
        site (numpy.ndarray):
            xyz coordinate of the site.
            The shape is (6,3).
        
    Returns:
        List of index of symmetry operators of the site symmetry group G (list):
            The symmetry operators leaves xyz identical.
    """
    
    if vervose>0:
        print(' site_symmetry()')
        print('  site:',numerical_vector(site))
    symop=dodesymop_array(pg)
    
    # サイト周りでvtgに対して対称操作を施す。
    vtg=np.array([[1,0,3],[0,1,4],[1,0,5],[0,1,6],[1,0,7],[0,0,1]],dtype=DTYPE_int)
    #vtg=add_vectors(vtg,site)
    a=np.zeros((len(symop),6,3),dtype=DTYPE_int)
    for i1,op in enumerate(symop):
        a[i1]=symop_vec(op,vtg,site)
        
    if brv=='p':
        flag=1
    else:
        pass
    traop=translation_new(brv,flag)
    lst=[]
    for i1,a1 in enumerate(a):
        # vtgに対して並進を含む全ての対称操作を施す。
        if vervose>1:
            print('%3d     a1:'%(i1),numerical_vector(a1))
        counter1=0
        for op in symop:
            tmp1=symop_vec(op,vtg,V0)
            if np.all(a1==tmp1):
                counter1=1
                if vervose>1:
                    print('      tmp1:',numerical_vector(tmp1))
                break
            else:
                flag1=0
                for tr in traop:
                    b=add_vectors(tmp1,tr)
                    if np.all(a1==b):
                        flag1=1
                        #print('         b:',numerical_vector(b))
                        break
                    else:
                        pass
                if flag1==1:
                    counter1=1
                    break
                else:
                    pass
        if counter1==1:
            lst.append(i1)
        else:
            pass
    if vervose>0:
        print('  lst:',lst)
    return lst
    

def equivalent_positions_in_unit_cell_dev(site,brv,pg,vervose=0):
    """
    単位胞内にある等価なサイトを得る。
    """
    if vervose>0:
        print(' equivalent_positions_in_unit_cell()')
        print('  site:',numerical_vector(site))
    symop=dodesymop_array(pg)
    eqpos=np.zeros((len(symop),6,3),dtype=DTYPE_int)
    for i,op in enumerate(symop):
        eqpos[i]=symop_vec(op,site,centre=V0)
    if vervose>1:
        for i,_eqpos in enumerate(eqpos):
            print('   (%d)'%(i),numerical_vector(_eqpos))
            
    # symopを施したサイトに並進操作を施し、単位胞内にあるものを得る。
    lst_symop_unit_cell=[]
    lst_symop_site_symm=[]
    tr=translation_new(brv,flag=1)
    for i1,_pos in enumerate(eqpos):
        for i2,_tr in enumerate(tr):
            pos=add_vectors(_pos,_tr)
            posn=numerical_vector(pos)
            if np.all(posn>=0.0) and np.all(posn<1.0):
                lst_symop_unit_cell.append([i1,i2])
                if vervose>1:
                    print('   (%d,%d):'%(i1,i2),numerical_vector(pos))
                else:
                    pass
            else:
                pass
        if np.all(_pos==site):
            lst_symop_site_symm.append(i1)
        else:
            pass
    print('lst_symop_unit_cell')
    for i in range(len(lst_symop_unit_cell)):
        print(lst_symop_unit_cell[i])
        
    print('\nSite symmetry:')
    print('lst_symop_site_symm')
    print(lst_symop_site_symm)
    #for i in range(len(lst_symop_site_symm)):
    #    print(lst_symop_site_symm[i])
    
    # 全対称操作をサイトシンメトリの操作を取り除く
    a=set(range(len(symop)))
    b=set(lst_symop_site_symm)-{0}
    idx_else=list(a-b)
    print(idx_else)
    
    print('lst_symop_unit_cell:',lst_symop_unit_cell)
    _lst_symop_unit_cell=[]
    for a in lst_symop_unit_cell:
        _lst_symop_unit_cell.append(a[0])
    print('_lst_symop_unit_cell:',_lst_symop_unit_cell)
    # idx_elseの対称操作のうち、各等価サイトを作る対称操作を調べる。
    tmp1=[]
    for idx2 in _lst_symop_unit_cell:
        
        _site=symop_vec(symop[idx2],site,V0)
        tmp=[]
        for idx1 in idx_else:
            pos1=symop_vec(symop[idx1],_site,V0)
            if np.all(pos==pos1):
                tmp.append(idx1)
                #break
            else:
                pass
        tmp1.append(tmp)
        print('  idx_coset',tmp)
        
    # いくつかある組み合わせのうち最初のものを選ぶ。
    idx_coset=[]
    for i in range(len(tmp1)):
        #idx_coset.append((tmp1[i][0]))
        idx_coset.append(tmp1[i])
    print('idx_coset')
    print(idx_coset)
    #if check_coset(site,idx_coset,symop,idx_site):
    #    return idx_coset
    #else:
    #    return 
    

#################
#   Utilities
#################
def remove_overlaps(l1):
    """
    Remove overlap elements in list with set method.
    
    Args:
        l1 (list):
    
    Returns:
        l2 (list)
    """
    tmp=set(l1)
    l2=list(tmp)
    l2.sort()
    return l2
    
def find_overlaps(l1,l2):
    """find overlap or not btween list1 and list2.
    
    Args:
        l1 (list):
        l2 (list):
    
    Returns:
        True : overlaping
        False: no overlap
    """
    l3=remove_overlaps(l1+l2)
    if len(l1)+len(l2)==len(l3): # no overlap
        return False
    else:
        return True

############################
# Similarity transformation
############################
def similarity_obj(obj,m):
    """similarity transformation of a triangle
    """
    out=np.zeros(obj.shape,dtype=DTYPE_int)
    for i1,od in enumerate(obj):
        out[i1]=similarity_triangle(od,m)
    return out

def similarity_triangle(triangle,m):
    """similarity transformation of a triangle
    """
    out=np.zeros(triangle.shape,dtype=DTYPE_int)
    for i1,vt in enumerate(triangle):
        out[i1]=similarity_vec(vt,m)
    return out

def similarity_vec(vt,m):
    """similarity transformation of a vector
    """
    vec1=[]
    op=similarity(m)
    return dot_product_1(op,vt)
    
def similarity(m):
    """Similarity transformation of Dodecagonal QC
    """
    m1=np.array([[ 1, 0, 0, -1, 0, 0],\
                [ 1, 1, 0, 0, 0, 0],\
                [ 0, 1, 1, 1, 0, 0],\
                [ 0, 0, 1, 1, 0, 0],\
                [ 0, 0, 0, 0, 1, 0],\
                [ 0, 0, 0, 0, 0, 1]],dtype=DTYPE_int)
    return matrixpow(m1.T,m)

############################
# Group
############################
def generate_multiplication_table(a,flag,ndim):
    """
    対称行列リストaから積表を求める
    Generation of a multiplication table from symmetry elements,
    for point group (flag='PG'), 
    for space group (flag='SG').
    
    input:
    list a, list of symmetry elements
    string flag: 'SG' for space group, 'PG' for point group,
    int ndim: dimention of periodic structure, (dummy for flag == 'PG')
    
    return:
    int ndarray table
        1（群をつくらない場合）
    """
    RTOL=1e-02
    ATOL=1e-03
    
    def translation_one_unit_cel(ndim):
        """
        ユニットセル1つ分だけシフトする。
        並進を含む対称操作が等価かどうか確認するときに用いる。
    
        int ndim: 3次元結晶の場合は3, icoの場合は6, decagonal, dodecagonalの場合は5
        """
        z0 = np.zeros([6,7])
        z1 = np.zeros([2,8])
    
        t1 = np.array([[1,0,0,0,0,0]]).T
        t2 = np.array([[0,1,0,0,0,0]]).T
        t3 = np.array([[0,0,1,0,0,0]]).T
    
        a1 = [-1,0,1]
    
        lst=[]
        if ndim==6:
            t4 = np.array([[0,0,0,1,0,0]]).T
            t5 = np.array([[0,0,0,0,1,0]]).T
            t6 = np.array([[0,0,0,0,0,1]]).T
            for i1 in a1:
                for i2 in a1:
                    for i3 in a1:
                        for i4 in a1:
                            for i5 in a1:
                                for i6 in a1:
                                    t = t1*i1+t2*i2+t3*i3+t4*i4+t5*i5+t6*i6
                                    tmp = np.block([z0,t])
                                    tmp = np.block([[tmp],[z1]])
                                    lst.append(tmp)
        elif ndim==3:
            for i1 in a1:
                for i2 in a1:
                    for i3 in a1:
                        t = t1*i1+t2*i2+t3*i3
                        tmp = np.block([z0,t])
                        tmp = np.block([[tmp],[z1]])
                        lst.append(tmp)
        elif ndim==5:
             t4 = np.array([[0,0,0,1,0,0]]).T
             t5 = np.array([[0,0,0,0,1,0]]).T
             for i1 in a1:
                for i2 in a1:
                    for i3 in a1:
                        for i4 in a1:
                            for i5 in a1:
                                t = t1*i1+t2*i2+t3*i3+t4*i4+t5*i5
                                tmp = np.block([z0,t])
                                tmp = np.block([[tmp],[z1]])
                                lst.append(tmp)
        else:
            pass
        return lst
    
    def equivalent(a,b,flag=0,ndim=6):
        """
        judge whether symmetric matrices a and b are equivalen or not. 
        Input
        nd.array a: symmetry element matrix
        nd.array b: symmetry element matrix
        int flag: 
            0 with translation symmetry, 
            1 without translation symmetry
        int ndim, dimension of periodic structure
        """
        counter = 0
        if flag==0:
            for t in translation_one_unit_cel(ndim):
                if np.all(np.isclose(a, b+t, rtol=RTOL, atol=ATOL)):
                    counter+=1
                    break
        else:
            if np.all(np.isclose(a, b, rtol=RTOL, atol=ATOL)):
                counter+=1
        if counter!=0:
            return True
        else:
            return False
    
    num = len(a)
    
    #print('generate_multiplication_table()')
    # 元の通し番号と6D表現行列を出力
    #print('\nSymmetry Elements')
    #for i1 in range(num):
    #    print('# %d:'%i1)
    #    print(a[i1])
    
    if flag=='SG':
        nflag = 0
    else:
        nflag = 1
        
    # 積表を作成
    lst = []
    for i1 in range(num):
        for i2 in range(num):
            b = np.dot(a[i2],a[i1])
            counter = 0
            for i3 in range(num):
                if equivalent(b,a[i3],nflag,ndim):
                    counter+=1
                    lst.append(i3)
                    break
            if counter == 0:
                lst.append(-1) # a1とa2の積がリストaの中に含まれていない場合は-1を代入する。
    if len(lst) == num**2:
        return np.array(lst).reshape(num,num)
    else:
        print('Cannot make multiplication table.')
        return 1

def check_closure(mul_table,combination):
    # 閉包性(closure)
    counter3 = 0
    num = len(combination)
    for i2 in range(num):
        for i3 in range(num):
            counter = 0
            for i4 in combination:
                if mul_table[i2][i3] == i4:
                    counter+=1
            if counter == 0:
                counter3+=1
                break
    if counter3 == 0:
        return True # closure
    else:
        return False

def check_identity_element(combination,num_identity=0):
    # 単位元(identity element)の存在
    counter1 = 0
    for i2 in range(len(combination)):
        if combination[i2] == num_identity:
            counter1+=1
            break
    if counter1 == 1:
        return True
    else:
        return False

def check_inverse_element(mul_table,num_identity=0):
    # 逆元(inverse element)の存在
    (m,n)=mul_table.shape
    if m == n:
        counter2 = 0
        for i2 in range(m):
            counter = 0
            for i3 in range(m):
                if mul_table[i2][i3] == num_identity:
                    counter+=1
                    break
            if counter != 1:
                counter2+=1
                break
        if counter2 == 0:
            return True
        else:
            return False
    else:
        return False

def check_connectivity(mul_table,combination):
    # 結合律(connectivity)
    counter3=0
    num = len(combination)
    for i2 in range(num):
        for i3 in range(num):
            for i4 in range(num):
                ab = mul_table[i2][i3]
                ab =combination.index(ab)
                abc1 = mul_table[ab][i4]
                bc = mul_table[i3][i4]
                bc =combination.index(bc)
                abc2 = mul_table[i2][bc]
                if abc1 != abc2:
                    counter3+=1
                    break
    if counter3 == 0:
        return True
    else:
        return False

def check_group(a,flag='SG',ndim=6):
    """
    """
    table=generate_multiplication_table(a,flag,ndim)
    comb=list(range(len(a)))
    
    if check_closure(table,comb): # 閉包性のチェック
        #if check_identity_element(comb,num_identity): # 単位元の存在
        # 前処理で単位元を必ず含んでいる為、単位元の存在を再度確認する必要はない。
        num_identity=0
        if check_inverse_element(table,num_identity): # 逆元の存在
            if check_connectivity(table,comb): # 結合律
                return True
            else:
                return False
        else:
            return False
    else:
        return False

def stereographic_projection(symop,idx_site,idx_coset,vn0):
    #print('Stereographic projection')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,8))
    
    x1e=[]
    y1e=[]
    x2e=[]
    y2e=[]
    x1i=[]
    y1i=[]
    x2i=[]
    y2i=[]
    for idx in idx_site:
        sop=symop[idx]
        vn=np.dot(sop,vn0)
        p=projection_numerical(vn)
        xe=p[0]
        ye=p[1]
        ze=p[2]
        xi=p[3]
        yi=p[4]
        zi=p[5]
        dde=np.sqrt(xe**2+ye**2+ze**2)+EPS
        ddi=np.sqrt(xi**2+yi**2+zi**2)+EPS
        xe=xe/dde
        ye=ye/dde
        ze=ze/dde
        xi=xi/ddi
        yi=yi/ddi
        zi=zi/ddi
        if ze>=0:
            x1e.append(float(xe))
            y1e.append(float(ye))
        else:
            x2e.append(float(xe))
            y2e.append(float(ye))
        if zi>=0:
            x1i.append(float(xi))
            y1i.append(float(yi))
        else:
            x2i.append(float(xi))
            y2i.append(float(yi))
            
    x3e=[]
    y3e=[]
    x4e=[]
    y4e=[]
    x3i=[]
    y3i=[]
    x4i=[]
    y4i=[]
    for idx in idx_coset:
        sop=symop[idx]
        vn=np.dot(sop,vn0)
        p=projection_numerical(vn)
        print(p)
        xe=p[0]
        ye=p[1]
        ze=p[2]
        xi=p[3]
        yi=p[4]
        zi=p[5]
        dde=np.sqrt(xe**2+ye**2+ze**2)
        ddi=np.sqrt(xi**2+yi**2+zi**2)
        xe=xe/dde
        ye=ye/dde
        ze=ze/dde
        xi=xi/ddi
        yi=yi/ddi
        zi=zi/ddi
        if ze>=0:
            x3e.append(float(xe))
            y3e.append(float(ye))
        else:
            x4e.append(float(xe))
            y4e.append(float(ye))
        if zi>=0:
            x3i.append(float(xi))
            y3i.append(float(yi))
        else:
            x4i.append(float(xi))
            y4i.append(float(yi))
            
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title('site symmetry (par)')
    ax1.scatter([0], [0], s=30000, marker='o', color='white', alpha=1.0, edgecolors='black')
    ax1.scatter(x2e, y2e, s=200, marker='o', color='white', alpha=1.0, edgecolors='black')
    ax1.scatter(x1e, y1e, s=40,  marker='o', color='black', alpha=1.0, edgecolors='black')
    ax1.set_xlim(-1.3,1.3)
    ax1.set_ylim(-1.3,1.3)
    ax1.axis("off")
    
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title('coset (par)')
    ax2.scatter([0], [0], s=30000, marker='o', color='white', alpha=1.0, edgecolors='black')
    ax2.scatter(x4e, y4e, s=200, marker='o', color='white', alpha=1.0, edgecolors='black')
    ax2.scatter(x3e, y3e, s=40,  marker='o', color='black', alpha=1.0, edgecolors='black')
    ax2.set_xlim(-1.3,1.3)
    ax2.set_ylim(-1.3,1.3)
    ax2.axis("off")
    
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title('site symmetry (perp)')
    ax3.scatter([0], [0], s=30000, marker='o', color='white', alpha=1.0, edgecolors='black')
    ax3.scatter(x2i, y2i, s=200, marker='o', color='white', alpha=1.0, edgecolors='black')
    ax3.scatter(x1i, y1i, s=40,  marker='o', color='black', alpha=1.0, edgecolors='black')
    ax3.set_xlim(-1.3,1.3)
    ax3.set_ylim(-1.3,1.3)
    ax3.axis("off")
    
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title('coset (perp)')
    ax4.scatter([0], [0], s=30000, marker='o', color='white', alpha=1.0, edgecolors='black')
    ax4.scatter(x4i, y4i, s=200, marker='o', color='white', alpha=1.0, edgecolors='black')
    ax4.scatter(x3i, y3i, s=40,  marker='o', color='black', alpha=1.0, edgecolors='black')
    ax4.set_xlim(-1.3,1.3)
    ax4.set_ylim(-1.3,1.3)
    ax4.axis("off")
    
    #plt.axis("off")
    plt.show()
    return 0

