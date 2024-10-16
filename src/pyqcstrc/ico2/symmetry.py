#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
#sys.path.append('.')
from pyqcstrc.ico2.math1 import (add, 
                                matrixpow, 
                                dot_product, 
                                dot_product_1, 
                                sub_vectors, 
                                add_vectors,
                                mul_vectors,
                                )
from pyqcstrc.ico2.utils import (remove_doubling_in_perp_space, 
                                remove_doubling,
                                )
from pyqcstrc.ico2.numericalc import (length_numerical,
                                     numerical_vector,
                                     numeric_value,
                                     )
import numpy as np

PI=np.pi
EPS=1e-6
V0=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.int64)
TAU=(1+np.sqrt(5))/2.0

def symop_obj(symop,obj,centre):
    """ Apply a symmetric operation on an object around given centre. in TAU-style
    
    """
    ndim=obj.ndim
    if ndim==3:
        return symop_vecs(symop,obj,centre)
    elif ndim==4:
        obj1=np.zeros(obj.shape,dtype=np.int64)
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

def symop_vecs(symop,tetrahedron,centre):
    """ Apply a symmetric operation on set of vectors around given centre. in TAU-style
    
    """
    tetrahedron1=np.zeros(tetrahedron.shape,dtype=np.int64)
    for i,vt in enumerate(tetrahedron):
        tetrahedron1[i]=symop_vec(symop,vt,centre)
    return tetrahedron1

def symop_vec(symop,vt,centre):
    """ Apply a symmetric operation on a vector around given centre. in TAU-style
    
    """
    vt=sub_vectors(vt,centre)
    vt=dot_product_1(symop,vt)
    return add_vectors(vt,centre)





def generator_obj_symmetric_obj(obj,centre):
    
    if obj.ndim==2 or obj.ndim==3 or obj.ndim==4:
        mop=icosasymop()
        num=len(mop)
        shape=tuple([num])
        a=np.zeros(shape+obj.shape,dtype=np.int64)
        for i,op in enumerate(mop):
            a[i]=symop_obj(op,obj,centre)
        if obj.ndim==4:
            n1,n2,_,_=obj.shape
            a=a.reshape(num*n1,n2,6,3)
        else:
            pass
        return a
    else:
        print('object has an incorrect shape!')
        return

def generator_obj_symmetric_surface(obj,centre):
    return generator_obj_symmetric_obj(obj,centre)
    
def generator_obj_symmetric_tetrahedron(obj,centre):
    return generator_obj_symmetric_obj(obj,centre)

def generator_obj_symmetric_vector_specific_symop(vt,centre,list_of_symmetry_operation_index):
    # using specific symmetry operations
    if vt.ndim==2:
        mop=icosasymop()
        shape=tuple([len(list_of_symmetry_operation_index)])
        a=np.zeros(shape+vt.shape,dtype=np.int64)
        for j0,i1 in enumerate(list_of_symmetry_operation_index):
            a[j0]=symop_obj(mop[i1],vt,centre)
        return a
    else:
        print('vt has an incorrect shape!')
        return
        
def generator_obj_symmetric_tetrahedron_specific_symop(obj,centre,list_of_symmetry_operation_index):
    # using specific symmetry operations
    if obj.ndim==3 or obj.ndim==4:
        mop=icosasymop()
        shape=tuple([len(list_of_symmetry_operation_index)])
        a=np.zeros(shape+obj.shape,dtype=np.int64)
        j0=0
        for i1 in list_of_symmetry_operation_index:
            a[j0]=symop_obj(mop[i1],obj,centre)
            j0+=1
        return a
    else:
        print('object has an incorrect shape!')
        return

def generator_obj_symmetric_obj_specific_symop(obj,centre,list_of_symmetry_operation_index):
    # using specific symmetry operations
    mop=icosasymop()
    shape=tuple([len(list_of_symmetry_operation_index)])
    a=np.zeros(shape+obj.shape,dtype=np.int64)
    #print('shape+obj.shape:',shape+obj.shape)
    if obj.ndim==4:
        for j0,i1 in enumerate(list_of_symmetry_operation_index):
            a[j0]=symop_obj(mop[i1],obj,centre)
        return a
    elif obj.ndim==5:
        for j0,i1 in enumerate(list_of_symmetry_operation_index):
            for j1,ob in enumerate(obj):
                a[j0][j1]=symop_obj(mop[i1],ob,centre)
        return a
    else:
        print('object has an incorrect shape!')
        return


def generator_obj_symmetric_tetrahedron_0(obj,centre,symmetry_operation_index):
    mop=icosasymop()
    return symop_obj(mop[symmetry_operation_index],obj,centre)

def generator_obj_symmetric_vec(vectors, centre):
    return generator_obj_symmetric_obj(vectors,centre)

def generator_equivalent_vectors(vectors,centre):
    a=generator_obj_symmetric_obj(vector,centre)
    return remove_doubling_in_perp_space(a)
    #return remove_doubling(a)

def generator_equivalent_vec(vector,centre):
    a=generator_obj_symmetric_obj(vector,centre)
    return remove_doubling_in_perp_space(a)
    #return remove_doubling(a)

def icosasymop():
    # icosahedral symmetry operations
    m1=np.array([[ 1, 0, 0, 0, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 1, 0, 0],\
                [ 0, 0, 0, 0, 1, 0],\
                [ 0, 0, 0, 0, 0, 1],\
                [ 0, 1, 0, 0, 0, 0]],dtype=np.int64)
    # mirror
    m2=np.array([[-1, 0, 0, 0, 0, 0],\
                [ 0,-1, 0, 0, 0, 0],\
                [ 0, 0, 0, 0, 0,-1],\
                [ 0, 0, 0, 0,-1, 0],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0,-1, 0, 0, 0]],dtype=np.int64)
    # c2
    m3=np.array([[ 0, 0, 0, 0, 0,-1],\
                [ 0,-1, 0, 0, 0, 0],\
                [ 0, 0, 0, 1, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 0,-1, 0],\
                [-1, 0, 0, 0, 0, 0]],dtype=np.int64)
    # c3
    m4=np.array([[ 0, 1, 0, 0, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 1, 0, 0, 0, 0, 0],\
                [ 0, 0, 0, 0, 0, 1],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0, 0, 0,-1, 0]],dtype=np.int64)
    # inversion
    m5=np.array([[-1, 0, 0, 0, 0, 0],\
                [ 0,-1, 0, 0, 0, 0],\
                [ 0, 0,-1, 0, 0, 0],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0, 0, 0,-1, 0],\
                [ 0, 0, 0, 0, 0,-1]],dtype=np.int64)
    symop=[]
    for m in range(2):
        for l in range(3):
            for k in range(2):
                for j in range(2):
                    for i in range(5):
                        s1=matrixpow(m1,i) # c5
                        s2=matrixpow(m2,j) # mirror
                        s3=matrixpow(m3,k) # c2
                        s4=matrixpow(m4,l) # c3
                        s5=matrixpow(m5,m) # inversion
                        tmp=np.dot(s5,s4)
                        tmp=np.dot(tmp,s3)
                        tmp=np.dot(tmp,s2)
                        tmp=np.dot(tmp,s1)
                        symop.append(tmp)
    return symop

def icosasymop_array():
    # icosahedral symmetry operations
    m1=np.array([[ 1, 0, 0, 0, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 1, 0, 0],\
                [ 0, 0, 0, 0, 1, 0],\
                [ 0, 0, 0, 0, 0, 1],\
                [ 0, 1, 0, 0, 0, 0]],dtype=np.int64)
    # mirror
    m2=np.array([[-1, 0, 0, 0, 0, 0],\
                [ 0,-1, 0, 0, 0, 0],\
                [ 0, 0, 0, 0, 0,-1],\
                [ 0, 0, 0, 0,-1, 0],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0,-1, 0, 0, 0]],dtype=np.int64)
    # c2
    m3=np.array([[ 0, 0, 0, 0, 0,-1],\
                [ 0,-1, 0, 0, 0, 0],\
                [ 0, 0, 0, 1, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 0, 0, 0, 0,-1, 0],\
                [-1, 0, 0, 0, 0, 0]],dtype=np.int64)
    # c3
    m4=np.array([[ 0, 1, 0, 0, 0, 0],\
                [ 0, 0, 1, 0, 0, 0],\
                [ 1, 0, 0, 0, 0, 0],\
                [ 0, 0, 0, 0, 0, 1],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0, 0, 0,-1, 0]],dtype=np.int64)
    # inversion
    m5=np.array([[-1, 0, 0, 0, 0, 0],\
                [ 0,-1, 0, 0, 0, 0],\
                [ 0, 0,-1, 0, 0, 0],\
                [ 0, 0, 0,-1, 0, 0],\
                [ 0, 0, 0, 0,-1, 0],\
                [ 0, 0, 0, 0, 0,-1]],dtype=np.int64)
    symop=np.zeros((120,6,6),dtype=np.int64)
    num=0
    for m in range(2): # 2
        for l in range(3): # 3
            for k in range(2): # 2
                for j in range(2): # 2
                    for i in range(5): # 5
                        s1=matrixpow(m1,i) # c5
                        s2=matrixpow(m2,j) # mirror
                        s3=matrixpow(m3,k) # c2
                        s4=matrixpow(m4,l) # c3
                        s5=matrixpow(m5,m) # inversion
                        tmp=np.dot(s5,s4)
                        tmp=np.dot(tmp,s3)
                        tmp=np.dot(tmp,s2)
                        tmp=np.dot(tmp,s1)
                        symop[num]=tmp
                        num+=1
    return symop

def translation(brv,flag=0):
    """translational symmetry
    
    brv : bravais lattce p, i, f, s, c
            s : superlattice for decagonal quasicrystal
    """
    symop=[]
    tmp=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
    symop.append(tmp)
    
    if brv=='p':
        if flag==0:
            lst=[-1,0,1]
        elif flag==1:
            lst=[0,1]
        elif flag==-1:
            lst=[-1,0]
        else:
            lst=[-1,0,1]
    elif brv=='f':
        print('not supported')
        return 
    elif brv=='s':
        print('not supported')
        return 
    else:
        print('not supported')
        return 
    for i1 in lst:
        for i2 in lst:
            for i3 in lst:
                for i4 in lst:
                    for i5 in lst:
                        for i6 in lst:
                            tmp=np.array([[i1,0,1],[i2,0,1],[i3,0,1],[i4,0,1],[i5,0,1],[i6,0,1]])
                            symop.append(tmp)
    return np.array(symop)

# new ver
def translation_new(brv,flag=0):
    """translational symmetry
    
    brv : bravais lattce p, i, f, s, c
            s : superlattice for decagonal quasicrystal
    """
    if flag==1:
        lst=[0,1]
    elif flag==-1:
        lst=[-1,0]
    else:
        lst=[-1,0,1]
        
    tr=np.zeros((len(lst)**6,6,3),dtype=np.int64)
    j1=0
    for i1 in lst:
        for i2 in lst:
            for i3 in lst:
                for i4 in lst:
                    for i5 in lst:
                        for i6 in lst:
                            tr[j1]=np.array([[i1,0,1],[i2,0,1],[i3,0,1],[i4,0,1],[i5,0,1],[i6,0,1]])
                            j1+=1
    
    if brv=='p':
        return tr
        
    elif brv=='f':
        tc=np.array([[[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],\
                    [[1,0,2],[1,0,2],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],\
                    [[1,0,2],[0,0,1],[1,0,2],[0,0,1],[0,0,1],[0,0,1]],\
                    [[1,0,2],[0,0,1],[0,0,1],[1,0,2],[0,0,1],[0,0,1]],\
                    [[1,0,2],[0,0,1],[0,0,1],[0,0,1],[1,0,2],[0,0,1]],\
                    [[1,0,2],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[1,0,2]],\
                    [[0,0,1],[1,0,2],[1,0,2],[0,0,1],[0,0,1],[0,0,1]],\
                    [[0,0,1],[1,0,2],[0,0,1],[1,0,2],[0,0,1],[0,0,1]],\
                    [[0,0,1],[1,0,2],[0,0,1],[0,0,1],[1,0,2],[0,0,1]],\
                    [[0,0,1],[1,0,2],[0,0,1],[0,0,1],[0,0,1],[1,0,2]],\
                    [[0,0,1],[0,0,1],[1,0,2],[1,0,2],[0,0,1],[0,0,1]],\
                    [[0,0,1],[0,0,1],[1,0,2],[0,0,1],[1,0,2],[0,0,1]],\
                    [[0,0,1],[0,0,1],[1,0,2],[0,0,1],[0,0,1],[1,0,2]],\
                    [[0,0,1],[0,0,1],[0,0,1],[1,0,2],[1,0,2],[0,0,1]],\
                    [[0,0,1],[0,0,1],[0,0,1],[1,0,2],[0,0,1],[1,0,2]],\
                    [[0,0,1],[0,0,1],[0,0,1],[0,0,1],[1,0,2],[1,0,2]],\
                    [[0,0,1],[0,0,1],[1,0,2],[1,0,2],[1,0,2],[1,0,2]],\
                    [[0,0,1],[1,0,2],[0,0,1],[1,0,2],[1,0,2],[1,0,2]],\
                    [[0,0,1],[1,0,2],[1,0,2],[0,0,1],[1,0,2],[1,0,2]],\
                    [[0,0,1],[1,0,2],[1,0,2],[1,0,2],[0,0,1],[1,0,2]],\
                    [[0,0,1],[1,0,2],[1,0,2],[1,0,2],[1,0,2],[0,0,1]],\
                    [[1,0,2],[0,0,1],[0,0,1],[1,0,2],[1,0,2],[1,0,2]],\
                    [[1,0,2],[0,0,1],[1,0,2],[0,0,1],[1,0,2],[1,0,2]],\
                    [[1,0,2],[0,0,1],[1,0,2],[1,0,2],[0,0,1],[1,0,2]],\
                    [[1,0,2],[0,0,1],[1,0,2],[1,0,2],[1,0,2],[0,0,1]],\
                    [[1,0,2],[1,0,2],[0,0,1],[0,0,1],[1,0,2],[1,0,2]],\
                    [[1,0,2],[1,0,2],[0,0,1],[1,0,2],[0,0,1],[1,0,2]],\
                    [[1,0,2],[1,0,2],[0,0,1],[1,0,2],[1,0,2],[0,0,1]],\
                    [[1,0,2],[1,0,2],[1,0,2],[0,0,1],[0,0,1],[1,0,2]],\
                    [[1,0,2],[1,0,2],[1,0,2],[0,0,1],[1,0,2],[0,0,1]],\
                    [[1,0,2],[1,0,2],[1,0,2],[1,0,2],[0,0,1],[0,0,1]],\
                    [[1,0,2],[1,0,2],[1,0,2],[1,0,2],[1,0,2],[1,0,2]]])
        a=np.zeros((len(tc)*len(tr),6,3),dtype=np.int64)
        counter1=0
        for tr1 in tr:
            for tc1 in tc:
                tc2=add_vectors(tr1,tc1)
                flag=0
                #"""
                for tc3 in tc2:
                    if numeric_value(tc3)>1.0:
                        flag+=1
                        break
                    else:
                        pass
                if flag==0:
                    a[counter1]=tc2
                    counter1+=1
                #"""
                #a[counter1]=tc2
                #counter1+=1
        a=a[:counter1]
        return a
        
    elif brv=='s':
        #tf=mul_vectors(tf,np.array([2,0,1]))
        return 
        
    elif brv=='i':
        tc=np.array([[[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],\
                     [[1,0,2],[1,0,2],[1,0,2],[1,0,2],[1,0,2],[1,0,2]]])
        a=np.zeros((len(tc)*len(tr),6,3),dtype=np.int64)
        counter1=0
        for tr1 in tr:
            for tc1 in tc:
                tc2=add_vectors(tr1,tc1)
                flag=0
                #"""
                for tc3 in tc2:
                    if numeric_value(tc3)>1.0:
                        flag+=1
                        break
                    else:
                        pass
                if flag==0:
                    a[counter1]=tc2
                    counter1+=1
                #"""
                #a[counter1]=tc2
                #counter1+=1
        a=a[:counter1]
        return a
         
    else:
        print('no lattice type selected.')
        return 
        
        
####### WIP ########
################ 
# site symmetry
################
def site_symmetry_and_coset(site,brv,verbose=0):
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
        POS_V  = np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.int64)
        POS_C  = np.array([[1,0,2],[1,0,2],[1,0,2],[1,0,2],[1,0,2],[1,0,2]],dtype=np.int64)
        POS_EC = np.array([[1,0,2],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.int64)
        
        if np.all(site==POS_V):
            a=[]
            for i in range(120):
                a.append(i)
            return a
        elif np.all(site==POS_EC):
            if brv=='f':
                a=[]
                for i in range(120):
                    a.append(i)
            else:
                a=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69]
            return a
        elif  np.all(site==POS_C):
            a=[]
            for i in range(120):
                a.append(i)
            return a
        else:
            pass
        
        # サイト周りでvtgに対して点群m35の対称操作を施す。
        vtg=np.array([[1,0,3],[0,1,4],[1,0,5],[0,1,6],[1,0,7],[0,1,8]],dtype=np.int64)
        a=np.zeros((len(symop),6,3),dtype=np.int64)
        for i1,op in enumerate(symop):
            a[i1]=symop_vec(op,vtg,site)
            
        if brv=='p':
            flag=1
        elif brv=='f':
            flag=0
        traop=translation_new(brv,flag)
        lst=[]
        for i1,a1 in enumerate(a):
            # vtgに対して並進を含む全ての対称操作を施す。
            #print('%3d     a1:'%(i1),numerical_vector(a1))
            counter1=0
            for op in symop:
                tmp1=symop_vec(op,vtg,V0)
                if np.all(a1==tmp1):
                    counter1+=1
                    #print('      tmp1:',numerical_vector(tmp1))
                    break
                else:
                    flag1=0
                    for tr in traop:
                        b=add_vectors(tmp1,tr)
                        if np.all(a1==b):
                            flag1+=1
                            #print('         b:',numerical_vector(b))
                            break
                        else:
                            pass
                    if flag1==1:
                        counter1+=1
                        break
                    else:
                        pass
            if counter1==1:
                lst.append(i1)
            else:
                pass
        print(lst)
        return lst
        
    def coset(site,symop,brv,idx_site):
        """
        """
        #print('coset():')
        #idx_site=site_symmetry(site,brv)
        
        # coordinate of equivalent sites
        pos_equiv=equivalent_positions(site,brv)
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
            
    def equivalent_positions(site,brv):
        """
        siteに対して点群の対称性を施したサイトのうち、並進操作のみで結ばれない位置を求める。
        適切な名前を決める必要がある！！！
        """
        #print('site:',numerical_vector(site))
        symop=icosasymop_array()
        eqpos=np.zeros((len(symop),6,3),dtype=np.int64)
        for i,op in enumerate(symop):
            eqpos[i]=symop_vec(op,site,centre=V0)
        eqpos1=remove_doubling(eqpos) 
    
        # 求めた等価なサイトのうち、並進操作を施して同一なのであれば、どちらか片方を選ぶようにする。
        if len(eqpos1)==1:
            pass
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
                
        # 求めたサイトのうち単位胞内にあるサイトを選ぶ
        out=np.zeros((len(lst_saved),6,3),dtype=np.int64)
        num=0
        for vt in lst_saved:
            vn=numerical_vector(vt)
            if np.all(vn>=0.0):# and np.all(vn<=1.0):
                out[num]=vt
                num+=1
            else:
                pass
        return out[:num]
        
    vn=numerical_vector(site)
    if verbose>0:
        print(' site: %3.2f %3.2f %3.2f %3.2f %3.2f %3.2f'%(vn[0],vn[1],vn[2],vn[3],vn[4],vn[5]))
        
    symop=icosasymop_array()
    idx_site=site_symmetry(site,symop,brv)
    idx_coset=coset(site,symop,brv,idx_site)
    #print(idx_coset)
    if verbose>0:
        print('  order of site symmetry:',len(idx_site))
        print('  number of equivalent positions:',len(idx_coset))
    #print('  idx_coset:',idx_coset)
    #print('  idx_site:',idx_site)
    return idx_site,idx_coset

def icosasymop3_array(flag=None):
    """
    symmetry operation
    
    flag: 
        'axial': axial vector (e.g. classical spin)
    """
    def _matrixpow(m,num):
        """
        do rotational operation num times
        """
        m0=np.array([[1.0, 0.0, 0.0],\
                     [0.0, 1.0, 0.0],\
                     [0.0, 0.0, 1.0]],dtype=np.float64)
        for _ in range(num):
            m0 = np.dot(m0,m)
        return m0
        
    def _genmatrix(axis,fold):
        """
        generate matrix for n-fold roatational symmetry, Cn
        
        input
        ndarray axis: roatational axis
        int n:
        """ 
        n1, n2, n3 = axis/np.linalg.norm(axis)
        theta = -2.0*PI/fold
        cos = np.cos(theta)
        sin = np.sin(theta)
    
        # Rodrigues' rotation formula
        a11 = n1**2.0*(1.0-cos) +    cos
        a12 = n1*n2  *(1.0-cos) + n3*sin
        a13 = n1*n3  *(1.0-cos) - n2*sin
        a21 = n1*n2  *(1.0-cos) - n3*sin
        a22 = n2**2.0*(1.0-cos) +    cos
        a23 = n2*n3  *(1.0-cos) + n1*sin
        a31 = n1*n3  *(1.0-cos) + n2*sin
        a32 = n2*n3  *(1.0-cos) - n1*sin
        a33 = n3**2.0*(1.0-cos) +    cos
        
        return np.array([[a11, a21, a31],\
                         [a12, a22, a32],\
                         [a13, a23, a33]], dtype=np.float64)
                         
    # matrix of symmetry operation
    m1=_genmatrix(np.array([    1.0,  TAU,   0.0],dtype=np.float64), 5) # c5
    m2=_genmatrix(np.array([   -TAU, +1.0, TAU+1],dtype=np.float64), 2) # mirror
    m3=_genmatrix(np.array([   -TAU, -1.0, TAU+1],dtype=np.float64), 2) # c2
    m4=_genmatrix(np.array([2*TAU+1,  TAU,   0.0],dtype=np.float64), 3) # c3
    m5=np.array([[-1.0, 0.0, 0.0],\
                 [ 0.0,-1.0, 0.0],\
                 [ 0.0, 0.0,-1.0]],dtype=np.float64) # inverse
                
    symop=np.zeros((120,3,3),dtype=np.float64)
    num=0
    for m in range(2): # 2
        for l in range(3): # 3
            for k in range(2): # 2
                for j in range(2): # 2
                    for i in range(5): # 5
                        s1=_matrixpow(m1,i) # c5
                        s2=_matrixpow(m2,j) # mirror
                        s3=_matrixpow(m3,k) # c2
                        s4=_matrixpow(m4,l) # c3
                        s5=_matrixpow(m5,m) # inversion
                        tmp=np.dot(s5,s4)
                        tmp=np.dot(tmp,s3)
                        tmp=np.dot(tmp,s2)
                        tmp=np.dot(tmp,s1)
                        if flag=='axial':
                            symop[num]=np.linalg.det(tmp)*tmp # det(AB)=det(A)*det(B)
                        else:
                            symop[num]=tmp
                        num+=1
    return symop
    
""" old
def coset(site,brv):
    symop=icosasymop()
    
    list1=site_symmetry(site,brv)
    
    # List of index of symmetry operators which are not in the G.
    tmp=range(len(symop))
    tmp=set(tmp)-set(list1)
    list2=list(tmp)
    
    list2_new=remove_overlaps(list2)
    
    if len(symop)==len(list1):
        list5=[0]
    else:
        # left coset decomposition:
        list4=[]
        for i2 in list2_new:
            list3=[]
            for i1 in list1:
                op1=np.dot(symop[i2],symop[i1])
                for i3,op in enumerate(symop):
                    if np.all(op1==op):
                        list3.append(i3)
                        break
                    else:
                        pass
            list4.append(list3)
        
        for i2 in range(len(list4)-1):
            a=list4[i2]
            b=[]
            d=[]
            list5=[0] # symmetry element of identity, symop[0]
            list5.append(list2_new[i2])
            i3=i2+1
            while i3<len(list4):
                b=list4[i3]
                if len(d)==0:
                    if find_overlaps(a,b):
                        pass
                    else:
                        d=a+b
                        list5.append(list2_new[i3])
                else:
                    if find_overlaps(d,b):
                        pass
                    else:
                        d=d+b
                        list5.append(list2_new[i3])
                i3+=1
            b=remove_overlaps(d)
            if len(symop)==len(list5)*len(list1):
                break
            else:
                pass
    return list5

def site_symmetry_and_coset(site,brv):
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
    
    symop=icosasymop()
    traop=translation(brv)
    
    # List of index of symmetry operators of the site symmetry group G.
    # The symmetry operators leaves xyz identical.
    list1=[]
    
    # List of index of symmetry operators which are not in the G.
    list2=[]
    
    for i2,op in enumerate(symop):
        site1=symop_vec(op,site,V0)
        flag=0
        for top in traop:
            site2=add_vectors(site1,top)
            tmp=sub_vectors(site,site2)
            if length_numerical(tmp)<EPS:
                list1.append(i2)
                flag+=1
                break
            else:
                pass
        if flag==0:
            list2.append(i2)
    
    list1_new=remove_overlaps(list1)
    list2_new=remove_overlaps(list2)
    
    if len(symop)==len(list1_new):
        list5=[0]
    else:
        # left coset decomposition:
        list4=[]
        for i2 in list2_new:
            list3=[]
            for i1 in list1_new:
                op1=np.dot(symop[i2],symop[i1])
                for i3,op in enumerate(symop):
                    if np.all(op1==op):
                        list3.append(i3)
                        break
                    else:
                        pass
            list4.append(list3)
        
        for i2 in range(len(list4)-1):
            a=list4[i2]
            b=[]
            d=[]
            list5=[0] # symmetry element of identity, symop[0]
            list5.append(list2_new[i2])
            i3=i2+1
            while i3<len(list4):
                b=list4[i3]
                if len(d)==0:
                    if find_overlaps(a,b):
                        pass
                    else:
                        d=a+b
                        list5.append(list2_new[i3])
                else:
                    if find_overlaps(d,b):
                        pass
                    else:
                        d=d+b
                        list5.append(list2_new[i3])
                i3+=1
            b=remove_overlaps(d)
            if len(symop)==len(list5)*len(list1_new):
                break
            else:
                pass
    
    return list1_new,list5
"""

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
    out=np.zeros(obj.shape,dtype=np.int64)
    for i1,od in enumerate(obj):
        out[i1]=similarity_triangle(od,m)
    return out

def similarity_triangle(triangle,m):
    """similarity transformation of a triangle
    """
    out=np.zeros(triangle.shape,dtype=np.int64)
    for i1,vt in enumerate(triangle):
        out[i1]=similarity_vec(vt,m)
    return out

def similarity_vec(vt,m):
    """similarity transformation of a vector
    """
    #vec1=[]
    op=similarity(m)
    return dot_product_1(op,vt)

def similarity(m):
    """Similarity transformation of icosahedral QCs
    """
    m1=0.5*np.array([[1, 1, 1, 1, 1, 1],\
                    [ 1, 1, 1,-1,-1, 1],\
                    [ 1, 1, 1, 1,-1,-1],\
                    [ 1,-1, 1, 1, 1,-1],\
                    [ 1,-1,-1, 1, 1, 1],\
                    [ 1, 1,-1,-1, 1, 1]],dtype=np.int64)
    return matrixpow(m1.T,m)

if __name__ == '__main__':
    
    import random
    from pyqcstrc.ico2.numericalc import (
                            projection_par_numerical,
                            numerical_vectors,
                            numerical_vector,
                            numeric_value,
                            )
    # test
    POS_V  = np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.int64)
    POS_C  = np.array([[1,0,2],[1,0,2],[1,0,2],[1,0,2],[1,0,2],[1,0,2]],dtype=np.int64)
    POS_EC = np.array([[1,0,2],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.int64)
    
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
    
    cen0=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
    
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
    #brv='p'
    brv='f'
    #brv='i'
    #site=POS_V
    #site=POS_C
    site=POS_EC
    
    eqpos=equivalent_positions(site,brv)
    for i,p in enumerate(eqpos):
        print(i,numerical_vector(p))
    print('len(eqpos):',len(eqpos))
    """
    
    
    
    ##-----------------------------------------------
    # TEST: site_symmetry_and_coset()
    ##-----------------------------------------------
    #"""
    print('TEST: site_symmetry_and_coset()')
    #site=POS_V
    #site=POS_C
    site=POS_EC
    
    print('site:')
    print(site)
    
    print('P-type icosahedral lattice')
    brv='p'
    idx_ssym,idx_coset=site_symmetry_and_coset(site,brv,verbose=1)
    print('idx_ssym:',idx_ssym)
    print('idx_coset:',idx_coset)
    
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
    #"""
