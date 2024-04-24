#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

EPS=1e-6

####################
# 2/m-3 #
####################
def cubegmat():
    """
    3x3 matices of cubic symmetry operations
    1 -x, y, z;
    2 -x,-y, z;
    3  y, z, x;
    4 -x,-y,-z;

    x,y,z -> identity operation
    """
    #>>>mirror
    m1=np.array([-1, 0, 0,
                0, 1, 0,
                0, 0, 1 ])
    m1.shape=(3,3)
    #>>>c2
    m2=np.array([-1, 0, 0,
                0,-1, 0,
                0, 0, 1 ])
    m2.shape=(3,3)
    #>>>c3
    m3=np.array([ 0, 1, 0,
                0, 0, 1,
                1, 0, 0 ])
    m3.shape=(3,3)
    #>>>inversion
    m4=np.array([-1, 0, 0,
                0,-1, 0,
                0, 0,-1 ])
    m4.shape=(3,3)
  
    return m1,m2,m3,m4

def translation():
    """
    translational symmetry
    """
    symop=[]
    tmp=np.array([0,0,0])
    symop.append(tmp)
    for i1 in [-1,0,1]:
        for i2 in [-1,0,1]:
            for i3 in [-1,0,1]:
                tmp=np.array([i1,i2,i3])
                symop.append(tmp)
    return symop

def cubesymop():
    """
    cubic symmetry operations
    """
    (m1,m2,m3,m4)=cubegmat()
    symop=[]
    for m in range(2):
        for l in range(3):
            for k in range(2):
                for j in range(2):
                    s1=_matrixpow(m1,j) # mirror
                    s2=_matrixpow(m2,k) # c2
                    s3=_matrixpow(m3,l) # c3
                    s4=_matrixpow(m4,m) # inversion
                    tmp=np.dot(s4,s3)
                    tmp=np.dot(tmp,s2)
                    tmp=np.dot(tmp,s1)
                    symop.append(tmp)
    return symop

##########
# common #
##########

def _matrixpow(ma,n):
    ma=np.array(ma)
    (mx,my)=ma.shape
    if mx==my:
        if n==0:
            return np.identity(mx)
        elif n<0:
            return np.zeros((mx,mx))
        else:
            tmp=np.identity(mx)
            for i in range(n):
                tmp=np.dot(tmp,ma)
            return tmp
    else:
        print('matrix has not regular shape')
        return
    
def site_symmetry(xyz, verbose=0):
    """
    Symmetry operators in the site symmetry group G and its left coset decomposition.
    
    Args:
        xyz (numpy.ndarray):
            xyz coordinate of the site.
            The shape is (3).
        verbose (int)
    
    Returns:
        List of index of symmetry operators of the site symmetry group G (list):
            The symmetry operators leaves xyz identical.
        
        List of index of symmetry operators in the left coset representatives of the poibt group G (list):
            The symmetry operators generates equivalent positions of the site xyz.
    
    """
    def remove_overlaps_in_a_list(l1):
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
        """
        find overlap or not btween list1 and list2.
        
        Args:
            l1 (list):
            l2 (list):
        
        Returns:
            0 (int): no intersection
            1 (int): intersection
        """
        l3=remove_overlaps_in_a_list(l1+l2)
        if len(l1)+len(l2)==len(l3): # no overlap
            return 0
        else:
            return 1
    
    symop=cubesymop()
    traop=translation()
    
     # List of index of symmetry operators of the site symmetry group G.
     # The symmetry operators leaves xyz identical.
    list1=[]
    
    # List of index of symmetry operators which are not in the G.
    list2=[]
    
    for i2 in range(len(symop)):
        flag=0
        for i1 in range(len(traop)):
            xyz1=np.dot(symop[i2],xyz)
            if abs(xyz1[0]+traop[i1][0]-xyz[0])<EPS and abs(xyz1[1]+traop[i1][1]-xyz[1])<EPS and abs(xyz1[2]+traop[i1][2]-xyz[2])<EPS:
                list1.append(i2)
                flag+=1
                break
            else:
                pass
        if flag==0:
            list2.append(i2)
    
    list1_new=remove_overlaps_in_a_list(list1)
    list2_new=remove_overlaps_in_a_list(list2)
    
    if verbose>0:
        print(' site coordinates: %3.2f %3.2f %3.2f'%(xyz[0],xyz[1],xyz[2]))
        print('     multiplicity:',len(list1_new))
        print('    site symmetry:',list1_new)
    else:
        pass
    
    if int(len(symop)/len(list1_new))==1:
        list5=[0]
        if verbose>0:
            print('       left coset:',list5)
        else:
            pass
    
    else:
        # coset decomposition:
        list4=[]
        for i2 in list2_new:
            list3=[]
            for i1 in list1_new:
                op1=np.dot(symop[i2],symop[i1]) # left coset
                for i3 in range(len(symop)):
                    if np.all(op1==symop[i3]):
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
                    if find_overlaps(a,b)==0:
                        d=a+b
                        list5.append(list2_new[i3])
                    else:
                        pass
                else:
                    if find_overlaps(d,b)==0:
                        d=d+b
                        list5.append(list2_new[i3])
                    else:
                        pass
                i3+=1
            b=remove_overlaps_in_a_list(d)
            if int(len(symop)/len(list1_new))==len(list5):
                if verbose>0:
                    print('       left coset:',list5)
                else:
                    pass
                break
            else:
                pass
    
    return list1_new, list5
    
if __name__ == '__main__':
    
    (m1,m2,m3,m4)=cubegmat()
    """
    print('----mirror----')
    print(m1)
    print('------c2------')
    print(m2)
    print('------c3------')
    print(m3)
    print('---inversion--')
    print(m4)
    """
    
    # site symmetry
    xyz_3c=np.array([0.0, 1/2, 1/2])
    xyz_1b=np.array([1/2, 1/2, 1/2])
    xyz=xyz_3c
    #xyz=xyz_1b
    
    verbose=1
    #verbose=0
    
    list1,list2=site_symmetry(xyz,verbose)

