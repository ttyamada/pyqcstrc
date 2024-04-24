#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import timeit
import os
import sys
import numpy as np
import pyqcstrc.dode.occupation_domain as od
import pyqcstrc.dode.math12 as math12
import pyqcstrc.dode.symmetry12 as symmetry12
import pyqcstrc.dode.intsct12 as intsct12
import pyqcstrc.dode.utils12 as utils12
import pyqcstrc.dode.strc12 as strc12
import pyqcstrc.dode.numericalc12 as numericalc12

EPS=1e-6

def translation():
    """
    translational symmetry
    """
    symop=[]
    tmp=np.array([0,0,0,0,0,0])
    symop.append(tmp)
    for i1 in [-1,0,1]:
        for i2 in [-1,0,1]:
            for i3 in [-1,0,1]:
                for i4 in [-1,0,1]:
                    tmp=np.array([i1,i2,i3,i4,0,0])
                    symop.append(tmp)
    return symop
    
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
    
    symop=symmetry12.dodesymop()
    traop=translation()
    
     # List of index of symmetry operators of the site symmetry group G.
     # The symmetry operators leaves xyz identical.
    list1=[]
    
    # List of index of symmetry operators which are not in the G.
    list2=[]

    xyi=numericalc12.projection_1_numerical(xyz[0],xyz[1],xyz[2],xyz[3],xyz[4],xyz[5])
    xi=xyi[3]
    yi=xyi[4]
    if verbose>0:
        print(' site coordinates: %3.2f %3.2f %3.2f %3.2f'%(xyz[0],xyz[1],xyz[2],xyz[3]))
        print('         in Epar : %5.3f %5.3f'%(xyi[0],xyi[1]))
        print('         in Eperp: %5.3f %5.3f'%(xyi[3],xyi[4]))
    else:
        pass
    
    for i2 in range(len(symop)):
        flag=0
        for i1 in range(len(traop)):
            xyz1=np.dot(symop[i2],xyz)
            xyz2=xyz1+traop[i1]
            a=numericalc12.projection_1_numerical(xyz2[0],xyz2[1],xyz2[2],xyz2[3],xyz2[4],xyz2[5])
            if abs(a[3]-xi)<EPS and abs(a[4]-yi)<EPS:
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
        # left coset decomposition:
        list4=[]
        for i2 in list2_new:
            list3=[]
            for i1 in list1_new:
                op1=np.dot(symop[i2],symop[i1])
                for i3 in range(len(symop)):
                    if np.all(op1==symop[i3]):
                        list3.append(i3)
                        break
                    else:
                        pass
            list4.append(list3)
        
        #print('----------------')
        #for i2 in range(len(list4)):
        #    print(list4[i2])
        #print('----------------')
        
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
        
def symop_equivalent_par_site_sym(v0,list1):
    """
    generate equivalent vectors of v0 in Epar under site symmetry group, G.
    """
    
    v0prjk=numericalc12.projection_numerical(v0)
    list3=[0] # num of identical operator
    keep=[v0prjk]
    for i1 in list1:
        v1=symmetry12.generator_obj_symmetric_vec_0(v0,i1)
        v1prjk=numericalc12.projection_numerical(v1)
        if abs(v0prjk[0]-v1prjk[0])<EPS and abs(v0prjk[1]-v1prjk[1])<EPS:
            pass
        else:
            flg=0
            for i2 in range(len(keep)):
                vkeep=keep[i2]
                if abs(vkeep[0]-v1prjk[0])<EPS and abs(vkeep[1]-v1prjk[1])<EPS:
                    flg+=1
                    break
                else:
                    pass
            if flg==0:
                keep.append(v1prjk)
                list3.append(i1)
    return list3

if __name__ == '__main__':
    
    xyz_1a=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    xyz_6a=np.array([0.0, 1/2, 0.0, 0.0, 0.0, 0.0])
    xyz_4b=np.array([0.0, 2/3, 0.0, 1/3, 0.0, 0.0])
    xyz_3a=np.array([0.0, 1/2, 1/2, 0.0, 0.0, 0.0])
    xyz_6b=np.array([1/2, 0.0, 0.0, 1/2, 0.0, 0.0])
    #xyz=xyz_1a
    #xyz=xyz_6a
    #xyz=xyz_4b
    #xyz=xyz_3a
    xyz=xyz_6b
    
    #verbose=0
    verbose=1
    
    # list1: symmetry operator of the site symmetry group, G
    # list2: symmetry operator of left coset representative of the G.
    list1,list2=site_symmetry(xyz,verbose)
    
    # generate equivalent vectors in Epar centred at high symmetricap position.
    v1=np.array([[-1, 0, 8],[ 0, 0, 1],[ 0, 0, 1],[ 1, 0, 8],[ 0, 0, 1],[ 0, 0, 1]]) # (1,0,0,1)/8
    v2=np.array([[ 0, 0, 1],[ 3, 0, 8],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # (0,3,0,0)/8
    v3=np.array([[ 0, 0, 1],[ 1, 0, 4],[ 1, 0, 4],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]]) # (0,1,1,0)/4
    
    (xyz,v0)=(xyz_6b,v1)
    #(xyz,v0)=(xyz_3a,v2)
    #(xyz,v0)=(xyz_3a,v3)
    
    list3=symop_equivalent_par_site_sym(v0,list1)
    
    for i in list3:
    #for i in list1:
        v=symmetry12.generator_obj_symmetric_vec_0(v0,i)
        print('%d %d %d, %d %d %d, %d %d %d, %d %d %d'%(v[0][0],v[0][1],v[0][2],v[1][0],v[1][1],v[1][2],v[2][0],v[2][1],v[2][2],v[3][0],v[3][1],v[3][2]))
    
    """
    symmetry_operators=symmetry12.dodesymop()
    
    print(' site coordinates: %3.2f %3.2f %3.2f'%(xyz[0],xyz[1],xyz[2]))
    print('     multiplicity:',len(list1))
    print('    site symmetry:',list1)
    print('       left coset:',list2)
    
    for i in list1:
        print(i)
        print(symmetry_operators[i])
        print('')
    print(list1)
    """