#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

import timeit
import os
import sys
import numpy as np
import cython

import crsys as crs
import qnnum as qnn
import qnndarray as qna
import qnvec as qnv
import qnmat as qnm
import math1 as mth
import numeric as num
import utils as utl
import qnsym as qns
import intsct as isct
import prjop as prj
import lattice as lt
import sitesym as ssm
from vesta import (write_vesta,write_xyz)
from numpy.typing import (NDArray)
    
#except ImportError:
#    print('import error\n')

#TAU=np.sqrt(3)/2.0

def occdom_init():
    isys=crs.isys
    n=crs.n
    N=crs.N

def volume(obj:qnv.Qnvec):
    return utl.obj_area_nd(obj)

#def symmetric(obj: qna.QnNdarray, centre:qnv.Qnvec, png:str):
def symmetric_od(irs: NDArray[np.int64], obj: qna.QnNdarray):
    """
    Generate symmterical occupation domain by site-symmetry elements of OD center
    
    Args:
        obj (numpy.ndarray):
            Asymmetric unit of the occupation domain
            The shape is (num,3,5) or (num,4,6), where num=numbre_of_triangles or tetrahedra
        inr : indices of site-symmetry operators
        #centre (numpy.ndarray):
        #    nd coordinate of the symmetric centre.
        #    The shape is (n)
        #pg (string):
        #    point group, '12/mmm', '-12m2', '-12', '12'
    Returns:
        Symmetric occupation domains (qnndarray):
            The shape is (num,3,6,3), where num=numbre_of_tetrahedron.
    
    """
    #print("obj.ndim",obj.ndim,"obj.shape",obj.shape) # for test
    # shape[0] : number of triangles or tetrahedra
    # shape[1] : numbder of points 3 for triangle 4 for tetrahedra 
    # shape[2] : space dimension 5 for dihed 6 for icos
    
    print("irs",irs)  # for test
    shape=obj.shape
    ndim=len(shape)
    print("shape",shape,"ndim in symmetric_od",ndim)  # for test
    if ndim==3 or ndim==4:
        #return symmetry.generator_obj_symmetric_tetrahedron(obj,centre)
        return generator_obj_symmetric_obj(irs,obj)
    else:
        print('object has an incorrect shape in symmetric!')
        return 
    
#def generator_obj_symmetric_obj(obj:qna.QnNdarray, centre:qnv.Qnvec, pg:str):
def generator_obj_symmetric_obj(irs: NDArray[np.int64], obj:qna.QnNdarray):
    """
    arrguments
    obj : vertices of asymmetric od (shape=(num,3,5) or (num,4,6) for num triangles or tetrahedra)
    irs : indices of site symmetry operators 

    calculate site-symmetry of od using symmetry operators
    """
    shape=obj.shape  # (num,3,5) or (num,4,6) expected for dihed or icos
    ndim=len(shape)  #dimension of obj 3 expected
    print("shape",shape,"ndim in generateor_obj_symmetric_obj",ndim)
    n=crs.n
    if ndim==3 or ndim==4:
        # generate equivalent symmetric od vertices 
        nss=len(irs)
        a=symmetric(irs,obj)  #  rotated obj (rotated triangle vertices)
        #if obj.ndim==4:
        #    n1,n2,_,_=obj.shape
        #    a=a.reshape(nss*n1,n2,n)
        return a
    else:
        print('shape in generator_obj_symmetric_obj should be 3 or 4!')
        return

##def generator_obj_symmetric_triangle(obj:qnv.Qnvec, centre:qnv.Qnvec, pg:str):
#def generator_obj_symmetric_triangle(obj:qnv.Qnvec, centre:qnv.Qnvec):
#    """
#    """
#    return generator_obj_symmetric_obj(obj,centre)

def symmetric_i(i1,obj):
    shape=obj.shape  # (num,3,n) or (num,4,n) assumed
    qnr0=qns.qnr0[i1]  # rotation matrix
    a=qna.Qnndarray(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            a[i][j]=qnr0@obj[i][j]
    return a

def symmetric(irs:NDArray[np.int64],obj:qna.QnNdarray):
    nsy=len(irs)
    sp0=obj.shape # (num,3,5) or (num,4,6) assumed
    print("sp0",sp0)
    shape=(nsy*sp0[0],sp0[1],sp0[2])  # (nsy,num,3,n) or (nsy,num,4,6)
    print("shape in symmetric",shape)
    qnr_i=qns.qnr_i
    a=qna.QnNdarray(shape)
    ni=0
    for n in range(nsy):
        for i in range(sp0[0]):
            for j in range(sp0[1]):
                #print("i",i,"j",j)  # for test
                #qnv.printqnv("obj",obj[i][j])  # for test 
                a[ni][j]=qnr_i[irs[n]]@obj[i][j]
                print("%s-th triangle %s-th vertex"%(ni,j),end="") # for test
                qnv.printqnv(" ",a[ni][j])  # for test
            ni+=1
    return a

#def generator_obj_symmetric_vector_specific_symop(obj:qnv.Qnvec,centre:qnv.Qnvec,irs:np.int64,pg:str):
def generator_obj_symmetric_vector_specific_symop(obj:qnv.Qnvec,centre:qnv.Qnvec,irs:np.int64):
    """
    vector: triangles
    (6,3)
    """       
    nss=len(irs)
    shape=tuple(nss,n,n)
    # using specific symmetry operations
    if obj.ndim==2:
        a=qna.QnNdarray(shape) # rotation operator matrices
        #a=np.zeros(shape+obj.shape,dtype=np.int64)
        j=0
        for i1 in irs:
            a[j]=symmetric_i(i1,obj)  # return rotated obj by i1-th symmetry operator
            j+=1
        return a
    else:
        print('object has an incorrect shape!')
        return

#def generator_obj_symmetric_triangle_specific_symop(obj:qnv.Qnvec,centre:qnv.Qnvec,irs:np.int64,pg=None):
def generator_obj_symmetric_triangle_specific_symop(obj:qnv.Qnvec,centre:qnv.Qnvec,irs:np.int64):
    """
    triangle: triangles
    (3,6,3)
    """
    # using specific symmetry operations
    if obj.ndim==3:
        mop=qna.qnndarray()
        shape=tuple([len(irs)])
        a=np.zeros(shape+obj.shape,dtype=np.int64)
        j=0
        for i1 in irs:
            a[j]=symmetric_i(mop[i1],obj)
            j+=1
        return a
    else:
        print('object has an incorrect shape!')
        return

    #print("obj.ndim",obj.ndim) # for test
    if obj.ndim==3 or obj.ndim==4:
        return qns.generator_obj_symmetric_triangle_0(obj,centre,irs,pg)
    else:
        print('object has an incorrect shape in symmetric_0!')
        return 
    

def shift(obj: qnv.Qnvec,shift : qnv.Qnvec):
    """
    Shift the occupation domain.
    
    Args:
        obj (numpy.ndarray):
            The occupation domain
            The shape is (num,3,6,3), where num=numbre_of_triangles.
        shift (numpy.ndarray):
            nd coordinate to which the occupation domain is shifted.
            The shape is (6,3)
        verbose (int):
            verbose = 0 (silent, default)
            verbose = 1 (normal)
    
    Returns:
        Shifted occupation domains (numpy.ndarray):
            The shape is (num,3,6,3), where num=numbre_of_tetrahedron.
    
    """
    return utl.shift_object(obj, shift)

def write(obj=None,path=None,basename=None,format=None,color='k',select=None,verbose=0):
    """
    Export occupation domains.
    
    Args:
        obj (numpy.ndarray): the occupation domain
            The shape is (num,3,6,3), where num=numbre_of_triangles.
        path (str): Path of the output XYZ file
        basename (str): Basename of the output XYZ file
        format (str): format of output file
            format = 'xyz' (default)
            format = 'vesta'
        color (str)
            one of the characters {'k','r','b','p'}, which are short-hand notations 
            for shades of black, red, blue, and pink, in case where 'vesta' format is
            selected (default, color = 'k').
        select (str):'simple', 'normal', or 'egdes'
            'simple': Merging triangles into one single objecte
            'normal': Each triangle is set as single objecte (large file)
            'egdes':  Select this option when the obj is a set of edges.
    
    Returns:
        int: 0 (succeed), 1 (fail)
    
    """
    
    if os.path.exists(path)==False:
        os.makedirs(path)
    else:
        pass
    
    if np.all(obj==None):
        print('    Empty OD')
        return 0
    else:
        if format=='vesta':
            if select==None:
                select='normal'
            write_vesta(obj,path,basename,color,select,verbose)
            return 0
        elif format == 'xyz':
            if select==None:
                select='triangle'
            write_xyz(obj,path,basename,select,verbose)
            return 0
        else:
            return 1



def simplification(obj,verbose=0):
    """
    Simplification of occupation domains.
    
    Args:
        obj (numpy.ndarray): the occupation domain
            The shape is (num,3,6,3), where num=numbre_of_trianges.
        num_cycle (int): numbre of cycles
        verbose (int)
            verbose = 0 (silent, default)
            verbose = 1 (normal)
            verbose > 2 (detail)
    
    Returns:
    
        Simplified occupation domains (numpy.ndarray)
            The shape is (num,3,6,3), where num=numbre_of_tetrahedron.
    
    """
    if np.all(obj==None):
        if verbose>0:
            print('    zero volume')
        return 
    else:
        vol0=utl.obj_area_nd(obj)
        obj_convex_hull=utl.generate_convex_hull(obj)
        obj_tmp=isct.intersection_two_obj_1(obj_convex_hull,obj)
        vol1=utl.obj_area_nd(obj_tmp)
        if np.all(vol0==vol1):
            if verbose>0:
                print('      simplification succeed:')
                print('      num of tetrahedra: %d --> %d'%(len(obj),len(obj_convex_hull)))
            return obj_convex_hull
        else:
            if verbose>0:
                print('      simplification: fail')
            return obj

def generate_border_edges(obj):
    """
    Generate border edges of the occupation domain.
    
    Args:
        obj (numpy.ndarray):
            The occupation domain
            The shape is (num,3,6,3), where num=numbre_of_tetrahedron.
    
    Returns:
        Border edges of the occupation domains (numpy.ndarray):
            The shape is (num,2,6,3), where num=numbre_of_edge.
    
    """
    triangle_surface=utl.generator_surface_1(obj)
    return utl.surface_cleaner(triangle_surface)

def outline(obj):
    """
    Generate outline of the occupation domain.
    
    Args:
        obj (numpy.ndarray): the shape is (num,3,6,3), where num=numbre_of_triangle.
    
    Returns:
        Outline of the occupation domain (numpy.ndarray):
            The shape is (num,2,6,3), where num=number of the outlines.
    
    """
    return utl.surface_cleaner(obj)
    
# new in version 0.0.2a2
def obj2podatm(obj,serial_number=1,path='.',basename='tmp',shift=[0,0,0,0,0,0]):
    
    def find_common_vertex(obj):
        #Find common vertex of tetrahedra in obj).
        counter1=0
        for i1 in [0,1,2]:
            vtx1=obj[0][i1]
            xyz1=prj.projection3(vtx1)
            counter2=0
            for i2 in range(1,len(obj)):
                counter3=0
                for i3 in [0,1,2]:
                    xyz2=prj.projection3(obj[i2][i3])
                    if np.all(xyz1==xyz2):
                        counter3=1
                        break
                if counter3==1:
                    counter2+=1
                else:
                    break
            if counter2==len(obj)-1:
                counter1=1
                break
            else:
                pass
        if counter1!=0:
            return vtx1
        else:
            return 
    
    # common vertex
    vrtx0=find_common_vertex(obj)
    
    if np.all(vrtx0!=None):
        
        if os.path.exists(path) == False:
            os.makedirs(path)
        else:
            pass
        fatm=open('%s/%s.atm'%(path,basename),'w', encoding="utf-8", errors="ignore")
        fpod=open('%s/%s.pod'%(path,basename),'w', encoding="utf-8", errors="ignore")
        
        #--------
        #  atm
        #--------
        fatm.write('%d \'Em\' 1 %d 1 2.0 0. 0. 1.0 0. 0. 0.\n'%(serial_number,serial_number))
        
        vn=num.numerical_vector(vrtx0)
        fatm.write('x=  %4.3f  %4.3f  %4.3f  %4.3f  %4.3f  %4.3f\n'%(\
        vn[0],vn[1],vn[2],vn[3],vn[4],vn[5]))
        
        # generate a list of verices and remove the common vertex from it.
        vtxs=utl.remove_doubling_in_perp_space(obj)
        vtxs=utl.remove_vector(vtxs,vrtx0)
        
        #--------
        #  pod
        #--------
        fpod.write('%d %d %d \'comment\'\n'%(serial_number,len(vtxs),2))
        for vtx in vtxs:
            vn=num.numerical_vector(vtx)
            fpod.write('ej=  %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f\n'%(\
            vn[0],vn[1],vn[2],vn[3],vn[4],vn[5]))
        #
        lst_indx=[]
        for triangle in obj:
            for vrtx1 in triangle:
                for i1 in range(len(vtxs)):
                    if np.all(vrtx1==vtxs[i1]):
                        lst_indx.append(i1+1) # add 1 to avoide index 0.
                        break
                    else:
                        pass
        fpod.write('nth= %d'%(len(vtxs)))
        for indx in lst_indx:
            fpod.write(' %d'%(indx))
        fpod.write('\n')
        
        return 0
    
    else:
        print('No common vertex found in the object. pod and atm cannot be created.')
        return 1

##### WIP
def write_podatm(obj, position, vlist=[0], path='.', basename='tmp', shift=[0., 0., 0., 0., 0., 0.], verbose=0):
    """
    Generate pod and atom files.
    
    Args:
        obj (numpy.ndarray): the occupation domain
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.
        position (numpy.ndarray): nd coordinates of the position of the occupation domain.
        vertices (list): 
        vlist (list): 
        path (str): Path of the output files
        basename (str): Basename of the output files
        verbose (int):
            verbose = 0 (silent, default)
            verbose = 1 (normal)
    
    Returns:
        int: 0 (succeed), 1 (fail)
    
    """
    
    if os.path.exists(path) == False:
        os.makedirs(path)
    else:
        pass
    
    if obj.tolist()==[[[[0]]]] or obj.tolist()==[[[0]]]:
        print('no volume obj')
        return 0
    else:
        fatm=open('%s/%s.atm'%(path,basename),'w', encoding="utf-8", errors="ignore")
        fpod=open('%s/%s.pod'%(path,basename),'w', encoding="utf-8", errors="ignore")
    
        """
        # get independent edges
        edges = utl.generator_obj_edge(obj, verbose-1)
        # get independent vertices of the edges
        v = utl.remove_doubling_dim4_in_perp_space(edges)
        """
        #v=vertices
        v=obj
    
        # shift
        #shft=[0.00001,0.00002,0.00000,-0.00001,0.00001,-0.00002]
        #shft=[0.0,0.0,0.0,0.0,0.0,0.0]
        shft=shift
        # .atm file
        for i in range(len(vlist)):
            a=v[vlist[i][0]-1]
            fatm.write('%d \'Em\' 1 %d 1 2.0 0. 0. 1.0 0. 0. 0.\n'%(i+1,i+1))
            #fatm.write('x=  %4.3f  %4.3f  %4.3f  %4.3f  %4.3f  %4.3f\n'%(\
            #(position[0][0]+position[0][1]*TAU)/(position[0][2]),\
            #(position[1][0]+position[1][1]*TAU)/(position[1][2]),\
            #(position[2][0]+position[2][1]*TAU)/(position[2][2]),\
            #(position[3][0]+position[3][1]*TAU)/(position[3][2]),\
            #(position[4][0]+position[4][1]*TAU)/(position[4][2]),\
            #(position[5][0]+position[5][1]*TAU)/(position[5][2])))
            p=num.numerical_vector(position)
            fatm.write('x=  %4.3f  %4.3f  %4.3f  %4.3f  %4.3f  %4.3f\n'%(\
            p[0],p[1],p[2],p[3],p[4],p[5]))
            
            fatm.write('xe1= 0.  0.  0.  0.  0.  0.  0. u1=0.0            \n')
            fatm.write('xe2= 0.  0.  0.  0.  0.  0.  0. u2=0.0            \n')
            fatm.write('xe3= 0.  0.  0.  0.  0.  0.  0. u3=0.0            \n')
            #fatm.write('xi=  %8.6f  %8.6f  %8.6f  %8.6f  %8.6f  %8.6f  0.000000  v=1.0\n'%(\
            #(a[0][0]+a[0][1]*TAU)/(a[0][2])+shft[0],\
            #(a[1][0]+a[1][1]*TAU)/(a[1][2])+shft[1],\
            #(a[2][0]+a[2][1]*TAU)/(a[2][2])+shft[2],\
            #(a[3][0]+a[3][1]*TAU)/(a[3][2])+shft[3],\
            #(a[4][0]+a[4][1]*TAU)/(a[4][2])+shft[4],\
            #(a[5][0]+a[5][1]*TAU)/(a[5][2])+shft[5]))
            a=num.numerical_vector(a)
            fatm.write('xi=  %8.6f  %8.6f  %8.6f  %8.6f  %8.6f  %8.6f  0.000000  v=1.0\n'%(\
            a[0],a[1],a[2],a[3],a[4],a[5]))
            fatm.write('isyd=1\n')
        fatm.close()
    
        # .pod file
        #fpod.write('nsymo=3 icent=1 brv=\'p\' io=%d\n'%(len(vlist)))
        #fpod.write('symmetry operator\n')
        #fpod.write('y,z,u,−x+z,v,          12f\n')
        #fpod.write('x,y−u,x−z,−u,−v        my\n')
        #fpod.write('x,y,z,u,−v             mz\n')
        for i1 in range(len(vlist)):
            a=v[vlist[i1][0]-1]
            tmp2=[vlist[i1][1]-1]
            for i2 in range(2,len(vlist[i1])):
                counter=0
                for i3 in range(len(tmp2)):
                    if vlist[i1][i2]-1==tmp2[i3]:
                        counter+=1
                        break
                    else:
                        pass
                if counter==0:
                    tmp2.append(vlist[i1][i2]-1)
                else:
                    pass
            #print('tmp2=',tmp2)
        
            tmp1=[]
            for i2 in range(1,len(vlist[i1])):
                for i3 in range(len(tmp2)):
                    if vlist[i1][i2]-1==tmp2[i3]:
                        tmp1.append(i3)
                        break
                    else:
                        pass
            #print('tmp1=',tmp1)
        
            fpod.write('%d %d %d \'comment\'\n'%(i1+1,len(tmp2),2))
            for i2 in range(len(tmp2)):
                b=v[tmp2[i2]]
                #fpod.write('ej=  %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f 0.00000\n'%(\
                #(b[0][0]+b[0][1]*TAU)/(b[0][2])-(a[0][0]+a[0][1]*TAU)/(a[0][2]),\
                #(b[1][0]+b[1][1]*TAU)/(b[1][2])-(a[1][0]+a[1][1]*TAU)/(a[1][2]),\
                #(b[2][0]+b[2][1]*TAU)/(b[2][2])-(a[2][0]+a[2][1]*TAU)/(a[2][2]),\
                #(b[3][0]+b[3][1]*TAU)/(b[3][2])-(a[3][0]+a[3][1]*TAU)/(a[3][2]),\
                #(b[4][0]+b[4][1]*TAU)/(b[4][2])-(a[4][0]+a[4][1]*TAU)/(a[4][2]),\
                #(b[5][0]+b[5][1]*TAU)/(b[5][2])-(a[5][0]+a[5][1]*TAU)/(a[5][2])))
                b=mth.sub_vectors(b,a)
                b=num.numerical_vector(b)
                """ 5次元ベクトルから7次元ベクトルへの変換　一意に決まらない!?
                
                fpod.write('ej=  %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f\n'%(\
                #b[0],b[1],b[2],b[3],b[4],b[5]))
                
                if b[1]>0.0 and b[3]>0.0:
                    if b[1]-b[3]>0.0:
                        e3=b[3]
                        e1=b[1]-b[3]
                    elif b[1]-b[3]<0.0:
                        e3=b[1]
                        e1=b[3]-b[1]
                    else:# b[1]-b[3]==0.0
                        e3=b[1]
                elif b[1]<0.0 and b[3]<0.0:
                    if b[1]-b[3]<0.0:
                        e3=b[1]
                        e1=b[3]-b[1]
                    elif b[1]-b[3]>0.0:
                        e3=b[3]
                        e1=b[1]-b[3]
                    else:# b[1]-b[3]==0.0
                        e3=b[1]
                """
                
                fpod.write('ej=  %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f %8.6f\n'%(\
                b[1],b[0]+b[2],b[1]+b[3],b[2],b[3],-b[0],b[4]))
                
                """
                for i3 in range(6):
                    if i3==0:
                        fpod.write('ej=  %8.6f'%(\
                        (b[i3][0]+b[i3][1]*TAU)/(b[i3][2]) - (a[i3][0]+a[i3][1]*TAU)/(a[i3][2])\
                        )
                    elif i3==5:
                        fpod.write(' %8.6f\n'%(\
                        (b[i3][0]+b[i3][1]*TAU)/(b[i3][2]) - (a[i3][0]+a[i3][1]*TAU)/(a[i3][2])\
                        )
                    else:
                        fpod.write(' %8.6f'%(\
                        (b[i3][0]+b[i3][1]*TAU)/(b[i3][2]) - (a[i3][0]+a[i3][1]*TAU)/(a[i3][2])\
                        )
                """
            fpod.write('nth= %d'%(int((len(vlist[i1])-1)/3)+1))
            for i2 in range(len(tmp1)):
                fpod.write(' %d'%(tmp1[i2]+1))
            fpod.write('\n100000000000000000000000\n')
        fpod.close()
    
        if verbose>0:
            print('    written in %s/%s.atm'%(path,basename))
            print('    written in %s/%s.pod'%(path,basename))
    
    return 0

#########################
#          WIP          #
#########################
def simple_hand_step1(obj, path, basename_tmp):
    """
    Simplification of occupation domains by hand (step1).
    
    Args:
        obj (numpy.ndarray): the occupation domain
            The shape is (num,3,6,3), where num=numbre_of_triangles.
        path (str): path of the tmporal file
        basename_tmp (str): name for tmporal file.
    
    Returns:
    
        Tmporal occupation domains (numpy.ndarray):
            The shape is (num,3,6,3), where num=numbre_of_triangles.
    
    """
    def write_xyz_smpl(a, path, basename):
        f=open('%s'%(path)+'/%s.xyz'%(basename),'w', encoding="utf-8", errors="ignore")
        f.write('%d\n'%(len(a)))
        f.write('%s\n'%(basename))
        for i1 in range(len(a)):
            xyz=prj.projection3(a[i1])
            f.write('Xx %8.6f %8.6f %8.6f # %d-th vertex # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
            (num.numeric_value(xyz),\
            num.numeric_value(xyz),\
            num.numeric_value(xyz),\
            i1,\
            a[i1][0][0],a[i1][0][1],a[i1][0][2],\
            a[i1][1][0],a[i1][1][1],a[i1][1][2],\
            a[i1][2][0],a[i1][2][1],a[i1][2][2],\
            a[i1][3][0],a[i1][3][1],a[i1][3][2],\
            a[i1][4][0],a[i1][4][1],a[i1][4][2],\
            a[i1][5][0],a[i1][5][1],a[i1][5][2]))
        f.closed
        return 0
        
    od1a=utl.remove_doubling_in_perp_space(obj)
    write_xyz_smpl(od1a, path, basename_tmp)
    print('written in %s'%(path)+'/%s.xyz'%(basename_tmp))
    print('open above XYZ file in vesta and make merge_list, and run simple_hand_step2()')
    return od1a

def simple_hand_step2(obj, merge_list):
    """
    Simplification of occupation domains by hand (step2).
    
    Args:
        obj (numpy.ndarray): the occupation domain
            The shape is (num,3,6,3), where num=numbre_of_triangles.
        merge_list (list[[int,int,int,int,],[],...,[]])
            A list containing lists of indices of vertices of triangle.
            The indices of vertices of triangle in temporal file obtaind
            by 'simple_hand_step1()'.
    
    Returns:
        Simplified occupation domains (numpy.ndarray):
            The shape is (num,4,6,3), where num=numbre_of_tetrahedron.
    
    """
    
    def merge(obj,mylist):
        tmp1=np.array([obj[mylist[0]-1]])
        for i in range(1,len(mylist)):
            tmp2=obj[mylist[i]-1]
            tmp1=np.append(tmp1,tmp2)
        return tmp1.reshape(len(mylist),6,3)
    
    for i in range(len(merge_list)):
        tmp1=merge(obj,merge_list[i])
        od2=isct.tetrahedralization_points(tmp1)
        if i==0:
            od1=np.array(od2)
        else:
            od1=np.vstack([od1,od2])
    return od1

def similarity(obj,m):
    """
    obj:
    m: order of similarity transformation
    """
    return qns.similarity_obj(obj,m)

def qcstrc(apar,cpar,mystrc,path,basename,phason_matrix,n1max,n5max,origin_shift,option=0,pg='-12m2',verbose=0):
    """
    mystrc
    
    """
    apar=apar*2/np.sqrt(6)
    dim=5
    v0=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]]) # (0,0,0,0)
    
    objs=[]
    pos=[]
    atm=[]
    eshift=[]
    
    """ # old ver.
    ########## From HERE ##########
    for strc in mystrc:
        obj1,wsite,atom,shift=strc
        
        dim=5
        v0=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]]) # (0,0,0,0)
        
        #-----------------------------------------------------------------------
        # generate independent occypation domains from their asymmetric units
        #-----------------------------------------------------------------------
        num_coset=symmetry.coset(wsite,dim)
        tmp=symmetry.generator_obj_symmetric_obj(obj1,wsite)
        num=len(tmp)
        tmp=symmetry.generator_obj_symmetric_obj_specific_symop(tmp,v0,num_coset)
        objs1=tmp.reshape(len(num_coset),num,3,6,3)
        
        #-----------------------------------------------------------------------
        # generate positions of the independent occypation domains
        #-----------------------------------------------------------------------
        pos1=symmetry.generator_obj_symmetric_vector_specific_symop(wsite,v0,num_coset)
        
        objs.append(objs1)
        pos.append(pos1)
        atm.append(atom)
        eshift.append(shift)
    ########## To HERE ##########
    """
    
    ########## From HERE ##########
    print('  point group:',pg)
    for strc in mystrc:
        obj,wsite,atom,shift=strc
        wsiten=num.numerical_vector(wsite)
        print('   wsite: %4.3f %4.3f %4.3f %4.3f %4.3f'%(wsiten[0],wsiten[1],wsiten[2],wsiten[3],wsiten[4]))
        #num_stsym=symmetry.site_symmetry(wsite,dim,pg)
        #num_coset=symmetry.coset(wsite,dim,pg)
        #num_stsym,num_coset=symmetry.site_symmetry_and_coset(site=wsite,brv='p',pg=pg,verbose=0)
        num_stsym=qns.site_symmetry(site=wsite,brv='p',pg=pg)
        eqposs,lst_idx_eqposs=qns.equivalent_positions_in_unit_cell(site=wsite,brv='p',pg=pg,vervose=1)
        #num_coset=qns.coset_a(wsite,dim,pg)
        print('    num_sisym:',num_stsym)
        print('     lst_idx_eqposs',lst_idx_eqposs)
        print('     eqposs.shape:',eqposs.shape)
        #print('    num_coset:',num_coset)
        #num_coset=num_coset[17]
        #num_equiv=qns.equivalent_positions(wsite,dim,pg)
        #print('   num_equiv:',num_equiv)
        
        
        
        #-----------------------------------------------------------------------
        # generate independent occupation domains from their asymmetric units
        #-----------------------------------------------------------------------
        obj1=qns.generator_obj_symmetric_obj_specific_symop(obj,wsite,num_stsym,pg)
        for i2,pos2 in enumerate(eqposs):
        #for i2 in num_equiv:
            #--------------------------------------------------------------------------------
            # place the independent occupation domain at each position equivalent to "wsite"
            #--------------------------------------------------------------------------------
            obj2=qns.generator_obj_symmetric_obj_specific_symop(obj1,v0,[lst_idx_eqposs[i2]],pg)
            print('obj2.shape:',obj2.shape)
            print('pos2.shape:',pos2.shape)
            #pos2=qns.generator_obj_symmetric_vector_specific_symop(wsite,v0,[i2],pg)
            #
            objs.append(obj2.reshape(1,len(obj2),3,6,3))
            #objs.append(obj2)
            pos.append(pos2)
            atm.append(atom)
            eshift.append(shift)
        
        """
        for i1 in num_stsym:
            #-----------------------------------------------------------------------
            # generate independent occupation domains from their asymmetric units
            #-----------------------------------------------------------------------
            obj1=qns.generator_obj_symmetric_obj_specific_symop(obj,wsite,[i1],pg)
            for i2,pos2 in enumerate(eqposs):
            #for i2 in num_equiv:
                #--------------------------------------------------------------------------------
                # place the independent occupation domain at each position equivalent to "wsite"
                #--------------------------------------------------------------------------------
                obj2=qns.generator_obj_symmetric_obj_specific_symop(obj1,v0,[lst_idx_eqposs[i2]],pg)
                print('obj2.shape:',obj2.shape)
                print('pos2.shape:',pos2.shape)
                #pos2=qns.generator_obj_symmetric_vector_specific_symop(wsite,v0,[i2],pg)
                #
                objs.append(obj2.reshape(1,len(obj2),3,6,3))
                pos.append(pos2)
                atm.append(atom)
                eshift.append(shift)
        """
    ########## To HERE ##########
    """
    ########## To HERE ##########
    for strc in mystrc:
        obj,wsite,atom,shift=strc

        #-------------------------------------------------------
        # generate an independent OD from its asymmetric unit.
        #-------------------------------------------------------
        obj1=qns.generator_obj_symmetric_obj(obj,wsite)
        
        #---------------------------------------------------------------
        # put the independent OD at each position equivalent to 'wsite'
        #---------------------------------------------------------------
        num_coset=qns.coset(wsite,dim)
        for i1 in num_coset:
            obj2=qns.generator_obj_symmetric_obj_specific_symop(obj1,v0,[i1])
            pos2=qns.generator_obj_symmetric_vector_specific_symop(wsite,v0,[i1])
            
            objs.append(obj2.reshape(1,len(obj2),3,6,3))
            pos.append(pos2)
            
            atm.append(atom)
            eshift.append(shift)
    ########## To HERE ##########
    """
    
    generated_strc=num.strc(objs,pos,phason_matrix,n1max,n5max,eshift,origin_shift,verbose)
    f=open('%s/%s.xyz'%(path,basename),'w', encoding="utf-8", errors="ignore")
    f.write('%d\n'%(len(generated_strc)))
    f.write('%s.xyz\n'%(basename))
    for b in generated_strc:
        if option==0: # x,y,z in Epar
            f.write('%s %8.6f %8.6f %8.6f\n'%(atm[int(b[1])],b[0][0]*apar,b[0][1]*apar,b[0][2]*cpar))
        elif option==1: # x,y,z in Epar, h1-h5
            f.write('%s %8.6f %8.6f %8.6f # %3d %3d %3d %3d %3d\n'%(atm[int(b[1])],b[0][0]*apar,b[0][1]*apar,b[0][2]*cpar,b[2],b[3],b[4],b[5],b[6]))
        elif option==2: # x,y,z in Epar, x,y,z in Eperp, h1-h5
            f.write('%s %8.6f %8.6f %8.6f # %8.6f %8.6f %8.6f %3d %3d %3d %3d %3d\n'%(\
                                    atm[int(b[1])],\
                                    b[0][0]*apar,b[0][1]*apar,b[0][2]*cpar,\
                                    b[0][3],b[0][4],b[0][5],\
                                    b[2],b[3],b[4],b[5],b[6],
                                    ))
        else:
            pass
    f.closed
    print('    written in %s/%s.xyz'%(path,basename))
    return 0

