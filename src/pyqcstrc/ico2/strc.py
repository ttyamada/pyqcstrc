#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import numpy as np
from numpy.typing import NDArray
import random
try:
    from pyqcstrc.ico2.numericalc import (numerical_vector,
                                          numerical_vectors,
                                          projection_numerical_par,
                                          #inside_outside_obj,
                                          inside_outside_tetrahedron,
                                          #inside_outside_tetrahedron_tau_v2,
                                          inside_outside_tetrahedron_rough,
                                          projection3_numerical,
                                          projection3_sets_numerical,
                                          projection_numerical,
                                          projection_sets_numerical,
                                          projection_sets_par_numerical,
                                          projection_sets_par_numerical_normalized,
                                          get_internal_component_numerical,
                                          get_internal_component_sets_numerical,
                                          length_numerical,
                                          )
    from pyqcstrc.ico2.symmetry_numerical import (generator_obj_symmetric_vector_specific_symop,
                                                  generator_obj_symmetric_vector_specific_symop_1,
                                                  generator_obj_symmetric_vectors_specific_symop_1,
                                                  generator_obj_symmetric_vectors_specific_symop,
                                                  generator_equivalent_numeric_vector_specific_symop,
                                                  )
    from pyqcstrc.ico2.symmetry import (generator_obj_symmetric_obj,
                                        generator_obj_symmetric_obj_specific_symop,
                                        site_symmetry_and_coset,
                                        icosasymop_array,
                                        icosasymop3_array,
                                        #equivalent_sites_unit_cell,
                                        equivalent_sites_with_centring,
                                        generator_equivalent_vec,
                                        get_index_of_symmetry_operation_for_equivalent_vectors,
                                        symmetry_operations_axial_vector,
                                        symmetry_operations_axial_vectors,
                                        )
    from pyqcstrc.ico2.math1 import (mul_vector,
                                     mul_vectors,
                                     projection,
                                     )
    from pyqcstrc.ico2.utils import (shift_object,
                                    )
except ImportError:
    print('import error in strc.py\n')

TAU=(1+np.sqrt(5))/2.0
EPS=1e-6
V0=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.int64)
V1=np.array([0, 0, 0, 0, 0, 0],dtype=np.float64)
V2=np.array([0, 0, 0],dtype=np.float64)
CONST1 = 1/np.sqrt(2.0+TAU)

def strc(aico,brv,model,nmax,oshift,x1,x2,x3,verbose,test_flag):
    """
    this function generates atomic and magnetic structures in 3-d physical space.
    
    input:
    
    oshift, list
    
    eshift, list
        [u1,u2,u3], shift along 5-, 3-, 2-fold axis in Epar (xe1,xe2,xe3).
    mu, list
        [mu1,mu2,mu3], mu along 5-, 3-, 2-fold axis in Epar (xe1,xe2,xe3).
    x1,x2,x3, list
        three 6d vectors for eshift and mu, which corresponds to xe1,xe2,xe3 in QUASI.
    verbose, 
    test_flag, for test
    """
    
    #x1=np.array([0, 0, -1, 0, 0, 0],dtype=np.float64) #5f
    #x2=np.array([0, 1, -1, 1, 0, 0],dtype=np.float64) #3f
    #x3=np.array([0, 1, -1, 0, 0, 0],dtype=np.float64) #2f
    
    #x1=np.array([1, 0, 0, 0, 0, 0],dtype=np.float64) #5f
    #x2=np.array([1, 0,-1, 0,-1, 0],dtype=np.float64) #3f
    #x3=np.array([1, 0, 0, 0,-1, 0],dtype=np.float64) #2f
    
    #x1=np.array([-1.,  0.,  0.,  0.,  0.,  0.],dtype=np.float64) #5f
    #x2=np.array([-1.,  0.,  0.,  1.,  0.,  1.],dtype=np.float64) #3f
    #x3=np.array([-1.,  0.,  0.,  1.,  0.,  0.],dtype=np.float64) #2f
    
    x1=np.array(x1,dtype=np.float64) #5f
    x2=np.array(x2,dtype=np.float64) #3f
    x3=np.array(x3,dtype=np.float64) #2f
    
    if brv=='s':
        aico=aico*2
    else:
        pass
        
    oshift=projection3_numerical(np.array(oshift,dtype=np.float64))
    if brv=='s':
        oshift=oshift/2
        
    print('Generating nD structure:')
    lst_shape=[]
    lst_objs=[]
    lst_pos=[]
    lst_atm=[]
    lst_be=[]
    lst_occ=[]
    lst_rmax=[]
    lst_mu=[]
    lst_xe1=[]
    lst_xe2=[]
    lst_xe3=[]
    lst_mxe1=[]
    lst_mxe2=[]
    lst_mxe3=[]
    lst_eshift=[]
    #lst_sphere_radius=[]
    
    #### TEST ####
    if test_flag==0:
        pass
    else:
        lst_6dspin=[]
        lst_mx1=[]
        lst_mx2=[]
        lst_mx3=[]
    ##############
    
    for i1,nod in enumerate(model):
        atom, pod, position, eshift, be, occ, rmax, mu = model[nod]
        obj=pod[1]
        if brv=='s':
            obj=mul_vectors(obj,np.array([1,0,2]))
            position=mul_vector(position,np.array([1,0,2]))
        else:
            pass
        indx_site_sym,indx_coset=site_symmetry_and_coset(position,brv,verbose)
        
        #========================================================================
        # generate positions of ODs necessary to generate atomic positions.
        #========================================================================
        """
        vn=numerical_vector(position)
        pos1=generator_equivalent_numeric_vector_specific_symop(vn,indx_coset)
        if verbose>0:
            print('  equivalent positions:')
            for i2,eqpos in enumerate(pos1):
                print('    %3d:'%(i2+1),eqpos)
        else:
            pass
        #"""
        #"""
        vtss=equivalent_sites_with_centring(position,brv,indx_coset)
        n1,n2,_,_=vtss.shape
        vnss=np.zeros((n1,n2,6),dtype=np.float64)
        for i2,vts in enumerate(vtss):
            vnss[i2]=numerical_vectors(vts)
        if verbose>0:
            print('  equivalent positions:')
            for i2,vns in enumerate(vnss):
                for i3,vn in enumerate(vns):
                    print('    %d-%d:'%(i2+1,i3+1),vn)
        else:
            pass
        #"""
        #========================================================================
        # generate positions of ODs necessary to generate atomic positions.
        #========================================================================
        """
        # V以外の位置にあるODについて、その位置と等価な位置を原点V周りで対称操作を施すことで得る。
        # 辺中心に原子をおいたP型AKNタイリングを作る場合では、上で発生する位置のみでで十分であることを確認した。
        #
        indx_coset=get_index_of_symmetry_operation_for_equivalent_vectors(position)
        vn=numerical_vector(position)
        #indx_coset=[0]
        pos1=generator_equivalent_numeric_vector_specific_symop(vn,indx_coset)
        if verbose>0:
            print('  equivalent positions:')
            for i2,eqpos in enumerate(pos1):
                print('    %3d:'%(i2+1),eqpos)
        else:
            pass
        #"""
        #========================================================================
        # generate positions of ODs necessary to generate atomic positions.
        #========================================================================
        """
        if np.all(position==V0):
            vn=numerical_vector(position)
            pos1=generator_equivalent_numeric_vector_specific_symop(vn,indx_coset)
        else:
            vts=equivalent_sites_unit_cell(position,indx_coset,brv,1)
            vns=get_internal_component_sets_numerical(vts)
            tmp=np.zeros((len(vts),6),dtype=np.float64)
            i3=0
            for i2,vn in enumerate(vns):
                if np.linalg.norm(vn)<=1.0:
                    tmp[i3]=numerical_vector(vts[i2])
                    i3+=1
                else:
                    pass
            pos1=tmp[:i3]
            if verbose>0:
                print('  equivalent positions:')
                for i2,eqpos in enumerate(pos1):
                    print('    %3d:'%(i2+1),eqpos)
        """
        #========================================================================
        if pod[0]=='polyhedron':
            if pod[2]==1: # asymmetric ODs
                #
                # generating i1-th symmetric occupation domain from its asymmetric unit.
                # here, each TAU-style value is transformed to numerical one.
                #
                #obj=generator_obj_symmetric_obj(pod[1],position)
                #
                #obj=shift_object(obj,position)
                #obj=generator_obj_symmetric_obj_specific_symop(obj,position,indx_site_sym)
                obj=generator_obj_symmetric_obj_specific_symop(obj,V0,indx_site_sym)
                #
                # Spherical approximation of the OD (tmp) to a spherical OD.
                if verbose>1:
                    radius_spherical_obj = spherical_approximation_obj(obj)
                    print('   radius of spherically approximated obj: %8.6f'%(radius_spherical_obj))
                #lst_sphere_radius.append(rad_obj)
                #
                #num_tetrahedron=len(obj)
                #num_coset=len(indx_coset)
                #num_site_symm=len(indx_site_sym)
                #print('obj.shape',obj.shape)
                #print('num_tetrahedron',num_tetrahedron)
                #print('num_coset',num_coset)
                #print('num_site_symm',num_site_symm)
                #indx_coset=[0]
                objs1=generator_obj_symmetric_obj_specific_symop(obj,V0,indx_coset)
                #tshape=tmp.shape
                #print('tmp.shape',tmp.shape)
                #print('n1',tshape[0])
                #print('n2',tshape[1])
                #print('n3',tshape[2])
                #print('num_coset',num_coset)
                #print('num_site_symm',num_site_symm)
                #print('num_tetrahedron',num_tetrahedron)
                
                #objs1=tmp.reshape(num_coset,tshape[1],tshape[2],4,6,3)
                #print('objs1.shape:',objs1.shape)
                n1,n2,n3,_,_,_=objs1.shape
                objs1_=np.zeros((n1,n2,n3,4,3),dtype=np.float64)
                for i2 in range(n1):
                    for i3 in range(n2):
                        for i4 in range(n3):
                            objs1_[i2][i3][i4]=get_internal_component_sets_numerical(objs1[i2][i3][i4])
                #print('objs1_.shape',objs1_.shape)
                lst_shape.append(pod[0])
                lst_objs.append(objs1_)
                lst_pos.append(vnss)
                lst_atm.append(atom)
                lst_be.append(be)
                lst_occ.append(occ)
                lst_rmax.append(rmax)
                lst_eshift.append(np.array(eshift,dtype=np.float64))
                lst_mu.append(np.array(mu,dtype=np.float64))
                
                #-----------------------------------------------------------------------
                # symmetry operation on x1,x2,x3 for each subdevided OD. 
                # used for shift vectors, i.e. xeshift
                #-----------------------------------------------------------------------
                ve1=projection_sets_par_numerical_normalized(x1)
                ve2=projection_sets_par_numerical_normalized(x2)
                ve3=projection_sets_par_numerical_normalized(x3)
                #
                flg='normal'
                # in the independent OD (obj)
                v1_=generator_obj_symmetric_vector_specific_symop_1(ve1,V2,indx_site_sym,flg) # 5f
                v2_=generator_obj_symmetric_vector_specific_symop_1(ve2,V2,indx_site_sym,flg) # 3f
                v3_=generator_obj_symmetric_vector_specific_symop_1(ve3,V2,indx_site_sym,flg) # 2f
                #
                # in the ODs at equivalent positions
                v1_=generator_obj_symmetric_vectors_specific_symop_1(v1_,V2,indx_coset,flg) # 5f
                v2_=generator_obj_symmetric_vectors_specific_symop_1(v2_,V2,indx_coset,flg) # 3f
                v3_=generator_obj_symmetric_vectors_specific_symop_1(v3_,V2,indx_coset,flg) # 2f
                #
                lst_xe1.append(v1_)
                lst_xe2.append(v2_)
                lst_xe3.append(v3_)
                
                #-----------------------------------------------------------------------
                # symmetry operation on x1,x2,x3 for each subdevided OD. 
                # used for axial vectors, i.e. magnetic moment
                #-----------------------------------------------------------------------
                # in the independent OD (obj)
                flg='axial'
                #flg='normal'
                v1_=generator_obj_symmetric_vector_specific_symop_1(ve1,V2,indx_site_sym,flg) # 5f
                v2_=generator_obj_symmetric_vector_specific_symop_1(ve2,V2,indx_site_sym,flg) # 3f
                v3_=generator_obj_symmetric_vector_specific_symop_1(ve3,V2,indx_site_sym,flg) # 2f
                #-----------------------------------------------------------------------
                # in the ODs at equivalent positions
                v1_=generator_obj_symmetric_vectors_specific_symop_1(v1_,V2,indx_coset,flg) # 5f
                v2_=generator_obj_symmetric_vectors_specific_symop_1(v2_,V2,indx_coset,flg) # 3f
                v3_=generator_obj_symmetric_vectors_specific_symop_1(v3_,V2,indx_coset,flg) # 2f
                #-----------------------------------------------------------------------
                lst_mxe1.append(v1_)
                lst_mxe2.append(v2_)
                lst_mxe3.append(v3_)
                
                if test_flag==0:
                    pass
                else:
                    #-----------------------------------------------------------------------
                    # symmetry operation on x1,x2,x3 for each subdevided OD. 
                    # used for axial vectors, i.e. magnetic moment
                    #-----------------------------------------------------------------------
                    # in the independent OD (obj)
                    flg='axial'
                    #flg='normal'
                    x1_=generator_obj_symmetric_vector_specific_symop_1(x1,V2,indx_site_sym,flg) # 5f
                    x2_=generator_obj_symmetric_vector_specific_symop_1(x2,V2,indx_site_sym,flg) # 3f
                    x3_=generator_obj_symmetric_vector_specific_symop_1(x3,V2,indx_site_sym,flg) # 2f
                    # in the ODs at equivalent positions
                    x1_=generator_obj_symmetric_vectors_specific_symop_1(x1_,V2,indx_coset,flg) # 5f
                    x2_=generator_obj_symmetric_vectors_specific_symop_1(x2_,V2,indx_coset,flg) # 3f
                    x3_=generator_obj_symmetric_vectors_specific_symop_1(x3_,V2,indx_coset,flg) # 2f
                    #-----------------------------------------------------------------------
                    lst_mx1.append(x1_)
                    lst_mx2.append(x2_)
                    lst_mx3.append(x3_)
                    #tmp=symmetry_operations_axial_vector(mu6d,V0,indx_site_sym)
                    #tmp=symmetry_operations_axial_vectors(tmp,V0,indx_coset)
                    #lst_6dspin.append(tmp)
                
            else: # symmetric ODs
                # WIP
                pass
        else: # for spherical model
            # WIP
            pass
            
    print('Generating atomic structure...')
    lst=[]
    for h1 in range(-nmax,nmax+1):
        if verbose>0:
            print(h1)
        for h2 in range(-nmax,nmax+1):
            for h3 in range(-nmax,nmax+1):
                for h4 in range(-nmax,nmax+1):
                    for h5 in range(-nmax,nmax+1):
                        for h6 in range(-nmax,nmax+1):
                            h123456=np.array([h1,h2,h3,h4,h5,h6],dtype=np.float64)
                            vn=projection_numerical(h123456)
                            ve=vn[0:3]*aico*CONST1
                            vi=vn[3:6]
                            #-------------------------------------
                            # i1-th independent occupation domain
                            #-------------------------------------
                            for i1,obj1 in enumerate(lst_objs): # i1-th atom.
                                positions=lst_pos[i1]
                                element=lst_atm[i1]
                                mu=lst_mu[i1]
                                xeshift=lst_eshift[i1] # shift of i1-th atom in Epar.
                                xe1=lst_xe1[i1]
                                xe2=lst_xe2[i1]
                                xe3=lst_xe3[i1]
                                mxe1=lst_mxe1[i1]
                                mxe2=lst_mxe2[i1]
                                mxe3=lst_mxe3[i1]
                                if test_flag==1:
                                    #spin6d_=lst_6dspin[i1]
                                    mx1=lst_mx1[i1]
                                    mx2=lst_mx2[i1]
                                    mx3=lst_mx3[i1]
                                #
                                # equivalnts by centring
                                for i2,pos in enumerate(positions):
                                    pose=projection_sets_par_numerical(pos)
                                    posi=projection3_sets_numerical(pos)
                                    for i3,obj2 in enumerate(obj1): # ODs at equivalent positions
                                        we=pose[i3]*aico*CONST1
                                        wi=posi[i3]
                                        point=-vi-wi+oshift
                                        for i4,obj3 in enumerate(obj2): # symmetric OD
                                            xe1_=xe1[i3][i4]
                                            xe2_=xe2[i3][i4]
                                            xe3_=xe3[i3][i4]
                                            mxe1_=mxe1[i3][i4]
                                            mxe2_=mxe2[i3][i4]
                                            mxe3_=mxe3[i3][i4]
                                            if test_flag==1:
                                                #spin6d=spin6d_[i3][i4]
                                                mx1_=mx1[i3][i4]
                                                mx2_=mx2[i3][i4]
                                                mx3_=mx3[i3][i4]
                                            # check whether the 'point' is inside the OD or not.
                                            counter=0
                                            for tetrahedron in obj3: # asymmetric units
                                                # roughly check whether "point" is inside the spherical OD or not.
                                                if inside_outside_tetrahedron_rough(point,tetrahedron): # inside
                                                    # check whether "point" is inside the tetrahedral OD or not.
                                                    if inside_outside_tetrahedron(point,tetrahedron): # inside
                                                        #xeshift_=np.array([xeshift[0]*xe1_,xeshift[1]*xe2_,xeshift[2]*xe3_])
                                                        xeshift_=np.array([xe1_,xe2_,xe3_]).T@xeshift
                                                        xyz=ve+we-xeshift_
                                                        if np.all(mu==0.0): # non-magnetic atom
                                                            lst.append([element,xyz,i1,h123456,0,i4])
                                                        else: # magnetic atom
                                                            # spin moment vector in Epar.
                                                            #mu_=np.array([mu[0]*xe1,mu[1]*xe2,mu[2]*xe3])
                                                            if test_flag==1:
                                                                mu_=np.array([mx1_,mx2_,mx3_]).T@mu
                                                                tmp=projection(mu6d_)[0]
                                                                mu_=numerical_vector(tmp)
                                                            mu_=np.array([mxe1_,mxe2_,mxe3_]).T@mu
                                                            lst.append([element,xyz,i1,h123456,mu_,i4])
                                                        counter+=1
                                                        break
                                                    else:
                                                        pass
                                                else:
                                                    pass
                                            if counter!=0:
                                                break
    return lst
    
def spherical_approximation_obj(obj):
    """
    this function approximates an occupation domain located at 'position' to a sphere.
    
    """
    n1,n2,_,_,_=obj.shape
    num=n1*n2*4
    vertices=obj.reshape(num,6,3)
    #position=get_internal_component_numerical(position)
    lst=[]
    for i1,vt in enumerate(vertices):
        #a=get_internal_component_numerical(vt)-position
        a=get_internal_component_numerical(vt)
        dd=np.sqrt(a[0]**2+a[1]**2+a[2]**2)
        lst.append(dd)
    #return [max(lst),position]
    return max(lst)
    
def spherical_approximation_tetrahedron(tet):
    """
    this function approximates an tetrahedron to a sphere.
    
    """
    cen1=centroid(tetrahedron)
    dd1=ball_radius(tetrahedron,cen1)
    return cen1,dd1
    
def inside_outside_shpere(point,radius,postion):
    p=point-postion
    if np.sqrt(p[0]**2+p[1]**2+p[2]**2)<=radius:
        return True # inside
    else:
        return False # outside
    
if __name__ == '__main__':
    
    v0 = np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
    v1 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2]])
    v2 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2]])
    v3 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[ 0, 0, 2],[-1, 0, 2],[ 0, 0, 2]])
    od0 = np.vstack([v0,v1,v2,v3]).reshape(1,4,6,3)
    
    
    x1= np.array([0, 0, -1, 0, 0, 0],dtype=np.float64) #5f
    x2= np.array([0, 1, -1, 1, 0, 0],dtype=np.float64) #3f
    x3= np.array([0, 1, -1, 0, 0, 0],dtype=np.float64) #2f
    
    
    
    indx_site_sym,indx_coset=site_symmetry_and_coset(v0,'p',0)
    
    #
    # symmetry operation on x1,x2,x3 for each subdevided OD. 
    #
    # in the independent OD (obj)
    v1_=generator_obj_symmetric_vector_specific_symop(x1,V1,indx_site_sym) # 5f
    v2_=generator_obj_symmetric_vector_specific_symop(x2,V1,indx_site_sym) # 3f
    v3_=generator_obj_symmetric_vector_specific_symop(x3,V1,indx_site_sym) # 2f
    #
    # in the ODs at equivalent positions
    v1_=generator_obj_symmetric_vectors_specific_symop(v1_,V1,indx_coset)
    v2_=generator_obj_symmetric_vectors_specific_symop(v2_,V1,indx_coset)
    v3_=generator_obj_symmetric_vectors_specific_symop(v3_,V1,indx_coset)
    
    ve1=projection_sets_par_numerical_normalized(v1_)
    ve2=projection_sets_par_numerical_normalized(v2_)
    ve3=projection_sets_par_numerical_normalized(v3_)
    
    for v in ve1[0]:
        print('%8.6f %8.6f %8.6f (%8.6f)'%(v[0],v[1],v[2],np.linalg.norm(v)))
        
        
    print('===============')
    #
    # symmetry operation on x1,x2,x3 for each subdevided OD. 
    #
    ve1=projection_sets_par_numerical_normalized(x1)
    ve2=projection_sets_par_numerical_normalized(x2)
    ve3=projection_sets_par_numerical_normalized(x3)
    #
    # in the independent OD (obj)
    v1_=generator_obj_symmetric_vector_specific_symop_1(ve1,V2,indx_site_sym,'normal') # 5f
    v2_=generator_obj_symmetric_vector_specific_symop_1(ve2,V2,indx_site_sym,'normal') # 3f
    v3_=generator_obj_symmetric_vector_specific_symop_1(ve3,V2,indx_site_sym,'normal') # 2f
    #
    # in the ODs at equivalent positions
    v1_=generator_obj_symmetric_vectors_specific_symop_1(v1_,V2,indx_coset,'normal') # 5f
    v2_=generator_obj_symmetric_vectors_specific_symop_1(v2_,V2,indx_coset,'normal') # 3f
    v3_=generator_obj_symmetric_vectors_specific_symop_1(v3_,V2,indx_coset,'normal') # 2f
    
    i2=0
    i3=0
    mxe1_=v1_[i2][i3]
    mxe2_=v2_[i2][i3]
    mxe3_=v3_[i2][i3]
    vec=np.array([mxe1_,mxe2_,mxe3_])
    print('vec:',vec)
    mu=np.array([1,0,0])
    mu_=vec@mu
    print(mu_)
    