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
                                        equivalent_sites_in_unit_cell,
                                        generator_equivalent_vec,
                                        get_index_of_symmetry_operation_for_equivalent_vectors,
                                        )
    from pyqcstrc.ico2.math1 import (mul_vector,
                                     mul_vectors,
                                     )
    from pyqcstrc.ico2.utils import (shift_object,
                                    )
except ImportError:
    print('import error in structure_factor\n')

TAU=(1+np.sqrt(5))/2.0
EPS=1e-6
V0=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.int64)
V1=np.array([0, 0, 0, 0, 0, 0],dtype=np.float64)
V2=np.array([0, 0, 0],dtype=np.float64)
CONST1 = 1/np.sqrt(2.0+TAU)

def strc(aico,brv,model,nmax,oshift,verbose):
    """
    this function generates atomic and magnetic structures in 3-d physical space.
    
    input:
    
    eshift, list
        [u1,u2,u3], shift along 5-, 3-, 2-fold axis in Epar (xe1,xe2,xe3).
    mu, list
        [mu1,mu2,mu3], mu along 5-, 3-, 2-fold axis in Epar (xe1,xe2,xe3).
    
    """
    
    #x1=np.array([0, 0, -1, 0, 0, 0],dtype=np.float64) #5f
    #x2=np.array([0, 1, -1, 1, 0, 0],dtype=np.float64) #3f
    #x3=np.array([0, 1, -1, 0, 0, 0],dtype=np.float64) #2f
    x1=np.array([1, 0, 0, 0, 0, 0],dtype=np.float64) #5f
    x2=np.array([1, 0,-1, 0,-1, 0],dtype=np.float64) #3f
    x3=np.array([1, 0, 0, 0,-1, 0],dtype=np.float64) #2f
    
    oshift=projection3_numerical(oshift)
    
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
    for i1,nod in enumerate(model):
        atom, pod, position, eshift, be, occ, rmax, mu = model[nod]
        obj=pod[1]
        if brv=='s':
            #obj=mul_vectors(obj,np.array([1,0,2]))
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
        vtss=equivalent_sites_in_unit_cell(position,brv,indx_coset)
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
                radius_spherical_obj = spherical_approximation_obj(obj)
                print('radius_spherical_obj:',radius_spherical_obj)
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
                
                
                
                
                
                
                
                #lst_pos.append(pos1)
                #
                #
                #
                tmp1=[]
                for vns in vnss:
                    tmp=[]
                    for vn in vns:
                        vni=projection3_numerical(vn)
                        dd=np.linalg.norm(vni)
                        print('dd:',dd)
                        if dd <= radius_spherical_obj:
                            tmp.append(vn)
                        else:
                            pass
                    if len(tmp)!=0:
                        tmp1.append(tmp)
                vnss=tmp1
                #
                #
                #
                for i2,vts in enumerate(vtss):
                    vns=numerical_vectors(vts)
                    for i3,vn in enumerate(vns):
                        print('%d-%d:'%(i2+1,i3+1),vn)
                else:
                    pass
                #
                #
                #
                #
                lst_pos.append(vnss)
                
                
                
                
                
                
                
                lst_atm.append(atom)
                lst_be.append(be)
                lst_occ.append(occ)
                lst_rmax.append(rmax)
                lst_eshift.append(eshift)
                lst_mu.append(mu)
                
                
                #
                # symmetry operation on x1,x2,x3 for each subdevided OD. 
                # used for shift vector, i.e. xeshift
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
                #
                lst_xe1.append(v1_)
                lst_xe2.append(v2_)
                lst_xe3.append(v3_)
                
                #
                # symmetry operation on x1,x2,x3 for each subdevided OD. 
                # used for axial vector, i.e. magnetic moment
                #
                #ve1=projection_sets_par_numerical_normalized(x1)
                #ve2=projection_sets_par_numerical_normalized(x2)
                #ve3=projection_sets_par_numerical_normalized(x3)
                #
                # in the independent OD (obj)
                v1_=generator_obj_symmetric_vector_specific_symop_1(ve1,V2,indx_site_sym,'axial') # 5f
                v2_=generator_obj_symmetric_vector_specific_symop_1(ve2,V2,indx_site_sym,'axial') # 3f
                v3_=generator_obj_symmetric_vector_specific_symop_1(ve3,V2,indx_site_sym,'axial') # 2f
                #
                # in the ODs at equivalent positions
                v1_=generator_obj_symmetric_vectors_specific_symop_1(v1_,V2,indx_coset,'axial') # 5f
                v2_=generator_obj_symmetric_vectors_specific_symop_1(v2_,V2,indx_coset,'axial') # 3f
                v3_=generator_obj_symmetric_vectors_specific_symop_1(v3_,V2,indx_coset,'axial') # 2f
                #
                lst_mxe1.append(v1_)
                lst_mxe2.append(v2_)
                lst_mxe3.append(v3_)
                
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
                            vn=projection_numerical(np.array([h1,h2,h3,h4,h5,h6],dtype=np.float64))
                            ve=vn[0:3]*aico*CONST1
                            vi=vn[3:6]
                            #-------------------------------------
                            # i1-th independent occupation domain
                            #-------------------------------------
                            for i1,obj1 in enumerate(lst_objs): # i1-th atom.
                                positions=lst_pos[i1]
                                element=lst_atm[i1]
                                #
                                mu=lst_mu[i1]
                                xeshift=lst_eshift[i1] # shift of i1-th atom in Epar.
                                #
                                xe1=lst_xe1[i1]
                                xe2=lst_xe2[i1]
                                xe3=lst_xe3[i1]
                                #
                                mxe1=lst_mxe1[i1]
                                mxe2=lst_mxe2[i1]
                                mxe3=lst_mxe3[i1]
                                #
                                #
                                # equivalnts by centring
                                for pos in positions:
                                    pose=projection_sets_par_numerical(pos)
                                    posi=projection3_sets_numerical(pos)
                                    for i2,obj2 in enumerate(obj1): # ODs at equivalent positions
                                        we=pose[i2]*aico*CONST1
                                        wi=posi[i2]
                                        point=vi-wi-oshift
                                        ####point=vi-oshift
                                        for i3,obj3 in enumerate(obj2): # symmetric OD
                                            xe1_=xe1[i2][i3]
                                            xe2_=xe2[i2][i3]
                                            xe3_=xe3[i2][i3]
                                            mxe1_=mxe1[i2][i3]
                                            mxe2_=mxe2[i2][i3]
                                            mxe3_=mxe3[i2][i3]
                                            counter=0
                                            for tetrahedron in obj3: # asymmetric units
                                                # roughly check whether the v is inside the spherical OD or not.
                                                if inside_outside_tetrahedron_rough(point,tetrahedron): # inside
                                                    # check whether the v is inside the spherical OD or not.
                                                    if inside_outside_tetrahedron(point,tetrahedron): # inside
                                                        #xeshift_=np.array([xeshift[0]*xe1_,xeshift[1]*xe2_,xeshift[2]*xe3_])
                                                        xeshift_=np.array([xe1_,xe2_,xe3_])@xeshift
                                                        xyz=ve-we+xeshift_
                                                        if mu==0: # non-magnetic atom
                                                            lst.append([element,xyz,i1,h1,h2,h3,h4,h5,h6,0])
                                                        else: # magnetic atom
                                                            # spin moment vector in Epar.
                                                            #mu_=np.array([mu[0]*xe1,mu[1]*xe2,mu[2]*xe3])
                                                            mu_=np.array([mxe1_,mxe2_,mxe3_])@mu
                                                            lst.append([element,xyz,i1,h1,h2,h3,h4,h5,h6,mu_])
                                                        counter+=1
                                                        break
                                                    else:
                                                        pass
                                                else:
                                                    pass
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
    
    print(ve1)
    