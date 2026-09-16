
def get_internal_component_sets_numerical(vts: qnv.Qnvec) -> qnv.Qnvec:
    """parallel and perpendicular components of a nd lattice vector in direct space.
    
    Parameters
    ----------
    vsn: array
        set of 6-dimensional vectors, xyzuvw1, xyzuvw2, ...
    """
    #vns=numerical_vectors(vts)
    #return projection3_sets_numerical(vns)
    return prj.projection3_sets_numerical(vts)

#########
#  WIP  #
#########
# projection onto Eperp
def projection_numerical_perp(vn: qnv.Qnvec) -> qnv.Qnvec:
    return prjvec_i(vn)
    """This returns nd vector which corresponds to a projection of vn onto Eperp.
    
    Parameters
    ----------
    v: array
        6-dimensional vector

    Returns
    -------
    nd vectors projected onto Eperp
    """
    #return 

#########
#  WIP  #
#########
# projection onto Eaparallel
def projection_numerical_par(vn: qnv.Qnvec) -> qnv.Qnvec:
    return prj.prjvec_e(vn)
    """This returns nd vector which corresponds to a projection of vn onto Epar.
    
    Parameters
    ----------
    v: array
        6-dimensional vector

    Returns
    -------
    nd vectors projected onto Eperp.
    """
    return prj.prjop_e

def inout_occupation_domain_numerical(obj: qnv.Qnvec,point: qnv.Qnvec):
    """
    """
    n_=3
    qv=qnv.Qnvec(n_,N)  # zero initialized qnvec
    triangles=[qv]*num # qnvec array
    for i1,triangle in enumerate(obj):
        triangles[i1]=get_internal_component_sets_numerical(triangle)
        
    counter=0
    for triangle in triangles:
        if inside_outside_triangle_numerical(triangle,point): # inside
            counter=1
            break
    if counter>0:
        return True
    else:
        return False

def inside_outside_triangle_numerical(triangle: qnv.Qnvec, point: qnv.Qnvec):
    """
    """
    tmp=np.append(triangle[0],triangle[1])
    tmp=np.append(tmp,triangle[2])
    tmp=tmp.reshape(3,3)
    area0=triangle_area_numerical(tmp)
    #
    tmp=np.append(point,triangle[1])
    tmp=np.append(tmp,triangle[2])
    tmp=tmp.reshape(3,3)
    area1=triangle_area_numerical(tmp)
    #
    tmp=np.append(point,triangle[0])
    tmp=np.append(tmp,triangle[2])
    tmp=tmp.reshape(3,3)
    area1+=triangle_area_numerical(tmp)
    #
    tmp=np.append(point,triangle[0])
    tmp=np.append(tmp,triangle[1])
    tmp=tmp.reshape(3,3)
    area1+=triangle_area_numerical(tmp)
    #N=triangle[0].N
    qn0=qnn.zero()
    if abs(area0-area1)<qn0:  #EPS:
        return True # inside
    else:
        return False # outside

# structure under linear phason
def strc(objs,positions,pmatrx,n1max,n5max,eshift,oshift,verbose):
    """
    """
    print()
    print('len(objs):',len(objs))
    for tmp in objs:
        print('tmp.shape:',tmp.shape)
    print('len(positions):',len(positions))
    for tmp in positions:
        print('tmp.shape:',tmp.shape)
    
    
    if np.any(pmatrx)!=0:  # under uniform phason strain
        orgshft=projection_numerical_phason(oshift,pmatrx)
        flg=1
    else:
        orgshft=projection_numerical(oshift)
        flg=0
        
    lst=[]
    for h1 in range(-n1max,n1max+1):
        if verbose>0:
            print(h1)
        for h2 in range(-n1max,n1max+1):
            for h3 in range(-n1max,n1max+1):
                for h4 in range(-n1max,n1max+1):
                    #for h5 in range(-n5max,n5max+1):
                    for h5 in range(0,n5max+1):
                        vn=np.array([h1,h2,h3,h4,h5,0],dtype=np.float64)
                        if flg==0: 
                            v=projection_numerical(vn)
                        else:
                            v=projection_numerical_phason(vn,pmatrx)
                        #-------------------------------------
                        # i-th independent occupation domain
                        #-------------------------------------
                        for i1,obj1 in enumerate(objs):
                            pos=numerical_vectors(positions[i1])
                            xe=numerical_vector(eshift[i1])
                            print('   i1:',i1)
                            print('    pos',pos)
                            print('    len(obj1):',len(obj1))
                            print('    len(pos):',len(pos))
                            
                            #print('eshift[i1]:',eshift[i1])
                            #if flg==0:
                            #    shfte=projection_numerical(xe)
                            #else:
                            #    shfte=projection_numerical_phason(xe,pmatrx)
                            #
                            # equivalent occupation domains
                            for i2,obj2 in enumerate(obj1):
                                pos_eq=pos[i2]
                                #if inout_occupation_domain_numerical(obj2,np.array(v[3:6])-np.array([shft[3],shft[4],shft[5]])): # inside
                                point=v[3:6]-orgshft[3:6]
                                if inout_occupation_domain_numerical(obj2,point): # inside
                                    if flg==0:
                                        w=projection_numerical(pos_eq)
                                        shfte=projection_numerical(xe)
                                    else:
                                        w=projection_numerical_phason(pos_eq,pmatrx)
                                        shfte=projection_numerical_phason(xe,pmatrx)
                                    lst.append([v-w+shfte,i1,h1,h2,h3,h4,h5])
    return lst

