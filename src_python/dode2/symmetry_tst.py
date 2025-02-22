if __name__ == '__main__':
    import numpy as np
    from occupation_domain import (read_xyz)
    from symmetry import (dodesymop_array,check_group,equivalent_positions_in_unit_cell_dev)
    # test
    
    import random
    from numericalc import (numerical_vectors,
                            numerical_vector,
                            numeric_value,)
                            
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
    
    def generate_random_triangle():
        return generate_random_vectors(3)
    """
    print("TEST: symop_vec()")
    symop=dodesymop()
    vt=generate_random_vector()
    counter=0
    cen0=np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])
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
        
    dim=5
    
    """
    vn0=np.array([1, 2, 3, 4, 5, 0]) 
    #vn0=np.array([0, 0, 0, 1, 1, 0])
    x1=[]
    y1=[]
    x2=[]
    y2=[]
    for sop in symop:
        vn=np.dot(sop,vn0)
        pvn=projection_numerical(vn)
        x=pvn[0]
        y=pvn[1]
        z=pvn[2]
        dd=np.sqrt(x**2+y**2+z**2)
        x=x/dd
        y=y/dd
        z=z/dd
        if z>0:
            x1.append(float(x))
            y1.append(float(y))
            print('A %5.4f %5.4f %5.4f'%(x,y,z))
        else:
            x2.append(float(x))
            y2.append(float(y))
            print('B %5.4f %5.4f %5.4f'%(x,y,z))
    ax.scatter([0], [0], s=31415,  marker='o', color='white', alpha=1.0, edgecolors='black')
    ax.scatter(x2, y2, s=200, marker='o', color='white', alpha=1.0, edgecolors='black')
    ax.scatter(x1, y1, s=40,  marker='o', color='black', alpha=1.0, edgecolors='black')
    plt.xlim(-1.1,1.1)
    plt.ylim(-1.1,1.1)
    plt.show()
    """
    
    #pg='12/mmm'
    #pg='12mm'
    #pg='-12m2'
    pg='-12'
    #pg='12'
    symop=dodesymop_array(pg)
    #print(len(symop))
    
    ########################################
    # 生成元から作った集合が群をなすかどうかを確認
    ########################################
    flag='PG'
    #flag='SG'
    dim=5
    if check_group(symop,flag,dim):
        print('checking group: pass')
    else:
        print('checking group: fail')
    ########################################
    ########################################
    
    # Symmetric positions, P\bar{12}m2(12^5mm)
    V_1a =np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]]) # ( 0,  0,  0,  0,  0) # \bar{12}m2(12^5mm)
    V_2a =np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[1,0,7],[0,0,1]]) # ( 0,  0,  0,  0, u1) # 6mm(6^5mm)
    V_4a =np.array([[0,0,1],[2,0,3],[0,0,1],[1,0,3],[1,0,7],[0,0,1]]) # ( 0,2/3,  0,1/3, u2) # 3m(3^2m)
    V_6a =np.array([[0,0,1],[1,0,2],[0,0,1],[0,0,1],[1,0,7],[0,0,1]]) # ( 0,1/2,  0,  0, u3) # mm2(mm1)
    V_6b =np.array([[0,0,1],[1,0,2],[1,0,2],[0,0,1],[1,0,7],[0,0,1]]) # ( 0,1/2,1/2,  0, u4) # mm2(mm1)
    V_12a=np.array([[0,0,1],[1,0,2],[1,0,7],[0,0,1],[1,0,7],[0,0,1]]) # ( 0,1/2,  z,  0, u5) # m11(m11)
    #site=V_1a
    site=V_2a
    #site=V_4a
    #site=V_6a
    #site=V_6b
    #site=V_12a
    
    brv='p'
    #site_symmetry(site,brv,pg,vervose=2)
    #equivalent_positions_in_unit_cell(site,brv,pg,vervose=2)
    
    equivalent_positions_in_unit_cell_dev(site,brv,pg,vervose=2)
    
    """
    
    
    lst=[]
    for i in range(len(symop)):
        lst.append(i)
    #print(lst)
    
    idx_ssym,idx_coset=site_symmetry_and_coset(site,'p',pg,verbose=1)
    print('idx_ssym:',idx_ssym)
    print('idx_coset:',idx_coset)
    
    lst_tmp=[]
    for i1 in lst:
        counter=0
        for i2 in idx_ssym:
            if i1==i2:
                counter=1
                break
            else:
                pass
        if counter==0:
            lst_tmp.append(i1)
    print('idx_else:',lst_tmp)
    
    vn=numerical_vector(site)
    #print('%6.4f %6.4f %6.4f %6.4f %6.4f'%(vn[0],vn[1],vn[2],vn[3],vn[4]))
    
    print('site symmetry:')
    for i1 in idx_ssym:
        vn1=symop[i1]@vn
        print('%d %6.4f %6.4f %6.4f %6.4f %6.4f'%(i1,vn1[0],vn1[1],vn1[2],vn1[3],vn1[4]))
    
    print('idx_coset:')
    for i1 in idx_coset:
        vn1=symop[i1]@vn
        print('%d %6.4f %6.4f %6.4f %6.4f %6.4f'%(i1,vn1[0],vn1[1],vn1[2],vn1[3],vn1[4]))
    
    print('idx_else:')
    for i1 in lst_tmp:
        vn1=symop[i1]@vn
        print('%d %6.4f %6.4f %6.4f %6.4f %6.4f'%(i1,vn1[0],vn1[1],vn1[2],vn1[3],vn1[4]))
"""