################ 
# site symmetry
################
import sys
import numpy as np
import cython

import crsys as crs
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import prjop as prj
import qnmath as qmt
import qnndarray as qna
import qnsym as qns
import lattice as lt

def sitesym_init():
    n=crs.n
    N=crs.N

def site_symmetry(x: qnv.Qnvec) -> np.ndarray: # return irs
    global nr,mpltbl,r
    """symmetry operator insixwa irs in the site symmetry group G.
    
    Args:
        x (qnndarray):  -> (5D or 6D) qnnum coordinates
            xyz coordinate of the site. -> coordinates in external and internal space
            (note that this constrains possible shift in the external space)
            The shape is (6,3). -> 5 (dihedral) or 6 (icosahedral) qnnum coordinates
        
    Returns:
        List of index of symmetry operators of the site symmetry group G (list):
            The symmetry operators leaves xyz identical.
    """
    #a=np.zeros((len(symop),6,3),dtype=np.int64)

    #n=len(x)
    #N=x.N
    if crs.isys==2:
        n_i=3
    else:
        n_i=2
    n=crs.n
    brv=lt.brv
    nr=qns.nr
    #print("n",n,"n_i",n_i) # for test
    r_qn=qns.r_qn  # symmetry operator for Q coordinates
    r_qn_i=qns.r_qn_i  # symmetry operator for Q coordinates
    mpltbl=qns.mpltbl
    r=qns.r_qn
    a_i=qnv.zerovs((nr,n_i))
    qnx_i=prj.prjvec_i(x) # internal space component os nD vector x
    #a=np.zeros((nr,n),dtype=qnn.Qnnum)
    a=qnv.zerovs((nr,n))
    qnx=prj.prjvec(x)
    b=qnv.zerov(n)

    irs=np.zeros(0,dtype=np.int64)
    #tr=lt.get_tr(brv)
    trop=lt.get_tr()  # centering translation vectors including zero vector
    #print("type(trop[0])",type(trop[0]))  # for test
    for i in range(nr):
        #qnm.printqnm("r_qn_i",r_qn_i[i]) # for test
        #qnv.printqnv("qnx_i",qnx_i) # for test
        #a_i[i]=r_qn_i[i]@qnx_i         # Q coordinates for 2D (3D) vector x_i
        #qnm.printqnm("r_qn",r_qn[i]) # for test
        a[i]=r_qn[i]@qnx   # Q coordinates for 2D (3D) vector x_i
        #print("type(a[i])",type(a[i]))  # for test
        # **** a[i] is not Qnvec but Qnmat
        # Qnmat to Qnvec transformation necessary 
        ai=qnm.qnm2qnv(a[i])
        #print("type(ai)",type(ai))  # for test
        #print("type(trop[0])",type(trop[0]))  # for test
        bi=ai+trop[0]
        #print("type(bi)",type(bi))  # for test
        #qnv.printqnv("qnx",qnx)  # for test
        for tr in trop:
            #print("type(tr)",type(tr),"type(ai)",type(ai)) # fpr test
            bi=ai+tr
            #print("type(bi)",type(bi)) # for test
            #b=qnm.qnm2qnv(b0)
            #print("type(b)",type(b),"type(qnx)",type(qnx)) # for test
            if bi==qnx:
                irs=np.append(irs,i)
            else:
                pass
    #print('irs',irs)
    return irs

# new symmetry operator index
def newl(ics,ns0):
    #print("ics",ics,"ns0",ns0) # for test
    for i in range(ns0):
        if i not in ics:
            return i
    print("new i not found")
    print("ics",ics)
    exit()
                
def coset(irs) -> np.array: # return coset representativ indices in symop
    """
    irs: iste symmetry operator index in r_qn
    isk: index for coset representatives
    """
    # number of site symmetry operators    
    ns0=nr        # order of point group
    ns1=len(irs)  # order of site symmetry group
    
    print("irs",irs)
    idxt=np.zeros(ns0,dtype=np.int64)
    isk=np.zeros(0,dtype=np.int64)  # coset representative indices
    ics=np.zeros(0,dtype=np.int64)  # all coset indices
    nc=(np.int64)(ns0/ns1) # number of cosets
    print("nunber of cosets",nc) # number of 
    for k in range(nc):
        #print("k",k) # for test
        if k==0:
            isk=np.append(0,isk)  # identity operator
            for j in range(ns1):
                ics=np.append(ics,irs[j])
            #print("len(ics)",len(ics)) # for test
            #print("ics",ics) # for test
        else:
            i=newl(ics,ns0) #new element not included in ics
            #print("i",i) # for test
            isk=np.append(isk,i)
            lics=len(ics)
            #print("lics",lics) # for test
            for j in irs:
                m=qns.mpltbl[i,j]
                #print("i",i,"ics[j]",ics[j],"m",m) # for test
                ics=np.append(ics,m)
    #print("len(isk)",len(isk))
    print("isk",isk) # for test
    return isk

def equivalent_positions(x: qnv.Qnvec, brv: str, isk: np.ndarray, r0: qnm.Qnmat) -> qnv.Qnvec:
    """
    siteに対して点群の対称性を施したサイトのうち、並進操作のみで結ばれない位置を求める。
        適切な名前を決める必要がある！！！
    """
    qnv.printqnv('x',x) # site : lattice coordinates
    print('equivalent_positions()')
    neq=len(isk)
    #print("neq",neq)  # for test
    n=len(x)
    #xs=np.zeros((neq,n),dtype=qnn.Qnnum)
    xs=qnv.zerovs((neq,n))
    print("type(xs)",type(xs),"type(xs[0])",type(xs[0]))
    for i in range(neq):
        xs[i]=r0[isk[i]]@x
        qnv.printqnv("x",xs[i])
    return xs

def equivalent_positions_in_unit_cell(x:qnv.Qnvec,brv:str,isk:np.ndarray,r0:qnm.Qnmat) -> qnv.Qnvec:
    """
         単位胞内にある等価なサイトを得る。
    """
    xs=equivalent_positions(x,brv,isk,r0)
    reduce_x(xs,brv) # -0.5 <=x[i] <0.5
    return xs
    
def reduce_x(xs:qnv.Qnvec,brv:str):
    print("type(xs)",type(xs),"type(xs[0])",type(xs[0]))  # for test
    nv=len(xs)
    n=xs[0].shape[0]
    qn1=qnn.any([1,0,2])  # 1/2
    qn2=qnn.any([-1,0,2]) # -/2
    qn3=qnn.any([1,0,1])  # 1
    qn4=qnn.any([-1,0,1]) # -1
    for i in range(nv):
        for j in range(n):
            if xs[i][j]<qn2:
                xs[i][j]+=qn3
            elif xs[i][j]>=qn1:
                xs[i][j]+=qn4
    return

def symop_vec(symop:qnm.Qnmat,vt:qnv.Qnvec,centre:qnv.Qnvec):
    """ 
    Apply a symmetric operation on a vector around given centre. in TAU-style
    """
    #vt=sub_vectors(vt,centre)
    vt=qnv.sub(vt,centre)
    vt=qnm.mul(symop,vt)
    return qnv.add(vt,centre)



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
    
#def check_coset(site,comb,symop,idx_site):
#    """
#    """
#    #symop=symop_array()
#    #list1=site_symmetry(site)
#        
#    list4=[]
#    for i2 in comb:
#        for i1 in idx_site: # i1-th symmetry operation of the site symmetry (point group, H)
#            op1=symop[i2]@symop[i1]
#            for i3,op in enumerate(symop):
#                if np.all(op==op1):
#                    num=i3
#                    break
#                else:
#                    pass
#            list4.append(num)
#    c=remove_overlaps(list4)
#    if len(c)==len(list4):
#        return True
#    else:
#        return False
         
#def equivalent_positions(site,brv,pg,vervose=0):
    #pg : rotation matrix in lattice coordinate system 
#    symop=qns.r_qn
#    eqpos=np.zeros((len(symop),6,3),dtype=np.int64)
#    for i,op in enumerate(symop):
#        eqpos[i]=symop_vec(op,site,centre=V0)
#    eqpos1=remove_doubling(eqpos)
#    print('  len(eqpos1)=',len(eqpos1))
#    for _eqpos1 in eqpos1:
#        print('  _eqpos1:',numerical_vector(_eqpos1))
#    
#    # 求めた等価なサイトのうち、並進操作を施して同一なのであれば、どちらか片方を選ぶようにする。
#    if len(eqpos1)==1:
#        lst_saved=[site]
#    else:
#        lst_saved=[site]
#        translation=translation_new(brv,flag=0)
#        for pos in eqpos1:
#            counter=0
#            for tr in translation:
#                pos1=add_vectors(pos,tr)
#                if np.all(site==pos1):
#                    counter+=1
#                    break
#                else:
#                    pass
#            if counter==0:
#                lst_saved.append(pos)
#    print('  len(lst_saved)=',len(lst_saved))
#    for _lst_saved in lst_saved:
#        print('  _lst_saved:',numerical_vector(_lst_saved))
#    
#    
#    # 求めたサイトのうち単位胞内にあるサイトを選ぶ
#    out=np.zeros((len(lst_saved),6,3),dtype=np.int64)
#    num=0
#    for vt in lst_saved:
#        vn=numerical_vector(vt)
#        #if np.all(vn>=0.0):# and np.all(vn<1.0):
#        if np.all(vn>=0.0) and np.all(vn<=1.0):
#            out[num]=vt
#            print('  vt:',numerical_vector(vt))
#            num+=1
#        else:
#            pass
#    return out[:num]
#        
#        
#    symop=dodesymop_array(pg)
#    
#    if verbose>0:
#        vn=numerical_vector(site)
#        print(' site: %4.3f %4.3f %4.3f %4.3f %4.3f %4.3f'%(vn[0],vn[1],vn[2],vn[3],vn[4],vn[5]))
#    if np.all(site==V0):
#        a=[]
#        for i in range(len(symop)):
#            a.append(i)
#        idx_site=a
#        idx_coset=[0]
#    else:
#        idx_site=site_symmetry(site,symop,brv)
#        idx_coset=coset(site,symop,brv,pg,idx_site)
#    if verbose>0:
#        print('  order of site symmetry:',len(idx_site))
#        print('  number of equivalent positions:',len(idx_coset))
#   return idx_site,idx_coset


#def equivalent_positions_in_unit_cell(site,brv,pg,vervose=0):
#    if vervose>0:
#        print(' equivalent_positions_in_unit_cell()')
#        print('  site:',numerical_vector(site))
#    symop=dodesymop_array(pg)
#    eqpos=np.zeros((len(symop),6,3),dtype=np.int64)
#    for i,op in enumerate(symop):
#        eqpos[i]=symop_vec(op,site,centre=V0)
#    eqpos1=remove_doubling(eqpos)
#    if vervose>1:
#        print('  len(eqpos1)=',len(eqpos1))
#        for _eqpos1 in eqpos1:
#            print('  _eqpos1:',numerical_vector(_eqpos1))
#    
#    # 求めた等価なサイトに、並進操作を施し、単位胞ないにあるもののみ得る。
#    if np.all(site==V0):
#        #return site.reshape(1,1,6,3),[0]
#        return site.reshape(1,6,3),[0]
#    else:
#        translation=translation_new(brv,flag=0)
#        out=np.zeros((len(translation)*len(eqpos1),6,3),dtype=np.int64)
#        num=0
#        for pos in eqpos1:
#            for tr in translation:
#                _pos=add_vectors(pos,tr)
#                _pos=numerical_vector(_pos)
#                if np.all(_pos>=0.0) and np.all(_pos<1.0):
#                    if vervose>1:
#                        print('  _pos:',numerical_vector(_pos))
#                    out[num]=_pos
#                    num+=1
#        out1=remove_doubling(out[:num])
#        if vervose>0:
#            for out1_ in out1:
#                if vervose>1:
#                    print('  equivalent site:',numerical_vector(out1_))
#            
#        # 得られたサイトが元のサイトとどのような対称操作で結ばれているのかを調べる。
#        out_idx_symop=[]
#        flag=0
#        for out_ in out1:
#            counter=0
#            for i,op in enumerate(symop):
#                vt=symop_vec(op,site,centre=V0)
#                for tr in translation:
#                    _vt=add_vectors(vt,tr)
#                    if np.all(_vt==out_):
#                        out_idx_symop.append(i)
#                        counter=1
#                        break
#                if counter!=0:
#                    break
#            if counter==0:
#                flag+=1
#                break
#            else:
#                pass
#        if flag==0:
#            if vervose>1:
#                print('out1.shape:',out1.shape)
#            return out1,out_idx_symop
#        else:
#            return

#def site_symmetry(site,brv,pg,vervose=0):
#    """symmetry operators in the site symmetry group G.
#        
#    Args:
#        site (numpy.ndarray):
#            xyz coordinate of the site.
#            The shape is (6,3).
#        
#    Returns:
#        List of index of symmetry operators of the site symmetry group G (list):
#            The symmetry operators leaves xyz identical.
#    """
#    
#    if vervose>0:
#        print(' site_symmetry()')
#        print('  site:',numerical_vector(site))
#    symop=dodesymop_array(pg)
#    
#    # サイト周りでvtgに対して対称操作を施す。
#    vtg=np.array([[1,0,3],[0,1,4],[1,0,5],[0,1,6],[1,0,7],[0,0,1]],dtype=np.int64)
#    #vtg=add_vectors(vtg,site)
#    a=np.zeros((len(symop),6,3),dtype=np.int64)
#    for i1,op in enumerate(symop):
#        a[i1]=symop_vec(op,vtg,site)
#        
#    if brv=='p':
#        flag=1
#    else:
#        pass
#    traop=translation_new(brv,flag)
#    lst=[]
#    for i1,a1 in enumerate(a):
#        # vtgに対して並進を含む全ての対称操作を施す。
#        if vervose>1:
#            print('%3d     a1:'%(i1),numerical_vector(a1))
#        counter1=0
#        for op in symop:
#            tmp1=symop_vec(op,vtg,V0)
#            if np.all(a1==tmp1):
#                counter1=1
#                if vervose>1:
#                    print('      tmp1:',numerical_vector(tmp1))
#                break
#            else:
#                flag1=0
#                for tr in traop:
#                    b=add_vectors(tmp1,tr)
#                    if np.all(a1==b):
#                        flag1=1
#                        #print('         b:',numerical_vector(b))
#                        break
#                    else:
#                        pass
#                if flag1==1:
#                    counter1=1
#                    break
#                else:
#                    pass
#        if counter1==1:
#            lst.append(i1)
#        else:
#            pass
#    if vervose>0:
#        print('  lst:',lst)
#    return lst

#def equivalent_positions_in_unit_cell_dev(site,brv,pg,vervose=0):
#    """
#          単位胞内にある等価なサイトを得る。
#    """
#    if vervose>0:
#        print(' equivalent_positions_in_unit_cell()')
#        print('  site:',numerical_vector(site))
#    symop=dodesymop_array(pg)
#    eqpos=np.zeros((len(symop),6,3),dtype=np.int64)
#    for i,op in enumerate(symop):
#        eqpos[i]=symop_vec(op,site,centre=V0)
#    if vervose>1:
#        for i,_eqpos in enumerate(eqpos):
#            print('   (%d)'%(i),numerical_vector(_eqpos))
#            
#    # symopを施したサイトに並進操作を施し、単位胞内にあるものを得る。
#    lst_symop_unit_cell=[]
#    lst_symop_site_symm=[]
#    tr=translation_new(brv,flag=1)
#    for i1,_pos in enumerate(eqpos):
#        for i2,_tr in enumerate(tr):
#            pos=add_vectors(_pos,_tr)
#            posn=numerical_vector(pos)
#            if np.all(posn>=0.0) and np.all(posn<1.0):
#                lst_symop_unit_cell.append([i1,i2])
#                if vervose>1:
#                    print('   (%d,%d):'%(i1,i2),numerical_vector(pos))
#                else:
#                    pass
#            else:
#                pass
#        if np.all(_pos==site):
#            lst_symop_site_symm.append(i1)
#        else:
#            pass
#    print('lst_symop_unit_cell')
#    for i in range(len(lst_symop_unit_cell)):
#        print(lst_symop_unit_cell[i])
#        
#    print('\nSite symmetry:')
#    print('lst_symop_site_symm')
#    print(lst_symop_site_symm)
#    #for i in range(len(lst_symop_site_symm)):
#    #    print(lst_symop_site_symm[i])
#    
#    # 全対称操作をサイトシンメトリの操作を取り除く
#    a=set(range(len(symop)))
#    b=set(lst_symop_site_symm)-{0}
#    idx_else=list(a-b)
#    print(idx_else)
#
#    print('lst_symop_unit_cell:',lst_symop_unit_cell)
#    _lst_symop_unit_cell=[]
#    for a in lst_symop_unit_cell:
#        _lst_symop_unit_cell.append(a[0])
#    print('_lst_symop_unit_cell:',_lst_symop_unit_cell)
#    # idx_elseの対称操作のうち、各等価サイトを作る対称操作を調べる。
#    tmp1=[]
#    for idx2 in _lst_symop_unit_cell:
#        
#        _site=symop_vec(symop[idx2],site,V0)
#        tmp=[]
#        for idx1 in idx_else:
#            pos1=symop_vec(symop[idx1],_site,V0)
#            if np.all(pos==pos1):
#                tmp.append(idx1)
#                #break
#            else:
#                pass
#        tmp1.append(tmp)
#        print('  idx_coset',tmp)
#        
#     # いくつかある組み合わせのうち最初のものを選ぶ。
#    idx_coset=[]
#    for i in range(len(tmp1)):
#        idx_coset.append((tmp1[i][0]))
#    print('idx_coset')
#    print(idx_coset)
#    #if check_coset(site,idx_coset,symop,idx_site):
#    #    return idx_coset
#    #else:
#    #    return 
#    
            