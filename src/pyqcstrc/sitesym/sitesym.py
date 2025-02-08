################ 
# site symmetry
################
import sys
import numpy as np
import pyqcstrc.qnnum.qnnum as qnn
import pyqcstrc.qnvec.qnvec as qnv
import pyqcstrc.qnmat.qnmat as qnm
import pyqcstrc.prjop.prjop as prj
import pyqcstrc.qnmath.qnmath as qmt
import pyqcstrc.qnndarray.qnndarray as qna
import pyqcstrc.qnsym.qnsym as qns
import pyqcstrc.lattice.lattice as lt


def site_symmetry(x,qnr,brv) -> np.ndarray: # return irs
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

    n=len(x)
    N=x[0].N
    a=np.zeros((len(qnr),n),dtype=qnn.Qnnum)

    irs=[]
    tr=lt.get_tr(brv,n,N)
    traop=lt.get_tr(brv,n,N)  # centering translation vectors including zero vector
    for i in range(len(qnr)):
        op=qnr[i]
        a[i]=op@x    
        for tr in traop:
            b=a[i]+tr
            if np.all(b==x):
                irs.append(i)
            else:
                pass
    print('lst:',irs)
    return irs
        
def coset(irs) -> np.array: # return coset representativ indices in symop
    """
    irs: iste symmetry operator index in qnr
    isk: index for coset representatives
    """
    ln0=len(qnr) # number of symmetry operators
    ln1=len(irs) # number of site symmetry operators
    shape=qnr.shape # (ng,n,n)
    ng0=shape[0]
    n=shape[1]
    ng1=len(irs)
    op0=np.zeros((ln0,n,n))
    op0[0]=symop[0] # nxn unit matrix
    idxt=np.zeros(ln0,dtype=np.int64)
    isk=[]
    m=0
    for i in range(ng0):
        if idxt[i]==0:
            isk.append(i) # coset representative
            m+=1
            for j in range(ng1):
                if mpltbl[i][j]==k:
                    ixdt[k]=1
    print("idx",isk[0:m]) # for test
    return isk

def equivalent_positions(x,brv,isk) -> qnv.vector:
    """
    siteに対して点群の対称性を施したサイトのうち、並進操作のみで結ばれない位置を求める。
        適切な名前を決める必要がある！！！
    """
    print('equivalent_positions()')
    print('  site:',numerical_vector(site)) # site : lattice coordinates
    ng=len(isk)
    n=len(x)
    xs=qnv.zeros((nb,n))
    r=qns.r
    for i in range(ng):
        xs[i]=r[isk[i]]@x
        
    return xs

def equivalent_positions_in_unit_cell(x,brv,isk):
    """
         単位胞内にある等価なサイトを得る。
    """
    xs=equivalent_positions(x,brv,isk)
    xs=reduce(xs,brv) # -0.5 <=x[i] <0.5
    return xs
    
def reduce_x(xs,brv):
    nv=len(xs)
    N=xs[0][0].N
    shape=xs.shape
    n=shape[1]
    qn1=qnn.Qnnum([1,0,2],N)  # 1/2
    qn2=qnn.Qnnum([-1,0,2],N) # -/2
    qn3=qnn.Qnnum([1,0,1],N)  # 1
    qn4=qnn.Qnnum([-1,0,1],N) # -1
    for i in range(nv):
        for j in range(n):
            if xs[i][j]<qn2:
                xs[i][j]+=qn3
            elif xs[i][j]>=qn1:
                xs[i][j]+=qn4
    return xs


# for test
if __name__ == '__main__':
    isys=3  # decagonal
    brv='p'
    N=5
    n=5
    prj.prjop_init(isys)
    qns.qnsym_init(isys)
    x=qnv.zerov(n,N)
    qnr=qns.qnr
    irs=site_symmetry(x,qnr0,brv)
    print("irs",irs)
    
    x1=qnv.zerov(n,N)           #(0,0,0,0,0)
    x1[0]=qnn.Qnnum([1,0,2],N)  #(1/2,0,0,0,0)
    irs=site_symmetry(x1,qnr,brv) 
    isk=coset(irs)
    xeq=equivalent_positions_in_unit_cell(x1,isk)
    qnv.printqnv("xeq",xeq)
    
    
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
#    symop=qns.qnr
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
            