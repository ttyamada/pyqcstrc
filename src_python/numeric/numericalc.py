#!/usr/bin/env python
#
# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import numpy as np
from numpy.typing import NDArray
import random
import qnnum.qnnum as qnn
import qnvec.qnvec as qnv
import qnmat.qnmat as qnm
import prjop.prjop as prj

#TAU=np.sqrt(3)/2.0
#SQRT3=np.sqrt(3)
#N=3

#EPS=1e-6 # tolerance

def coplanar_check_numeric_tau(pts: qnv.Qnvec, num_iteration: int=5) -> bool:
    """check the points (pts) are in coplanar or not
    
    Parameters
    ----------
    pns: array
        6d coordinates of the points, xyz, in TAU-style
    num_iteration: int
        number of iterations.
    
    Returns
    -------
    bool
    
    """
    p=get_internal_component_sets_numerical(pts)
    return coplanar_check_numeric(p,num_iteration)

#def coplanar_check_numeric(pns: NDArray[np.float64],num_iteration: int=5) -> bool:
#    """check the points (pns) are in coplanar or not
#    メモ：xyz1とxyz2の選び方次第で、outer_product(v1,v2)が小さくなりcoplanarと間違って判定する場合がある。
#    これを避けるために適切なxyz1とxyz2の選び方が必要。以下では、ランダムにxyz1とxyz2の選ぶ。
#    
#    Parameters
#    ----------
#    pns: array
#        coordinate of the points in Eperp, xyz.
#    num_iteration: int
#        number of iterations.
#    
#    Returns
#    -------
#    bool
#    """
#    num=len(pns)
#    if num>3:
#        flag=0
#        lst0=[i for i in range(num)]
#        for _ in range(num_iteration):
#            lst3=random.sample(lst0, 3)
#            #
#            xyz1=pns[lst3[1]]-pns[lst3[0]]
#            xyz2=pns[lst3[2]]-pns[lst3[0]]
#            vec=np.cross(xyz1,xyz2)
#            flg=0
#            if np.all(abs(vec)<EPS):
#                pass
#            else:
#                flag=1
#                break
#        if flag==1:
#            counter=0
#            lst=list(filter(lambda x: x not in lst3, lst0))
#            for i in lst:
#            #for i in list(filter(lambda x: x not in lst3, lst0)):
#                xyzi=pns[i]-pns[lst3[0]]
#                if abs(np.dot(vec,xyzi))<1e-10:
#                    pass
#                else:
#                    counter=1
#                    break
#            if counter==0:
#                return True
#            else:
#                return False
#        else:
#            'error in coplanar_check_numeric. increase num_iteration.'
#            return 
#    else:
#        return True

def point_on_segment(point: qnv.Qnvec, line_segment: qnv.Qnvec) -> bool:
    #judge whether a point is on a line segment, A-B, or not.
    #xyx0=projection3_numerical(point)
    #tmp=projection3_sets_numerical(line_segmant)
    xyx0=point
    xyx1=line_segment[0]  # start point in qnvector
    xyx2=line_segment[1]  # end point in qnvector
    
    vecPA=xyx0-xyx1
    vecBA=xyx2-xyx1
    lPA=qnv.dot(vecPA,vecPA)  # squared norm for qnnumber np.linalg.norm(vecPA)
    lBA=qnv.dot(vecBA,vecBA)  # squared norm for qnnumber np.linalg.norm(vecBA)
    qn1=qnn.Qnnum(1,0,1)
    qn0=qnn.Qnnum(0,0,1)
    #if lBA>0.0 and abs(np.dot(vecPA,vecBA)-lPA*lBA)<EPS:
    if lBA>qnn.zero and abs(qnv.dot(vecPA,vecBA)-lPA*lBA)==qn0:
        s=lPA/lBA
        if s>=qn0 and s<=qn1:#if s>=0.0 and s<=1.0:
            return True
        elif s>qn1: #elif s>1.0:
            return False #       A==B P
        else:
            return False #    P A==B
    else:
        return False

def on_out_surface(point: qnv.Qnvec, triangle: qnv.Qnvec) -> bool:
    """
    check whether the point is inside the triangle.
    
    Parameters
    ----------
    point: array
        6d coordinates of the point in TAU-style.
    triangle: array
        6d coordinates of three vertices of triangle in TAU-style
    
    Returns
    -------
    float => qnnum
    """
    
    def func(p_xyz:qnn.Qnvec,tr_xyz,indx:qnn.Qnvec) -> qnn.Qnvec:
        out=np.zeros((3,3),dtype=np.float64)
        for i in range(3):
            if i==indx:
                out[i]=p_xyz
            else:
                out[i]=tr_xyz[i]
        return out
        
    p=projection3(point) #get_internal_component_numerical(point)
    #triangle0=get_internal_component_sets_numerical(triangle)
    
    area0=triangle_area_numerical(triangle) # qnnum
    
    triangle1=func(p,triangle,0)
    triangle2=func(p,triangle,1)
    triangle3=func(p,triangle,2)
    
    area1=triangle_area_numerical(triangle1)+\
        triangle_area_numerical(triangle2)+\
        triangle_area_numerical(triangle3)
    N=point.N
    qn0=qnn.Qnnum([0,0,1],N) # 0 in qnnum
    if abs(area0-area1)==qn0:  #< EPS:
        return True
    else:
        return False

# qnnum version => qnnum.qnn2flt
# equivalent to qnn2flt
def numeric_value(t: qnn.Qnnum) -> float:
    return qnn.qnn2flt(t)
    """Numeric value of a TAU-style value, a.

    Parameters
    ----------
    t: array
        value in TAU-style
    
    Returns
    -------
    float
    """
    #return (t[0]+t[1]*TAU)/t[2]
    #return (t[0]+t[1]*SQRT3)/t[2] # float???

# qnnum version => qnnum.qnn2flt
# equivalent to qnv2flt
def numerical_vector(vt: qnv.Qnvec) -> NDArray[np.float64]:
    return qnv.qnv2flt(vt)
#    """Numeric value of a TAU-style vector, v.
#
#    Parameters
#    ----------
#    vt: array
#        vector in TAU-style
#    
#    Returns
#    -------
#    array
#    """
#    n=len(vt)
#    w=np.zeros(n,dtype=np.float64)
#    for i in range(n):
#        w[i]=numeric_value(vt[i])
#    return w

# qnnum vectors to float (not necessary)
# equivalent to qnv.qnv2flt
def numerical_vectors(vts: NDArray[np.int64]) -> NDArray[np.float64]:
    """Numeric value of a TAU-style vector, v.

    Parameters
    ----------
    vts: array
        vector in TAU-style
    
    Returns
    -------
    array
    """
    if vts.ndim==3: # triangle vertex
        n1,n2,_=vts.shape
        w=np.zeros((n1,n2),dtype=np.float64) # float???
        for i1,vt in enumerate(vts):
            w[i1]=numerical_vector(vt)
        return w
    elif vts.ndim==4: # tetrahedron vertex
        n1,n2,n3,_=vts.shape
        w=np.zeros((n1,n2,n3),dtype=np.float64) # float?
        for i1,triangle in enumerate(vts):
            for i2,vt in enumerate(triangle): # surface triangles
                w[i1][i2]=numerical_vector(vt)
        return w
    else:
        print('error')
        return 

# vector length in float (not necessary)
def length_numerical(vt: NDArray[np.int64]) -> float:
    """numerical value of norm of vector, v, in Tau-style
    
    Parameters
    ----------
    vt: array
        vector in TAU-style
    
    Returns
    -------
    
    """
    vn=numerical_vector(vt)
    return np.linalg.norm(vn)
    #return np.sqrt(np.sum(np.abs(v**2)))



def check_intersection_segment_surface_numerical_6d_tau(line_segment: qnv.Qnvec, triangle: qnv.Qnvec) -> bool:
    """check intersection between a line segment and a triangle.
    
    Parameters
    ----------
    line_segment: array
        6-dimensional coordinates of line segment,xyzuvw1, xyzuvw2, in TAU-style
    triangle: array
        containing 6-dimensional coordinates of tree vertecies of a triangle (a) in TAU-style
    
    Returns
    -------
    
    """
    #ln=get_internal_component_sets_numerical(line_segment)
    #tr=get_internal_component_sets_numerical(triangle)
    return check_intersection_segment_surface_numerical(ln,tr)
    

def check_intersection_segment_surface_numerical(line_segment: qnv.Qnvec, triangle: qnv.Qnvec) -> bool:
    
    """check intersection between a line segment and a triangle.
    
    Parameters
    ----------
    line_segment: array
        two 3-dimensional coordinates of line segment, xyz1, xyz2.
    triangle: array
        containing 3-dimensional coordinates of tree vertecies of a triangle (a), xyz1, xyz2, xyz3.
    
    Returns
    -------
    """
    """
    counter=0
    for i in range(3):
        if check_intersection_two_segment_numerical(line_segment,triangle[i])!=3:
            counter+=1
            break
        else:
            pass
    if counter>0:
        return True
    else:
        return False
    
    
    vecAB=ln[1]-ln[0] # AB # R
    vecCD=tr[1]-tr[0] # CD # E1
    vecCE=tr[2]-tr[0] # CE # E2
    vecCA=ln[0]-tr[0] # CA # T
    
    vecP=np.cross(vecAB,vecCE) # P
    vecQ=np.cross(vecCA,vecCD) # Q
    
    bunbo=np.dot(vecP,vecCD)
    #print('bunbo=',bunbo)
    if abs(bunbo)<EPS: # the line_segment is parrallel to the triangle.
        return False
    else:
        u=np.dot(vecP,vecCA)/bunbo
        #print('u=',u)
        if u>=0.0-EPS and u<=1.0+EPS:
            v=np.dot(vecQ,vecAB)/bunbo
            #print('v=',v)
            if v>=0.0-EPS and u+v<=1.0+EPS:
                t=np.dot(vecQ,vecCE)/bunbo
                #print('t=',t)
                if t>=0.0-EPS and t<=1.0+EPS:
                    #print('  intersect\n')
                    return True # intersect
                else:
                    #print('\n')
                    return False
            else:
                #print('\n')
                return False
        else:
            #print('\n')
            return False
    """
    
    #  edge: 0-1,0-2,1-2
    comb=[[0,1],[0,2],[1,2]] # line index
    counter=0
    for j in comb:
        if check_intersection_two_segment_numerical(line_segment,triangle[j]):
            counter+=1
            break
        else:
            pass
    if counter>0:
        return True # intersecting
    else:
        return False

def check_intersection_two_segment_numerical_6d_tau(segment_1: qnv.Qnvec, segment_2: qnv.Qnvec) -> bool:
    """check intersection between two line segments
    
    Parameters
    ----------
    line_segment_1,line_segment_2 : array
        two 6-dimensional coordinates of line segment,xyzuvw1, xyzuvw2, in TAU-style
    triangle: array
        containing 3-dimensional coordinates of tree vertecies of a triangle (a), xyz1, xyz2, xyz3.
    
    Returns
    -------
    
    """
    # TAU-style to Float
    #segment_1=numerical_vectors(segment_1)
    #segment_2=numerical_vectors(segment_2)
    
    #ln1=get_internal_component_sets_numerical(segment_1)
    #ln2=get_internal_component_sets_numerical(segment_2)
    return check_intersection_two_segment_numerical(segment_1,segment_2)

def check_intersection_two_segment_numerical(ln1:qnv.Qnvec, ln2:qnv.Qnvec) -> bool:
    """check intersection between two line segments.
    
    Parameters
    ----------
    line_segment_1,line_segment_2: array
        two 3-dimensional coordinates of line segment, xyz1, xyz2.
        line_segment_1: A--B
        line_segment_2: C--D
    
    Returns
        int, out = 0 (Intersection was found when a view allong to Z-axis)
                   1 (Intersection was found when a view allong to X-axis)
                   2 (Intersection was found when a view allong to Y-axis)
                   3 (No intersection was found)
    -------
    """
    
    """
    vecAB=ln1[1]-ln1[0] # AB
    vecCD=ln2[2]-ln2[0] # CD
    vecCA=ln1[0]-ln2[1] # CA
    
    # check whether two line-segments are intersecting or not.
    comb=[\
    [0,1,2],\
    [1,2,0],\
    [2,0,1]]
    out=0
    for c in comb:
        bunbo=vecAB[c[0]]*vecCD[c[1]]-vecCD[c[0]]*vecAB[c[1]]
        if abs(bunbo)<EPS:
            t=(vecAB[c[0]]*vecCA[c[1]]-vecCA[c[0]]*vecAB[c[1]])/bunbo
            if t>=0.0 and t<=1.0:
                s=(vecCD[c[0]]*vecCA[c[1]]-vecCA[c[0]]*vecCD[c[1]])/bunbo
                if s>=0.0 and s<=1.0:
                    if abs(-s*vecAB[c[2]]+t*vecCD[c[2]]-vecCA[c[2]])<=EPS:
                        break
                    else:
                        out+=1
                else:
                    out+=1
            else:
                out+=1
        else:
            out+=1
    return out
    """
    
    # line1-A
    L1a=ln1[0]
    
    # line1-B
    L1b=ln1[1]
    
    # line2-A
    L2a=ln2[0]
    
    # line2-B
    L2b=ln2[1]
    
    vecAB=L1b-L1a
    vecAC=L2a-L1a
    vecCD=L2b-L2a
    
    # bunshi
    t1=qnv.dot(vecAC,vecCD)*qnv.dot(vecCD,vecAB)-qnv.dot(vecCD,vecCD)*qnv.dot(vecAC,vecAB)
    # bunbo
    t2=qnv.dot(vecAB,vecCD)*qnv.dot(vecCD,vecAB)-qnv.dot(vecAB,vecAB)*qnv.dot(vecCD,vecCD)
    N=La1[0].N
    qn0=qnn.Qnnum([0,0,1],N)
    qn1=qnn.Qnnum([1,0,1],N)
    if abs(t2)<qn0:  #EPS:
        return False
    else:
        s=t1/t2
        t=(-qnv.dot(vecAC,vecCD)+s*qnv.dot(vecAB,vecCD))/qnv.dot(vecCD,vecCD)
        if s>=qn0 and s<=qn1 and t>=qn0 and t<=qn1:
            dd=qn0  #0
            for i in range(3):
                dd+=((L2a[i]-L1a[i])-s*(L1b[i]-L1a[i])+t*(L2b[i]-L2a[i]))**2
            if dd<qn0:  #EPS:
                return True # intersecting
            else:
                return False
        else:
            return False
    
def triangle_area(a: qnv.Qnvec) -> qnn.Qnnum:  #-> float:
    """Numerial calcuration of area of given triangle, a.
    The coordinates of the tree vertecies of the triangle are given in TAU-style.
    
    Parameters
    ----------
    a: array containing 3-dimensional coordinates of tree vertecies of a triangle (a) in TAU-style
    
    Returns
    -------
    area of given triangle: qnnum  #float
    """
    
    #x1=a[1].v[0])-a[0].v[0]
    #y1=a[1].v[1])-a[0].v[1]
    #z1=a[1].v[2])-a[0].v[2]
    
    #x2=a[2].v[0]-a[0].v[0]
    #y2=a[2].v[1]-a[0].v[1]
    #z2=a[2].v[2]-a[0].v[2]
    
    #v1=np.array([x1,y1,z1])
    #v2=np.array([x2,y2,z2])
    v1=a[1]-a[0]
    v2=a[2]-a[0]
    
    v3=qnv.cros(v2,v1) # cross product of 3D qnvec
    vol=qnv.dot(v3,v3) # squared norm
    qn2=qnn.Qnnum([2,0,1])  # 2 in qnnum
    return abs(vol)/qn2
    #return np.sqrt(np.sum(np.abs(v3**2)))/2.0

def triangle_area_numerical(a: qnv.Qnvec) -> qnv.Qnvec:
    """Numerial calcuration of area of given triangle, a.
    The coordinates of the tree vertecies of the triangle are given.
    
    Parameters
    ----------
    a: array containing 3-dimensional coordinates of tree vertecies of a triangle (a)
    
    Returns
    -------
    area of given triangle: float
    """

    v1=a[1]-a[0]
    v2=a[2]-a[0]
    v3=qnv.cros(v2,v1) # cross product (qnnum area)
    qn2=qnn.Qnnum([2,0,1]) # 2
    return abs(v3)/qn2

def inside_outside_obj_tau(point: qnv.Qnvec, obj: qnv.Qnvec) -> bool:
    
    # TAU-style to Float
    #point=numerical_vector(point) # qnvector
    #obj=numerical_vectors(obj) # qnvector array
    # 
    #point=get_internal_component_numerical(ln)  # ln ???
    #obj=get_internal_component_sets_numerical(obj)
    return inside_outside_obj(point,obj)
    
def inside_outside_obj(point: qnv.Qnvec, obj: qnv.Qnvec) -> bool:
    """this function judges whether the point is inside an object (set of triangle) or not
        
    Parameters
    ----------
    point: array
        coordinate of the point,xyz
    obj: array
        vertex coordinates of triangle, (xyz1, xyz2, xyz3), (), (), ...
    """

    flg=0
    for tetrahedron in obj:
        if inside_outside_triangle(point,triangle):
            flg+=1
            break
    if flg==0:
        return True # inside
    else:
        return False # outside

def inside_outside_triangle_tau(point: qnv.Qnvec, triangle: qnv.Qnvec) -> bool:
    """this function judges whether the point is inside a triangle or not
        
    Parameters
    ----------
    point: array
        6d coordinate of the point in TAU-style.
    tetrahedron: array
        6d vertex coordinates of triangle in TAU-style.
    """
    point=get_internal_component_numerical(point)
    triangle=get_internal_component_sets_numerical(triangle)
    return inside_outside_triangle(point,triangle)

def inside_outside_triangle(point: qnv.Qnvec, triangle: qnv.Qnvec) -> bool:
    """this function judges whether the point is inside a tetrahedron or not
        
    Parameters
    ----------
    point: array
        coordinate of the point,xyz
    tetrahedron: array
        vertex coordinates of triangle, (xyz1, xyz2, xyz3)
    """
    area0=triangle_area_numerical(triangle)
    
    def small_triangle(indx,p,triangle0):
        tri=np.zeros((4,3),dtype=np.float64)
        for i,vt in enumerate(triangle0):  #????
            if i==indx:
                tri[i]=p
            else:
                tri[i]=vt
        return tri
    
    tet1=small_triangle(0,point,triangle)
    area1=triangle_area_numerical(tet1)
    
    tet2=small_triangle(1,point,triangle)
    area1+=triangle_area_numerical(tet2)
    
    tet3=small_triangle(2,point,triangle)
    area1+=triangle_area_numerical(tet3)
    
    if abs(area0-area1)<EPS:
        return True # inside
    else:
        return False # outside

def obj_volume_6d_numerical(obj: qnv.Qnvec) -> qnn.Qnnum:  #float:
    """This function returns volume of an object (set of triangle).
        
    Parameters
    ----------
    object: array
        6-dimensional vertex coordinates of triangle.
    """
    N=point.N
    qn0=qnn.Qnnum([0,0,1],N) # 0 in qnnum
    vol=qn0
    for triangle in obj:
        vol+=triangle_volume_6d_numerical(triangle)
    return vol

def triangle_volume_6d_numerical(triangle: qnv.Qnvec) -> qnn.Qnnum:  # float:
    """This function returns volume of a triangle
        
    Parameters
    ----------
    tetrahedron: array
        6-dimensional vertex coordinates of the tetrahedron, xyzuvw0,xyzuvw1,xyzuvw2
    """
    #a=get_internal_component_sets_numerical(triangle)
    #return triangle_volume_numerical(a)
    #return triangle_area_numerical(a)
    return triangle_area_numerical(triangle)

#def obj_volume_numerical(obj: NDArray[np.float64]) -> float:
#    """This function returns volume of an object (set of triangle).
#        
#    Parameters
#    ----------
#    object: array
#        3-dimensional vertex coordinates of triangle.
#    """
#    vol=0
#    for triangle in obj:
#        vol+=triangle_area_numerical(triangle)
#    return vol

#def triangle_area_numerical(triangle: NDArray[np.float64]) -> float:
#    """This function returns volume of a triangle
#        
#    Parameters
#    ----------
#    tetrahedron: array
#        vertex coordinates of the triangle, xyz0,xyz1,xyz2
#    """
#    xy1=np.ones((3,3),dtype=np.float64)
#    for i in range(3):
#        for j in range(3):
#            xy1[i][j]=triangle[i][j]
#    detm = np.linalg.det(xyz)
#    return abs(detm)/2










#def get_internal_component_numerical(vt: NDArray[np.int64]) -> NDArray[np.float64]:
    """
    Parameters
    ----------
    vn: array
        6-dimensional vector, xyzuvw.
    """
#    vn=numerical_vector(vt)
#    return projection3_numerical(vn)

def get_internal_component_sets_numerical(vns: qnv.Qnvec) -> qnv.Qnvec:
    """parallel and perpendicular components of a 6D lattice vector in direct space.
    
    Parameters
    ----------
    vsn: array
        set of 6-dimensional vectors, xyzuvw1, xyzuvw2, ...
    """
    #vns=numerical_vectors(vts)
    #return projection3_sets_numerical(vns)
    return projection3_sets_numerical(vns)

def projection_numerical(vn: qnv.Qnvec) -> qnv.Qnvec:
    """parallel and perpendicular components of a 6D lattice vector in direct space.
    
    Parameters
    ----------
    vn: array
        6-dimensional vector, xyzuvw.
    """
    #v1 =  TAU*vn[0]+vn[1]-0.5*vn[3] # x in Epar
    #v2 = -0.5*vn[0]+vn[2]+TAU*vn[3] # y in Epar
    #v3 = vn[4]                      # z in Epar
    #v4 = -TAU*vn[0]+vn[1]-0.5*vn[3] # x in Eperp
    #v5 = -0.5*vn[0]+vn[2]-TAU*vn[3] # y in Eperp
    #v6 = vn[5]                      # z in Epperp, dummy
    #return np.array([v1,v2,v3,v4,v5,v6],dtype=np.float64)
    return projop(vn)

def projection_sets_numerical(vns: qnv.Qnvec) -> qnv.Qnvec:
    """parallel and perpendicular components of a 6D lattice vector in direct space.
    
    Parameters
    ----------
    vsn: array
        set of 6-dimensional vectors, xyzuvw1, xyzuvw2, ...
    """
    num=len(vns)
    shape=vns.shape
    m=np.zeros(shape,dtype=qnn.Qnnum)
    for i in range(num):
        m[i]=prj.prjop_i(vns[i])
    return m
    
def projection3_numerical(vn: qnv.Qnvec) -> float:
    return prj.prjop_i(vn)
#    """perpendicular component of a 6D lattice vector in direct space.
#    
#    Parameters
#    ----------
#    vn: array
#        6-dimensional vector, xyzuvw.
#    """
#    #v1 =  TAU*vn[0]+vn[1]-0.5*vn[3] # x in Epar
#    #v2 = -0.5*vn[0]+vn[2]+TAU*vn[3] # y in Epar
#    #v3 = vn[4]                      # z in Epar
#    v4 = -TAU*vn[0]+vn[1]-0.5*vn[3]  # x in Eperp
#    v5 = -0.5*vn[0]+vn[2]-TAU*vn[3]  # y in Eperp
#    v6 = vn[5]                      # z in Epperp, dummy
#    #return np.array([v4,v5],dtype=np.float64)
#    return np.array([v4,v5,v6],dtype=np.float64)

def projection3_sets_numerical(vns: qnv.Qnvec) -> qnv.Qnvec:
    """perpendicular component of a 6D lattice vector in direct space.
    
    Parameters
    ----------
    vsn: array
        set of 6-dimensional vectors, xyzuvw1, xyzuvw2, ...
    """
    shape=vns.shape
    n=vns[0].n
    N=vns[0].N
    nc=shape[0]
    print("N",N)
    #if N==2:
    #    prj0=prj.Qnprj_Octa()
    #elif N==5:
    #    prj0=prj.Qnprj_Deca()
    #elif N==3:
    #    prj0=prj.Qnprj_Dode()
    
    if n==5:  # dihedral
        ni=2
    elif n==6: # icosahedral
        ni=3
    m=np.zeros((nc,ni),dtype=qnn.Qnnum)
    for i in range(nc):
        m[i]=projection3_numerical(vns[i])  # projection of nD lattice coordinates onto internal space
    return m

def projection_numerical_phason(vn: qnv.Qnvec,mat: qnm.Qnmat) -> qnv.Qnvec:
    """parallel and perpendicular components of a 6D lattice vector in direct space under uniform phason strain.
    
    Parameters
    ----------
    vn: array
        6-dimensional vector, xyzuvw.
    mat: array
        phason matrix
    """
    u11=mat[0][0]
    u12=mat[0][1]
    u21=mat[1][0]
    u22=mat[1][1]
    v1 =  TAU*vn[0]+vn[1]-0.5*vn[3] # x in Epar
    v2 = -0.5*vn[0]+vn[2]+TAU*vn[3] # y in Epar
    v3 = vn[4]                      # z in Epar
    v4 = (-TAU+u11*TAU-0.5*u21)*vn[0] + (1+u11)*vn[1] +     u21*vn[2] + (-0.5-0.5*u11+TAU*u21)*vn[3] # x in Eperp
    v5 = (-0.5+TAU*u12-0.5*u22)*vn[0] +     u12*vn[1] + (1+u22)*vn[2] + (-TAU-0.5*u12+TAU*u22)*vn[3] # y in Eperp
    v6= vn[5]                                                                                      # z in Epperp, dummy
    return np.array([v1,v2,v3,v4,v5,v6],dtype=np.float64)


def get_internal_component_sets_numerical(vts: qnv.Qnvec) -> qnv.Qnvec:
    """parallel and perpendicular components of a 6D lattice vector in direct space.
    
    Parameters
    ----------
    vsn: array
        set of 6-dimensional vectors, xyzuvw1, xyzuvw2, ...
    """
    #vns=numerical_vectors(vts)
    #return projection3_sets_numerical(vns)
    return projection3_sets_numerical(vts)

#########
#  WIP  #
#########
# equivalent to prjop_e
def projection_numerical_perp(vn: qnv.Qnvec) -> qnv.Qnvec:
    return prjop_e(vn)
    """This returns 6D vector which corresponds to a projection of vn onto Eperp.
    
    Parameters
    ----------
    v: array
        6-dimensional vector

    Returns
    -------
    6d vectors projected onto Eperp
    """
    #return 

#########
#  WIP  #
#########
# equivalent to prjop 
def projection_numerical_par(vn: qnv.Qnvec) -> qnv.Qnvec:
    return prjop(vn)
    """This returns 6D vector which corresponds to a projection of vn onto Epar.
    
    Parameters
    ----------
    v: array
        6-dimensional vector

    Returns
    -------
    6d vectors projected onto Eperp.
    """
    #m=np.array([\
    #        [ 0.5,          0.577350269,  0.0,         -0.288675135,  0.0,  0.0],\
    #        [ 0.288675135,  0.5,          0.288675135,  0.0,          0.0,  0.0],\
    #        [ 0.0,          0.288675135,  0.5,          0.288675135,  0.0,  0.0],\
    #        [-0.288675135,  0.0,          0.577350269,  0.5,          0.0,  0.0],\
    #        [ 0.0,          0.0,          0.0,          0.0,          1.0,  0.0],\
    #        [ 0.0,          0.0,          0.0,          0.0,          0.0,  0.0],\
    #    ])
    #return m@vn

def inout_occupation_domain_numerical(obj: qnv.Qnvec,point: qnv.Qnvec):
    """
    """
    #triangles=np.zeros((len(obj),3,3),dtype=np.float64)
    N=obj.N
    n=3
    qv=qnv.Qnvec(n,N)  # zero initialized qnvec
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
    N=triangle[0].N
    qn0=qnn.Qnnum([0,0,1],N)
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

################
# Unnecessary functions？？？
################

#def matrix_dot(m1,m2):
#    return np.dot(m1,m2)

#def inner_product_numerical(v1,v2):
#    return np.dot(v1,v2)

