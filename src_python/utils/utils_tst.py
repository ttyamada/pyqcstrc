import numpy as np
import cython

import crsys
import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import numeric as num
import qnmath as qmt
import prjop as prj
from utils import (remove_doubling,
                   generator_all_edges)

#if __name__ == '__main__':

isys=4 # for octagonal
crsys.crsys_init(isys)
qnn.qnnum_init()
qnv.qnvec_init()
qnm.qnmat_init()
num.numeric_init()

N=crsys.N
n=crsys.n
ns=3
M0=qnn.Qnnum([0,0,1])
M1=qnn.Qnnum([1,0,1])
M2=qnn.Qnnum([0,1,1])
M3=qnn.Qnnum([1,1,2])

qnv0=np.zeros(ns,dtype=qnv.Qnvec) 
qnv0[0]=qnv.anyv(np.array([M2,M3,M0]))
qnv0[1]=qnv.anyv(np.array([M0,M1,M2]))
qnv0[2]=qnv.anyv(np.array([M1,M2,M0]))
for i in range(3):
    qnv.printqnv("qnv0["+format(i)+"]",qnv0[i])

qnc0=qnv.cros(qnv0[0],qnv0[1])
qnv.printqnv("qnc0",qnc0)
for i in range(3):
    qnn.printqnn("dat(qnc0,qnv0["+format(i)+"])",qnv.dot(qnc0,qnv0[i]))
#qnn.printqnn("dot(qnc0,qnv0[0])",qnv.dot(qnc0,qnv0[0])) # this should be zero
#qnn.printqnn("dot(qnc0,qnv0[1])",qnv.dot(qnc0,qnv0[1])) # this should be zero
#qnn.printqnn("dot(qnc0,qnv0[2])",qnv.dot(qnc0,qnv0[2])) # this should be non-zero



#print(vts.shape)
vinp=np.zeros(ns,dtype=qnn.Qnnum)
for i in range(ns):
    vinp[i]=qnv.dot(qnv0[i],qnv0[i])
qnn.printqnns("vinp",vinp)

#qnn.printqnn("dot(qnvo[i],qnv0[i])",vinp[i])
ip=np.zeros(ns,dtype=np.int64)
vts1=qmt.qsort(vinp,ip,ns) # use qnmath
print("ip",ip)
qnn.printqnns("vinp",vinp)
qnv.printqnv("vts1",vts1)

# nD lattice vector for defining ODs
#n=5
#N=2
# 8 corner vectors for AB tiling OD
vts2=qnv.zerovs((8,n)) # for octagon for Ammann-Beenker tiling
M0=qnn.Qnnum([0,0,1])
M1=qnn.Qnnum([1,0,2])  # 1
M2=qnn.Qnnum([-1,0,2]) # -1
M3=qnn.Qnnum([0,1,4])  # sqrt(2)/2
M4=qnn.Qnnum([0,-1,4]) # -sqrt(2)/2
# AB OD edge vectors in qnnum
vts2[0]=qnv.anyv(np.array([M1,M0,M0,M1,M0])) #(1 0 0 1 0)/2
vts2[1]=qnv.anyv(np.array([M0,M0,M2,M1,M0])) #(0 0 -1 1 0)/2
vts2[2]=qnv.anyv(np.array([M0,M1,M2,M0,M0])) #(0 1 -1 0 0)/2
vts2[3]=qnv.anyv(np.array([M2,M1,M0,M0,M0])) #(-1 1 0 0 0)/2
vts2[4]=qnv.anyv(np.array([M2,M0,M0,M2,M0])) #(-1 0 0 -1 0)/2
vts2[5]=qnv.anyv(np.array([M0,M0,M1,M2,M0])) #(0 0 1 -1 0)/2
vts2[6]=qnv.anyv(np.array([M0,M2,M1,M0,M0])) #(0 -1 1 0 0)/2
vts2[7]=qnv.anyv(np.array([M1,M2,M0,M0,M0])) #(1 -1 0 0 0)/2
qnv.printqnvs("vts2",vts2)

#isys=4
qnm.qnmat_init()
prj.prjop_init()
num.numeric_init()

# calculate internal space components of vts2
print("vts2.shape",vts2.shape)
vns2=num.get_internal_component_sets_numerical(vts2) # perp space components
qnv.printqnvs("vts2",vts2)
#for i in range(8):
#    str="vts2["+format(i)+"]"
#    qnv.printqnv(str,vts2[i])

#================
# 重複のテスト
#================
#n=5
#vst=generate_random_vectors(nset)
vst=qnv.zerovs((5,n))
#for i in range(nset):
#    vst[i]=qnv.Qnvec(n)
# set vt values
vst[0]=qnv.anyv(np.array([M0,M1]))
vst[1]=qnv.anyv(np.array([M1,M2]))
vst[2]=qnv.anyv(np.array([M1,M3]))
vst[3]=qnv.anyv(np.array([M0,M3]))
vst[4]=qnv.anyv(np.array([M2,M1]))
print("vst.shape",vst.shape)
qnv.printqnvs("vst",vst)

vst_d3=np.concatenate([vst,vst]) # doubling vst vectors
print("vst_d3.shape",vst_d3.shape) # for test
qnv.printqnvs("vst_d3",vst_d3)
#vst_d4=np.stack([vst_d3,vst_d3]) # doubling vst_d3 vectors
#qnv.printqnvs("vst_d4",vst_d4)

a=remove_doubling(vst_d3)
qnv.printqnvs("a",a)

# projection operator
qnm.printqnm("prj.prj0",prj.prj0)

vst_d4=np.concatenate([vts2,vts2])  # 5D vectors
qnv.printqnvs("vst_d4",vst_d4); print()
              
vst_d5=qnv.zerovs((16,n))
for i in range(16):
    vst_d5[i]=prj.prjvec(vst_d4[i]) # 
qnv.printqnvs("vst_d5",vst_d5); print()

vst_d6=qnv.zerovs((16,n))
for i in range(16):
    vst_d6[i]=prj.prjvec_e(vst_d4[i]) # 
qnv.printqnvs("vst_d6_e",vst_d6); print()

vst_d7=qnv.zerovs((16,n))
for i in range(16):
    vst_d7[i]=prj.prjvec_i(vst_d4[i]) # 
qnv.printqnvs("vst_d7_i",vst_d7); print()

print("vst_d7.shape",vst_d7.shape)
#a=remove_doubling_in_perp_space(vst_d7) # this does not work?
a=remove_doubling(vst_d7) # this does not work?
if len(a)==n:
    print('remove_doubling_in_perp_space: pass')
else:
    print('remove_doubling_in_perp_space: error')

#================
# 面と辺のテスト
#================
#triangle=generate_random_triangle()

# generate triangles
tri=qnv.zerovs((3,n))
print("triangle.shape",tri.shape)
qnv.printqnv("vst[0]",vst[0])
qnv.printqnv("vst[1]",vst[1])
qnv.printqnv("vst[2]",vst[2])
tri[0]=vst[0]
tri[1]=vst[1]
tri[2]=vst[2] # 3 vectors define a triangle
qnv.printqnvs("triangle",tri) # triangle

# doubled triangle
#obj=np.concatenate([tri,tri]) # doubled triangle
obj=np.stack([tri,tri]) # doubled triangle
print("obj.shape",obj.shape,"obj.ndim",obj.ndim)
qnv.printqnv("obj[0][0]",obj[0][0])
qnv.printqnv("obj[0][1]",obj[0][1])
qnv.printqnv("obj[0][2]",obj[0][2])
qnv.printqnv("obj[1][0]",obj[1][0])
qnv.printqnv("obj[1][1]",obj[1][1])
qnv.printqnv("obj[1][2]",obj[1][2])
#qnm.printqnm("obj",obj) # triangle
#generator_surface_1(obj)

# a tetrahedon
#obj=triangle

#surface=generator_surface_1(obj.reshape(1,3,6,3))
#surface=obj.reshape(1,3,6,3)
print("obj.shape",obj.shape,"obj.ndim",obj.ndim)
#surface=obj.reshape(1,6)
surface=obj
#generator_edge(surface)
generator_all_edges(surface)
    
