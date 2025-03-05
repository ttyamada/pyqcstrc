import numpy as np
import cython

import qnnum as qnn
import qnvec as qnv
import qnmat as qnm
import numeric as num

#if __name__ == '__main__':
    
N=2 # for octagonal
n=3
ns=3
M0=qnn.Qnnum([0,0,1],N)
M1=qnn.Qnnum([1,0,1],N)
M2=qnn.Qnnum([0,1,1],N)
M3=qnn.Qnnum([1,1,2],N)

qnv0=np.zeros(ns,dtype=qnv.Qnvec) 
qnv0[0]=qnv.anyv(n,N,[M2,M3,M0])
qnv0[1]=qnv.anyv(n,N,[M0,M1,M2])
qnv0[2]=qnv.anyv(n,N,[M1,M2,M0])
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
n=5
N=2
# 8 corner vectors for AB tiling OD
vts2=qnv.zerovs((8)) # for octagon for Ammann-Beenker tiling
M0=qnn.Qnnum([0,0,1],N)
M1=qnn.Qnnum([1,0,2],N)  # 1
M2=qnn.Qnnum([-1,0,2],N) # -1
M3=qnn.Qnnum([0,1,4],N)  # sqrt(2)/2
M4=qnn.Qnnum([0,-1,4],N) # -sqrt(2)/2
# AB OD edge vectors in qnnum
vts2[0]=qnv.anyv(n,N,[M1,M0,M0,M1,M0]) #(1 0 0 1 0)/2
vts2[1]=qnv.anyv(n,N,[M0,M0,M2,M1,M0]) #(0 0 -1 1 0)/2
vts2[2]=qnv.anyv(n,N,[M0,M1,M2,M0,M0]) #(0 1 -1 0 0)/2
vts2[3]=qnv.anyv(n,N,[M2,M1,M0,M0,M0]) #(-1 1 0 0 0)/2
vts2[4]=qnv.anyv(n,N,[M2,M0,M0,M2,M0]) #(-1 0 0 -1 0)/2
vts2[5]=qnv.anyv(n,N,[M0,M0,M1,M2,M0]) #(0 0 1 -1 0)/2
vts2[6]=qnv.anyv(n,N,[M0,M2,M1,M0,M0]) #(0 -1 1 0 0)/2
vts2[7]=qnv.anyv(n,N,[M1,M2,M0,M0,M0]) #(1 -1 0 0 0)/2
qnv.printqnvs("vts2",vts2)

isys=4
prj=prj.Prjop(isys)
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
n=5
#vst=generate_random_vectors(nset)
vst=qnv.zerovs((n))
#for i in range(nset):
#    vst[i]=qnv.Qnvec(n,N)
# set vt values
vst[0]=qnv.anyv(2,N,[M0,M1])
vst[1]=qnv.anyv(2,N,[M1,M2])
vst[2]=qnv.anyv(2,N,[M1,M3])
vst[3]=qnv.anyv(2,N,[M0,M3])
vst[4]=qnv.anyv(2,N,[M2,M1])
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
qnv.printqnvs("vst_d4",vst_d4)
              
vst_d5=qnv.zerovs((16))
for i in range(16):
    vst_d5[i]=prj.prjop(vst_d4[i]) # 
qnv.printqnvs("vst_d5",vst_d5)

vst_d6=qnv.zerovs((16))
for i in range(16):
    vst_d6[i]=prj.prjop_e(vst_d4[i]) # 
qnv.printqnvs("vst_d6",vst_d6)

vst_d7=qnv.zerovs((16))
for i in range(16):
    vst_d7[i]=prj.prjop_i(vst_d4[i]) # 
qnv.printqnvs("vst_d7",vst_d7)

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
tri=qnv.zerovs(3)
print("triangle.shape",tri.shape)
qnv.printqnv("vst[0]",vst[0])
qnv.printqnv("vst[1]",vst[1])
qnv.printqnv("vst[2]",vst[2])
tri[0]=vst[0]
tri[1]=vst[1]
tri[2]=vst[2] # 3 vectors define a triangle
qnv.printqnvs("triangle",tri) # triangle
# doubled triangle
obj=np.concatenate([tri,tri]) # doubled triangle
#generator_surface_1(obj)

# a tetrahedon
#obj=triangle

#surface=generator_surface_1(obj.reshape(1,3,6,3))
#surface=obj.reshape(1,3,6,3)
print("obj.shape",obj.shape)
#surface=obj.reshape(1,6)
surface=obj
#generator_edge(surface)
generator_all_edges(surface)
    
