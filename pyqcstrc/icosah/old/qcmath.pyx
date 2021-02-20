# -*- coding: utf-8 -*-
#
# PyQCstrc
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
import sys
import numpy as np
cimport numpy as np
cimport cython
from scipy.spatial import Delaunay

#DTYPE = int
ctypedef np.float64_t DTYPE_t

#cdef np.float64_t TAU=1.618033988749895 # (1.0+np.sqrt(5.0))/2.0
cdef np.float64_t TAU=(1.0+np.sqrt(5.0))/2.0
#cdef np.float64_t BVAL=np.sqrt(3)/2.0
cdef np.float64_t EPS=1e-6 # tolerance

cpdef np.ndarray generator_obj_symmetric_tetrahedron_0(np.ndarray[np.int64_t, ndim=3] obj,np.ndarray[np.int64_t, ndim=2] centre, int numop):
	cdef int i
	cdef list od,mop
	cdef np.ndarray[np.int64_t,ndim=4] tmp4
	od=[]
	print('Generating asymmetric POD')
	mop=icosasymop()
	od.extend(symop_obj(mop[numop],obj,centre))
	tmp4=np.array(od).reshape(len(od)/72,4,6,3) # 72=4*6*3
	print(' Number of tetrahedron: %d'%(len(tmp4)))
	return tmp4	

#cpdef np.ndarray full_sym_obj_edge(np.ndarray[np.int64_t, ndim=4] od):
cpdef np.ndarray generator_edge(np.ndarray[np.int64_t, ndim=4] obj):
	# remove doubling edges in a set of triangles of the surface of OD
	# parameter: object (dim4)
	# return: doubling edges in a set of triangles (dim4)
	cdef int i,j,num1,num2,counter_ab,counter_ac,counter_bc,verbose
	cdef list edge
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b,tmp1c,tmp1d
	cdef np.ndarray[np.int64_t,ndim=2] tmp2a,tmp2b,tmp2c
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a,tmp4b#,obj
	
	verbose=0
	
	if verbose>=1:
		print('      Generating edges')
	else:
		pass
	# three edges of 1st triangle
	tmp1a=np.append(obj[0][0],obj[0][1]) #edge 0-1
	tmp1a=np.append(tmp1a,obj[0][0]) 
	tmp1a=np.append(tmp1a,obj[0][2])	 #edge 0-2
	tmp1a=np.append(tmp1a,obj[0][1]) 
	tmp1a=np.append(tmp1a,obj[0][2])	 #edge 1-2	
	num2=len(tmp1a)/2/6/3
	tmp4a=np.array(tmp1a).reshape(num2,2,6,3)
	num1=len(obj)
	for i in range(1,num1):
		tmp2a=obj[i][0]
		tmp2b=obj[i][1]
		tmp2c=obj[i][2]
		tmp1d=np.array([],dtype=int)
		counter_ab=0
		counter_ac=0
		counter_bc=0
		for j in range(0,num2):
			tmp1a=np.array([],dtype=int)
			if np.all(tmp2a==tmp4a[j][0]) and np.all(tmp2b==tmp4a[j][1]):
				counter_ab+=1
				break
			elif np.all(tmp2b==tmp4a[j][0]) and np.all(tmp2a==tmp4a[j][1]):
				counter_ab+=1
				break
			else:
				counter_ab+=0
		if counter_ab==0:
			tmp1a=np.append(tmp2a,tmp2b)
		else:
			tmp1a=np.array([],dtype=int)
		for j in range(0,num2):
			tmp1b=np.array([],dtype=int)
			if np.all(tmp2a==tmp4a[j][0]) and np.all(tmp2c==tmp4a[j][1]):
				counter_ac+=1
				break
			elif np.all(tmp2c==tmp4a[j][0]) and np.all(tmp2a==tmp4a[j][1]):
				counter_ac+=1
				break
			else:
				counter_ac+=0
		if counter_ac==0:
			tmp1b=np.append(tmp2a,tmp2c)
		else:
			tmp1b=np.array([],dtype=int)
		for j in range(0,num2):
			tmp1c=np.array([],dtype=int)	
			if np.all(tmp2b==tmp4a[j][0]) and np.all(tmp2c==tmp4a[j][1]):
				counter_bc+=1
				break
			elif np.all(tmp2c==tmp4a[j][0]) and np.all(tmp2b==tmp4a[j][1]):
				counter_bc+=1
				break
			else:
				counter_bc+=0
		if counter_bc==0:
			tmp1c=np.append(tmp2b,tmp2c)
		else:
			tmp1c=np.array([],dtype=int)
		tmp1d=np.append(tmp1d,tmp1a)
		tmp1d=np.append(tmp1d,tmp1b)
		tmp1d=np.append(tmp1d,tmp1c)
		tmp1d=np.append(tmp4a,tmp1d)
		num2=len(tmp1d)/36 # 36=2*6*3
		tmp4a=tmp1d.reshape(num2,2,6,3)
	if verbose>=1:
		print('       Number of edges: %d'%(len(tmp4a)))
	else:
		pass
	return tmp4a

cdef int equivalent_triangle(np.ndarray[np.int64_t, ndim=1] triangle1,\
							np.ndarray[np.int64_t, ndim=1] triangle2):
	cdef int i,i1,i2,i3,i4,i5,i6,counter
	cdef np.ndarray[np.int64_t,ndim=1] t1,t2,t3,a1,a2,a3,b1,b2,b3,c1,c2,c3,d1,d2,d3,e1,e2,e3,f1,f2,f3
	cdef np.ndarray[np.int64_t,ndim=2] tmp2a,tmp2b,tmp2c,tmp2d,tmp2e,tmp2f
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b
	cdef list comb
	tmp3a=triangle1.reshape(len(triangle1)/18,6,3) # 18=6*3
	tmp3b=triangle2.reshape(len(triangle2)/18,6,3)
	#triangle 1: a1,a2,a3
	#triangle 2: b1,b2,b3
	# all combinations, 6
	# a1=b1,a2=b2,a3=b3	1 2 3
	# a1=b2,a2=b3,a3=b1	2 3 1
	# a1=b3,a2=b1,a3=b2	3 1 2
	# a1=b1,a2=b3,a3=b2	1 3 2
	# a1=b3,a2=b2,a3=b1	3 2 1
	# a1=b2,a2=b1,a3=b3	2 1 3
	comb=[[0,1,2,0,1,2],\
		[0,1,2,1,2,0],\
		[0,1,2,2,0,1],\
		[0,1,2,0,2,1],\
		[0,1,2,2,1,0],\
		[0,1,2,1,0,2]]
	counter=0
	for i in range(len(comb)):
		[i1,i2,i3,i4,i5,i6]=comb[i]
		#if np.all(tmp3a[i1]==tmp3b[i4]) and np.all(tmp3a[i2]==tmp3b[i5]) and np.all(tmp3a[i3]==tmp3b[i6]):
		t1,t2,t3,a1,a2,a3=projection(tmp3a[i1][0],tmp3a[i1][1],tmp3a[i1][2],tmp3a[i1][3],tmp3a[i1][4],tmp3a[i1][5])
		t1,t2,t3,b1,b2,b3=projection(tmp3b[i4][0],tmp3b[i4][1],tmp3b[i4][2],tmp3b[i4][3],tmp3b[i4][4],tmp3b[i4][5])
		t1,t2,t3,c1,c2,c3=projection(tmp3a[i2][0],tmp3a[i2][1],tmp3a[i2][2],tmp3a[i2][3],tmp3a[i2][4],tmp3a[i2][5])
		t1,t2,t3,d1,d2,d3=projection(tmp3b[i5][0],tmp3b[i5][1],tmp3b[i5][2],tmp3b[i5][3],tmp3b[i5][4],tmp3b[i5][5])
		t1,t2,t3,e1,e2,e3=projection(tmp3a[i3][0],tmp3a[i3][1],tmp3a[i3][2],tmp3a[i3][3],tmp3a[i3][4],tmp3a[i3][5])
		t1,t2,t3,f1,f2,f3=projection(tmp3b[i6][0],tmp3b[i6][1],tmp3b[i6][2],tmp3b[i6][3],tmp3b[i6][4],tmp3b[i6][5])
		tmp2a=np.array([a1,a2,a3])
		tmp2b=np.array([b1,b2,b3])
		tmp2c=np.array([c1,c2,c3])
		tmp2d=np.array([d1,d2,d3])
		tmp2e=np.array([e1,e2,e3])
		tmp2f=np.array([f1,f2,f3])
		if np.all(tmp2a==tmp2b) and np.all(tmp2c==tmp2d) and np.all(tmp2e==tmp2f):
			counter+=1
			break
		else:
			pass
	if counter==0:
		return 1 # not equivalent traiangles
	else:
		return 0 # equivalent traiangle

# another version of equivalent_triangle()
cdef int equivalent_triangle_1(np.ndarray[np.int64_t, ndim=1] triangle1,\
							np.ndarray[np.int64_t, ndim=1] triangle2):
	cdef np.ndarray[np.int64_t,ndim=1] tmp1
	cdef np.ndarray[np.int64_t,ndim=3] tmp3
	tmp1=np.append(triangle1,triangle2)
	tmp3=remove_doubling_dim3_in_perp_space(tmp1.reshape(len(tmp1)/18,6,3))
	if len(tmp3)!=3:
		return 1 # not equivalent traiangles
	else:
		return 0 # equivalent traiangle


# new version (much faster)
cpdef np.ndarray generator_surface_1(np.ndarray[np.int64_t, ndim=4] obj):
	#
	# remove doubling surface in a set of tetrahedra in the OD (dim4)
	#
	cdef int i,j,k,val,num1,counter1,counter2,counter3,verbose
	cdef list edge,comb,list1
	cdef np.ndarray[np.int64_t,ndim=1] tmp1,tmp1a,tmp1k,tmp1j
	cdef np.ndarray[np.int64_t,ndim=2] tmp2a
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a
	
	verbose=2
	
	if verbose>0:
		print('      generator_surface_1()')
	else:
		pass
	#
	# (1) preparing a list of triangle surfaces without doubling (tmp2)
	#
	if verbose>1:
		print('       Number of triangles in POD: %d'%(len(obj)*4))
	else:
		pass
	
	#
	# initialization of tmp2 by 1st tetrahedron
	#
	# three triangles of tetrahedron: 1-2-3, 1-2-4, 1-3-4, 2-3-4
	comb=[[0,1,2],[0,1,3],[0,2,3],[1,2,3]] 
	#
	# Four triangles of 1-th tetrahedron
	for k in range(len(comb)):
		if k==0:
			tmp1=np.append(obj[0][comb[k][0]],obj[0][comb[k][1]])
		else:
			tmp1=np.append(tmp1,obj[0][comb[k][0]])
			tmp1=np.append(tmp1,obj[0][comb[k][1]])
		tmp1=np.append(tmp1,obj[0][comb[k][2]])
	num1=len(tmp1)/len(comb)
	tmp2a=tmp1.reshape(len(comb),num1)
	#

	list1=[]
	for i in range(1,len(obj)): # i-th tetrahedron
		tmp1=np.array([0])
		for k in range(len(comb)): # k-th triangle of i-th tetrahedron
			tmp1k=np.append(obj[i][comb[k][0]],obj[i][comb[k][1]])
			tmp1k=np.append(tmp1k,obj[i][comb[k][2]])
			counter1=0
			for j in range(len(tmp2a)): # j-th triangle in list 'tmp2'
				tmp1j=tmp2a[j]
				#val=equivalent_triangle(tmp1k,tmp1j) # val=0 (equivalent), 1 (non equivalent)
				val=equivalent_triangle_1(tmp1k,tmp1j) # val=0 (equivalent), 1 (non equivalent)
				if val==0:
					counter1+=1
					list1.append(j)
					break
				else:
					pass
			if counter1==0:
				if len(tmp1)==1:
					tmp1=tmp1k
				else:
					tmp1=np.append(tmp1,tmp1k)
		if len(tmp1)==1:
			pass
		else:
			tmp1=np.append(tmp2a,tmp1)
			tmp2a=tmp1.reshape(len(tmp1)/num1,num1)

	if verbose>1:
		print('       Number of unique triangles: %d'%(len(tmp2a)))
	else:
		pass

	tmp1=np.array([0])
	for i in range(len(tmp2a)):
		counter1=0
		for j in list1:
			if i==j:
				counter1+=1
				#break
			else:
				pass
		if counter1==0:
			if len(tmp1)==1:
				tmp1=tmp2a[i]
			else:
				tmp1=np.append(tmp1,tmp2a[i])
		elif counter1==1:
			pass
		else:
			if verbose>1:
				print('      ERROR_001 %d: check your model.'%(counter1))
			else:
				pass
				
	if verbose>=1:
		print('       Number of triangles on POD surface:%d'%(len(tmp1)/54))
	else:
		pass
	
	return tmp1.reshape(len(tmp1)/54,3,6,3)

cpdef np.ndarray generator_surface(np.ndarray[np.int64_t, ndim=4] obj):
	#
	# remove doubling surface in a set of tetrahedra in the OD (dim4)
	#
	cdef int i,j,k,val,num1,counter1,counter2,counter3,verbose
	cdef list edge,comb
	cdef np.ndarray[np.int64_t,ndim=1] tmp1,tmp1a,tmp1k,tmp1j
	cdef np.ndarray[np.int64_t,ndim=2] tmp2
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a
	
	verbose=2
	
	if verbose>0:
		print('      Generating surfaces()')
	else:
		pass
	#
	# (1) preparing a list of triangle surfaces without doubling (tmp2)
	#
	if verbose>=1:
		print('       Number of triangles in POD: %d'%(len(obj)*4))
	else:
		pass
	
	#
	# initialization of tmp2 by 1st tetrahedron
	#
	# three triangles of tetrahedron: 1-2-3, 1-2-4, 1-3-4, 2-3-4
	comb=[[0,1,2],[0,1,3],[0,2,3],[1,2,3]] 
	#
	# Four triangles of 1-th tetrahedron
	for k in range(len(comb)):
		if k==0:
			tmp1=np.append(obj[0][comb[k][0]],obj[0][comb[k][1]])
		else:
			tmp1=np.append(tmp1,obj[0][comb[k][0]])
			tmp1=np.append(tmp1,obj[0][comb[k][1]])
		tmp1=np.append(tmp1,obj[0][comb[k][2]])
	num1=len(tmp1)/len(comb)
	tmp2=tmp1.reshape(len(comb),num1)
	#
	for i in range(1,len(obj)): # i-th tetrahedron
		for k in range(len(comb)): # k-th triangle of i-th tetrahedron
			tmp1k=np.append(obj[i][comb[k][0]],obj[i][comb[k][1]])
			tmp1k=np.append(tmp1k,obj[i][comb[k][2]])
			counter1=0
			for j in range(len(tmp2)): # j-th triangle in list 'tmp2'
				tmp1j=tmp2[j]
				val=equivalent_triangle(tmp1k,tmp1j) # val=0 (equivalent), 1 (non equivalent)
				if val==0:
					counter1+=1
					break
				else:
					pass
			if counter1==0:
				tmp1=np.append(tmp2,tmp1k)
				tmp2=tmp1.reshape(len(tmp1)/num1,num1)
			else:
				pass
	if verbose>1:
		print('       Number of unique triangles: %d'%(len(tmp2)))
	else:
		pass
	#
	# (2) how many each unique triangle in "tmp2" is found in "obj", and count.
	#	 if a triangle is NOT part of surface of OD, we can find the triangle in "obj" two times.
	counter2=0
	counter3=0
	for j in range(len(tmp2)): # j-th triangle in the unique triangle list 'tmp2'
		tmp1j=tmp2[j]
		counter1=0
		for i in range(0,len(obj)): # i-th tetrahedron in 'obj'
			for k in range(len(comb)): # k-th triangke of i-th tetrahedron
				tmp1i=np.append(obj[i][comb[k][0]],obj[i][comb[k][1]])
				tmp1i=np.append(tmp1i,obj[i][comb[k][2]])
				val=equivalent_triangle(tmp1i,tmp1j) # val=0 (equivalent), 1 (non equivalent)
				if val==0:
					counter1+=1
					if counter1==2:
						break
					else:
						pass
				else:
					pass
		if counter1==1:
			if counter2==0:
				tmp1a=tmp1j
			else:
				tmp1a=np.append(tmp1a,tmp1j)
			counter2+=1
		elif counter1==2:
			pass
		elif counter1>2:
			counter3+=1
		else:
			pass
	if counter3>0:
		if verbose>=1:
			print('      ERROR_001 %d: check your model.'%(counter3))
		else:
			pass
		return np.array([0]).reshape(1,1,1,1)
	else:
		tmp4a=tmp1a.reshape(len(tmp1a)/54,3,6,3) # 54=3*6*3
		if verbose>=1:
			print('       Number of triangles on POD surface:%d'%(counter2))
		else:
			pass
		return tmp4a

cpdef np.ndarray intersection_line_segment_triangle(np.ndarray[np.int64_t, ndim=3] line_segment,np.ndarray[np.int64_t, ndim=3] triangle):
	cdef int i,j,counter,num1,num2,num3
	cdef np.ndarray[np.int64_t,ndim=1] p,tmp,tmp1
	#
	# -----------------
	# triangle
	# -----------------
	# vertex 1: triangle[0],  consist of (a1+b1*TAU)/c1, ... (a6+b6*TAU)/c6	a_i,b_i,c_i = tetrahedron_1[0][i:0~5][0],triangle[0][i:0~5][1],triangle[0][i:0~5][2]
	# vertex 2: triangle[1]
	# vertex 3: triangle[2]
	#
	# triangle
	# triangle 1: v1,v2,v3
	#
	# -----------------
	# line_segment
	# -----------------
	# line_segment[i][0][0],line_segment[i][0][1],line_segment[i][0][2],line_segment[i][0][3],line_segment[i][0][4],line_segment[i][0][5]
	# line_segment[i][1][0],line_segment[i][1][1],line_segment[i][1][2],line_segment[i][1][3],line_segment[i][1][4],line_segment[i][1][5]
	#
	#
	# intersection between (edge) and (triangle_1)
	#
	# combination_index
	# e.g. v1,v2,w1,w2,w3 (edge 1 and triangle 1) ...
	#
	# edges
	# triangle_1
	segment_1=line_segment[0]
	segment_2=line_segment[1]
	surface_1=triangle[0]
	surface_2=triangle[1]
	surface_3=triangle[2]
	p=intersection_segment_surface(segment_1,segment_2,surface_1,surface_2,surface_3)
	return p

cpdef intersection_using_surface(np.ndarray[np.int64_t, ndim=4] obj1_surface,\
								np.ndarray[np.int64_t, ndim=4] obj2_surface,\
								np.ndarray[np.int64_t, ndim=4] obj1_edge,\
								np.ndarray[np.int64_t, ndim=4] obj2_edge,\
								np.ndarray[np.int64_t, ndim=4] obj1_tetrahedron,\
								np.ndarray[np.int64_t, ndim=4] obj2_tetrahedron,\
								path):
	# This is very simple but work correctly only when each subdivided 
	# three ODs (i.e part part, ODA and ODB) are able to define as a
	# set of tetrahedra.
	cdef int i1,i2,i3,counter,counter1,counter2,num1,num2
	cdef np.ndarray[np.int64_t,ndim=1] p,tmp,tmp1,tmp1a,tmp1b
	cdef np.ndarray[np.int64_t,ndim=2] point_tmp,point1_tmp
	cdef np.ndarray[np.int64_t,ndim=3] tet
	cdef np.ndarray[np.int64_t,ndim=3] tr1,tr2,ed1,ed2,tmp3
	cdef np.ndarray[np.int64_t,ndim=3] point_a,point_a1,point_a2
	cdef np.ndarray[np.int64_t,ndim=3] point_b,point_b1,point_b2
	cdef np.ndarray[np.int64_t,ndim=3] point,point1,point_common
	cdef np.ndarray[np.int64_t,ndim=4] obj_common,obj_1,obj_2
	cdef list comb
	counter=0
	print('Intersecting of OD1 and ODB')
	#print 'intersection_using_surface()'
	for i1 in range(len(obj1_surface)):
		tr1=obj1_surface[i1]
		for i2 in range(len(obj2_edge)):
			ed2=obj2_edge[i2]
			tmp=intersection_line_segment_triangle(ed2,tr1)
			if len(tmp)!=1:
				if counter==0:
					p=tmp
				else:
					p=np.append(p,tmp)
				counter+=1
			else:
				pass
	for i1 in range(len(obj2_surface)):
		tr2=obj2_surface[i1]
		for i2 in range(len(obj1_edge)):
			ed1=obj1_edge[i2]
			tmp=intersection_line_segment_triangle(ed1,tr2)
			if len(tmp)!=1:
				if counter==0:
					p=tmp
				else:
					p=np.append(p,tmp)
				counter+=1
			else:
				pass
	if counter==0:
		print(' No intersection')
		return np.array([0],dtype=int).reshape(1,1,1,1)
	else:
		point=p.reshape(len(p)/18,6,3) # 18=6*3
		point1=remove_doubling_dim3(point)
		print(' dividing into three PODs:')
		print('    Common   :     OD1 and     ODB')
		print('  UnCommon 1 :     OD1 and Not OD2')
		print('  UnCommon 2 : Not OD1 and     OD2')
		#
		#	 OD1 and ODB	 : common part (point_common)
		#	 OD1 and Not OD2 : ODA (point_a)
		# Not OD1 and OD2	 : ODB (point_b)
		#
		# --------------------------
		# (1) Extract vertces of 2nd OD which are insede 1st OD --> point_a1
		#	 Extract vertces of 2nd OD which are outsede 1st OD --> point_b2
		#
		counter1=0
		counter2=0
		tmp3=remove_doubling_dim4(obj1_surface) # generating vertces of 1st OD	
		for i1 in range(len(tmp3)):
			point_tmp=tmp3[i1]
			counter=0
			for i2 in range(len(obj2_tetrahedron)):
				tet=obj2_tetrahedron[i2]
				num1=inside_outside_tetrahedron(point_tmp,tet[0],tet[1],tet[2],tet[3])
				if num1==0:
					counter+=1
					break
				else:
					pass
			if counter>0:
				if counter1==0:
					tmp1a=point_tmp.reshape(18) # 18=6*3
				else:
					tmp1a=np.append(tmp1a,point_tmp)
				counter1+=1
			else:
				if counter2==0:
					tmp1b=point_tmp.reshape(18)
				else:
					tmp1b=np.append(tmp1b,point_tmp)
				counter2+=1
		point_a1=tmp1a.reshape(len(tmp1a)/18,6,3)
		point_b2=tmp1b.reshape(len(tmp1b)/18,6,3)
		#
		# (2) Extract vertces of 1st OD which are insede 2nd OD --> point_b1
		#	 Extract vertces of 1st OD which are outsede 2nd OD --> point_a2
		#
		counter1=0
		counter2=0
		tmp3=remove_doubling_dim4(obj2_surface) # generating vertces of 2nd OD
		for i1 in range(len(tmp3)):
			point_tmp=tmp3[i1]
			counter=0
			for i2 in range(len(obj1_tetrahedron)):
				tet=obj1_tetrahedron[i2]
				num1=inside_outside_tetrahedron(point_tmp,tet[0],tet[1],tet[2],tet[3])
				if num1==0:
					counter+=1
					break
				else:
					pass
			if counter>0:
				if counter1==0:
					tmp1a=point_tmp.reshape(18) # 18=6*3
				else:
					tmp1a=np.append(tmp1a,point_tmp)
				counter1+=1
			else:
				if counter2==0:
					tmp1b=point_tmp.reshape(18)
				else:
					tmp1b=np.append(tmp1b,point_tmp)
				counter2+=1
		point_b1=tmp1a.reshape(len(tmp1a)/18,6,3)
		point_a2=tmp1b.reshape(len(tmp1b)/18,6,3)
		#
		# (3) Sum point A, point B and Intersections --->>> common part
		#
		# common part = point1 + point_a1 + point_b1
		tmp=np.append(point1,point_a1)
		tmp=np.append(tmp,point_b1)
		point_common=tmp.reshape(len(tmp)/18,6,3) # 18=6*3
		point_common=remove_doubling_dim3(point_common)
		#
		# point_a = point_a1 + point_a2 + point1
		tmp=np.append(point1,point_a1)
		tmp=np.append(tmp,point_a2)
		point_a=tmp.reshape(len(tmp)/18,6,3)
		point_a=remove_doubling_dim3(point_a)
		#
		# point_b = point_b1 + point_b2 + point1
		tmp=np.append(point1,point_b1)
		tmp=np.append(tmp,point_b2)
		point_b=tmp.reshape(len(tmp)/18,6,3)
		point_b=remove_doubling_dim3(point_b)
		#
		return point_common,point_a,point_b

cpdef np.ndarray shift_object(np.ndarray[np.int64_t, ndim=4] obj,np.ndarray[np.int64_t, ndim=2] shift, int vorbose):
	cdef int i1,i2,i3
	cdef long n1,n2,n3
	cdef long v0,v1,v2,v3,v4,v5
	cdef double vol1,vol2
	cdef list a
	cdef np.ndarray[np.int64_t, ndim=4] obj_new
	
	a=[]
	v0,v1,v2=obj_volume_6d(obj)
	#vol0=(v0+TAU*v1)/float(v2)
	vol1=obj_volume_6d_numerical(obj)
	
	if vorbose>0:
		print('shift_object()')
		if vorbose>1:
			if vorbose>0:
				print(' volume: %d %d %d (%10.8f)'%(v0,v1,v2,vol1))
			else:
				pass
	else:
		pass
	
	for i1 in range(len(obj)):
		for i2 in range(4):
			for i3 in range(6):
				n1,n2,n3=add(obj[i1][i2][i3][0],obj[i1][i2][i3][1],obj[i1][i2][i3][2],shift[i3][0],shift[i3][1],shift[i3][2])
				a.append(n1)
				a.append(n2)
				a.append(n3)
	obj_new=np.array(a).reshape(len(obj),4,6,3)
	
	v3,v4,v5=obj_volume_6d(obj_new)
	vol2=obj_volume_6d_numerical(obj_new)
	if v0==v3 and v1==v4 and v2==v5 or abs(vol1-vol2)<vol1*EPS:
		if vorbose>0:
			print(' succeeded')
		else:
			pass
		return obj_new
	else:
		print(' fail')
		return np.array([[[[0]]]])


cdef int rough_check_intersection_two_tetrahedron(np.ndarray[np.int64_t, ndim=3] tetrahedron1,\
		 											np.ndarray[np.int64_t, ndim=3] tetrahedron2):
	cdef double dd1,dd2
	cdef np.ndarray[np.int64_t,ndim=2] cent1,cent2
	
	cent1=centroid(tetrahedron1)
	cent2=centroid(tetrahedron2)
	dd1=ball_radius(tetrahedron1,cent1)
	dd2=ball_radius(tetrahedron2,cent2)
	dd0=distance_in_perp_space(cent1,cent2)
	if dd0<dd1+dd2: # two balls are intersecting.
		return 0
	else: #
		return 1

cdef int rough_check_intersection_tetrahedron_obj(np.ndarray[np.int64_t, ndim=3] tetrahedron,\
		 											np.ndarray[np.int64_t, ndim=4] obj):
	cdef double dd1,dd2
	cdef np.ndarray[np.int64_t,ndim=2] cent1,cent2
	
	cent1=centroid(tetrahedron)
	cent2=centroid_obj(obj)
	dd1=ball_radius(tetrahedron,cent1)
	dd2=ball_radius_obj(obj,cent2)
	dd0=distance_in_perp_space(cent1,cent2)
	if dd0<dd1+dd2: # two balls are intersecting.
		return 0
	else: #
		return 1

cdef int rough_check_intersection_two_obj(np.ndarray[np.int64_t, ndim=4] obj1,\
		 									np.ndarray[np.int64_t, ndim=4] obj2):
	cdef double dd1,dd2
	cdef np.ndarray[np.int64_t,ndim=2] cent1,cent2
	
	cent1=centroid_obj(obj1)
	cent2=centroid_obj(obj2)
	dd1=ball_radius_obj(obj1,cent1)
	dd2=ball_radius_obj(obj2,cent2)
	dd0=distance_in_perp_space(cent1,cent2)
	if dd0<dd1+dd2: # two balls are intersecting.
		return 0
	else: #
		return 1

cdef np.ndarray intersection_two_tetrahedron(np.ndarray[np.int64_t, ndim=3] tetrahedron_1,\
											np.ndarray[np.int64_t, ndim=3] tetrahedron_2,\
											int verbose):
	if rough_check_intersection_two_tetrahedron(tetrahedron_1,tetrahedron_2)==0:
		return intersection_two_tetrahedron_4(tetrahedron_1,tetrahedron_2,verbose)
	else:
		return np.array([[[[0]]]])

cdef np.ndarray intersection_tetrahedron_obj(np.ndarray[np.int64_t, ndim=3] tetrahedron,\
											np.ndarray[np.int64_t, ndim=4] obj,\
											int verbose):
	if verbose>0:
		print(' intersection_tetrahedron_obj()')
	else:
		pass
	if rough_check_intersection_tetrahedron_obj(tetrahedron,obj)==0:
		return intersection_tetrahedron_obj_4(tetrahedron,obj,verbose)
	else:
		return np.array([[[[0]]]])

cpdef np.ndarray intersection_two_obj(np.ndarray[np.int64_t, ndim=4] obj1,\
									np.ndarray[np.int64_t, ndim=4] obj2,\
									int verbose):
	cdef int i1,counter
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a,tmp4_common
	counter=0
	tmp4_common=np.array([[[[0]]]])
	if verbose>0:
		print(' intersection_two_obj()')
	else:
		pass
	if rough_check_intersection_two_obj(obj1,obj2)==0:
		for i1 in range(len(obj1)):
			tmp4a=intersection_tetrahedron_obj(obj1[i1],obj2,verbose)
			if len(tmp4a)!=1:
				if counter==0:
					tmp1a=tmp4a.reshape(len(tmp4a)*72)
				else:
					tmp1a=np.append(tmp1a,tmp4a)
				counter+=1
			else:
				pass
	else:
		pass
	if counter>0:
		tmp4_common=tmp1a.reshape(len(tmp1a)/72,4,6,3)
	else:
		pass
	return tmp4_common

cpdef np.ndarray intersection_using_tetrahedron_4(np.ndarray[np.int64_t, ndim=4] obj1,\
												np.ndarray[np.int64_t, ndim=4] obj2,\
												int option,\
												int verbose,\
												int dummy):
	#
	# Intersection; obj1 and obj2
	#
	cdef int i1,i2,i3,i4,counter1,counter2,counter3,counter4,counter5,q1,q2,q3,flag,num,flag1,flag2
	cdef long v1a,v1b,v1c,v2a,v2b,v2c,v3a,v3b,v3c
	cdef double vol1,vol2,vol3,vol4,vol5
	cdef np.ndarray[np.int64_t,ndim=1] tmp1b,tmp1c
	cdef np.ndarray[np.int64_t,ndim=4] tmp4_common,tmp4a,tmp4b
	
	if verbose>0:
		print(' intersection_using_tetrahedron_4()')
	else:
		pass
	
	v1a,v1b,v1c=obj_volume_6d(obj1)
	vol2=obj_volume_6d_numerical(obj1)
	if verbose>1:
		print('   obj1, volume = %d %d %d (%10.8f)'%(v1a,v1b,v1c,vol2))
		print('                = ',obj_volume_6d_numerical(obj1))
	else:
		pass	
	v1a,v1b,v1c=obj_volume_6d(obj2)
	vol2=obj_volume_6d_numerical(obj2)
	if verbose>1:
		print('   obj2, volume = %d %d %d (%10.8f)'%(v1a,v1b,v1c,vol2))
	else:
		pass

	counter3=0
	tmp1b=np.array([0])
	tmp1c=np.array([0])
	tmp4_common=np.array([0]).reshape(1,1,1,1)
	for i1 in range(len(obj1)):
		# volume check
		v1a,v1b,v1c=tetrahedron_volume_6d(obj1[i1])
		vol1=tetrahedron_volume_6d_numerical(obj1[i1])
		if verbose>1:
			print('  %2d-th tetrahedron in obj1, %d %d %d (%10.8f)'%(i1,v1a,v1b,v1c,vol1))
		else:
			pass
		counter1=0
		for i2 in range(len(obj2)):
			if verbose>2:
				v2a,v2b,v2c=tetrahedron_volume_6d(obj2[i2])
				vol2=tetrahedron_volume_6d_numerical(obj2[i2])
				print('  %2d-th tetrahedron in obj2, %d %d %d (%10.8f)'%(i2,v2a,v2b,v2c,vol2))
			else:
				pass
			tmp4_common=intersection_two_tetrahedron_4(obj1[i1],obj2[i2],verbose-1)
			
			if tmp4_common.tolist()!=[[[[0]]]]: # if common part is not empty
				if counter1==0:
					tmp1c=tmp4_common.reshape(72*len(tmp4_common)) # 72=4*6*3
				else:
					tmp1c=np.append(tmp1c,tmp4_common)
				counter1+=1
			else:
				pass
			
		if counter1!=0:
			tmp4a=tmp1c.reshape(len(tmp1c)/72,4,6,3)
			if option==0:
				####################
				## simplification ##
				####################
				#tmp4a=simplification_convex_polyhedron(tmp4a,2,verbose-1)
				pass
			else:
				pass
			# volume check
			v3a,v3b,v3c=obj_volume_6d(tmp4a)
			vol3=(v3a+v3b*TAU)/float(v3c)
			if v3a==v1a and v3b==v1b and v3c==v1c: # vol1==vol3 ＃ この場合、obj1[i1]全体が、obj2に含まれている
				if verbose>1:
					print('                 common part (all), %d %d %d (%10.8f)'%(v3a,v3b,v3c,vol3))
				else:
					pass
				if counter3==0:
					tmp1b=obj1[i1].reshape(72)
				else:
					tmp1b=np.append(tmp1b,obj1[i1])
				counter3+=1
			else: # to avoid overflow
				vol4=obj_volume_6d_numerical(tmp4a)
				if abs(vol1-vol4)<=1e-8:
					if verbose>1:
						print('                 common part (all), %d %d %d (%10.8f)'%(v1a,v1b,v1c,vol4))
					else:
						pass
					if counter3==0:
						tmp1b=obj1[i1].reshape(72)
					else:
						tmp1b=np.append(tmp1b,obj1[i1])
					counter3+=1
				elif abs(vol1-vol4)>1e-8 and vol4>=0.0:
					if abs(vol4-vol3)<1e-8:
						if verbose>1:
							print('                 common part (partial_1), %d %d %d (%10.8f)'%(v3a,v3b,v3c,vol4))
						else:
							pass
						if counter3==0:
							tmp1b=tmp4a.reshape(len(tmp4a)*72)
						else:
							tmp1b=np.append(tmp1b,tmp4a)
						counter3+=1
					else:
						tmp4b=tmp4a
						vol4=obj_volume_6d_numerical(tmp4b)
						v3a,v3b,v3c=obj_volume_6d(tmp4b)
						vol3=(v3a+v3b*TAU)/float(v3c)
						if abs(vol4-vol3)<1e-8:
							if verbose>1:
								print('                 common part (partial_2), %d %d %d (%10.8f)'%(v3a,v3b,v3c,vol4))
							else:
								pass
							if counter3==0:
								tmp1b=tmp4b.reshape(len(tmp4b)*72)
							else:
								tmp1b=np.append(tmp1b,tmp4b)
							counter3+=1
						else:
							if verbose>1:
								print('                 common part (partial_3), %d %d %d (%10.8f)'%(v3a,v3b,v3c,vol4))
							else:
								pass
							if counter3==0:
								tmp1b=tmp4a.reshape(len(tmp4a)*72)
							else:
								tmp1b=np.append(tmp1b,tmp4a)
							counter3+=1
				else:
					if verbose>1:
						print('                 common part, %d %d %d (%10.8f) out of expected!'%(v3a,v3b,v3c,vol3))
						print('                 numerical value, %10.8f'%(vol4))
					else:
						pass
					pass
		else: # if common part (obj1_reduced and obj2) is NOT empty
			if verbose>1:
				print('                 common part, empty')
			else:
				pass
			pass

	if counter3!=0:
		tmp4_common=tmp1b.reshape(len(tmp1b)/72,4,6,3)
		v1a,v1b,v1c=obj_volume_6d(tmp4_common)
		vol2=obj_volume_6d_numerical(tmp4_common)
		if verbose>1:
			print('   common obj, volume = %d %d %d (%10.8f)'%(v1a,v1b,v1c,vol2))
		else:
			pass
		return tmp4_common
	else:
		if tmp4_common.tolist()!=[[[[0]]]]:
			return tmp4_common
		else:
			return np.array([[[[0]]]])

cdef double ball_radius(np.ndarray[np.int64_t, ndim=3] tetrahedron,\
						np.ndarray[np.int64_t, ndim=2] centroid):
	#  this transforms a tetrahedron to a boll which covers the tetrahedron
	#  the centre of the boll is the centroid of the tetrahedron.
	cdef int i1,i2,counter
	cpdef long w1,w2,w3
	cdef np.ndarray[np.int64_t,ndim=1] v1,v2,v3,v4,v5,v6
	cdef double dd0,radius
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a
	cdef np.ndarray[np.int64_t,ndim=3] tmp3b
	
	counter=0
	tmp1a=np.array([0])
	for i1 in range(4):
		for i2 in range(6):
			w1,w2,w3=sub(centroid[i2][0],centroid[i2][1],centroid[i2][2],tetrahedron[i1][i2][0],tetrahedron[i1][i2][1],tetrahedron[i1][i2][2])
			if counter!=0:
				tmp1a=np.append(tmp1a,[w1,w2,w3])
			else:
				tmp1a=np.array([w1,w2,w3])
				counter+=1
	tmp3b=tmp1a.reshape(4,6,3)
	radius=0.0
	for i1 in range(4):
		v1,v2,v3,v4,v5,v6=projection(tmp3b[i1][0],tmp3b[i1][1],tmp3b[i1][2],tmp3b[i1][3],tmp3b[i1][4],tmp3b[i1][5])
		dd=np.sqrt(((v4[0]+v4[1]*TAU)/float(v4[2]))**2+((v5[0]+v5[1]*TAU)/float(v5[2]))**2+((v6[0]+v6[1]*TAU)/float(v6[2]))**2)
		if dd>radius:
			radius=dd
		else:
			pass
	return radius

cdef double ball_radius_obj(np.ndarray[np.int64_t, ndim=4] obj,\
						np.ndarray[np.int64_t, ndim=2] centroid):
	#  this transforms an OBJ to a boll which covers the OBJ
	#  the centre of the boll is the centroid of the OBJ.
	cdef int i1,i2,i3,num1,counter
	cpdef long w1,w2,w3
	cdef np.ndarray[np.int64_t,ndim=1] v1,v2,v3,v4,v5,v6
	cdef double dd0,radius
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a
	
	counter=0
	tmp1a=np.array([0])
	num1=len(obj)
	for i1 in range(num1):
		for i2 in range(4):
			for i3 in range(6):
				w1,w2,w3=sub(centroid[i3][0],centroid[i3][1],centroid[i3][2],obj[i1][i2][i3][0],obj[i1][i2][i3][1],obj[i1][i2][i3][2])
				if counter!=0:
					tmp1a=np.append(tmp1a,[w1,w2,w3])
				else:
					tmp1a=np.array([w1,w2,w3])
					counter+=1
	tmp4a=tmp1a.reshape(num1,4,6,3)
	radius=0.0
	for i1 in range(num1):
		for i2 in range(4):
			v1,v2,v3,v4,v5,v6=projection(tmp4a[i1][i2][0],tmp4a[i1][i2][1],tmp4a[i1][i2][2],tmp4a[i1][i2][3],tmp4a[i1][i2][4],tmp4a[i1][i2][5])
			dd=np.sqrt(((v4[0]+v4[1]*TAU)/float(v4[2]))**2+((v5[0]+v5[1]*TAU)/float(v5[2]))**2+((v6[0]+v6[1]*TAU)/float(v6[2]))**2)
			if dd>radius:
				radius=dd
			else:
				pass
	return radius

cdef double distance_in_perp_space(np.ndarray[np.int64_t, ndim=2] pos1,\
		 					np.ndarray[np.int64_t, ndim=2] pos2):
	cdef int i1
	cpdef long w1,w2,w3
	cdef np.ndarray[np.int64_t,ndim=1] v1,v2,v3,v4,v5,v6
	cdef double dd
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a
	cdef np.ndarray[np.int64_t,ndim=2] tmp2a
	
	tmp1a=np.array([0])
	for i1 in range(6):
		w1,w2,w3=sub(pos1[i1][0],pos1[i1][1],pos1[i1][2],pos2[i1][0],pos2[i1][1],pos2[i1][2])
		if i1!=0:
			tmp1a=np.append(tmp1a,[w1,w2,w3])
		else:
			tmp1a=np.array([w1,w2,w3])
	tmp2a=tmp1a.reshape(6,3)
	v1,v2,v3,v4,v5,v6=projection(tmp2a[0],tmp2a[1],tmp2a[2],tmp2a[3],tmp2a[4],tmp2a[5])
	dd=np.sqrt(((v4[0]+v4[1]*TAU)/float(v4[2]))**2+((v5[0]+v5[1]*TAU)/float(v5[2]))**2+((v6[0]+v6[1]*TAU)/float(v6[2]))**2)
	return dd

cdef np.ndarray intersection_tetrahedron_obj_4(np.ndarray[np.int64_t, ndim=3] tetrahedron,\
											np.ndarray[np.int64_t, ndim=4] obj,\
											int verbose):
	#
	# Intersection; tetrahedron and obj
	#
	cdef int i1,i2,i3,i4,counter1,counter2,counter3,counter4,counter5,q1,q2,q3,flag,num,flag1,flag2
	cdef long v1a,v1b,v1c,v2a,v2b,v2c,v3a,v3b,v3c
	cdef double vol1,vol2,vol3,vol4,vol5
	cdef np.ndarray[np.int64_t,ndim=1] tmp1b,tmp1c
	cdef np.ndarray[np.int64_t,ndim=4] tmp4_common,tmp4a,tmp4b
	
	if verbose>0:
		print(' intersection_tetrahedron_obj_4()')
	else:
		pass

	counter3=0
	tmp1b=np.array([0])
	tmp1c=np.array([0])
	tmp4_common=np.array([0]).reshape(1,1,1,1)
	
	# volume check
	v1a,v1b,v1c=tetrahedron_volume_6d(tetrahedron)
	vol1=tetrahedron_volume_6d_numerical(tetrahedron)
	if verbose>1:
		print('  tetrahedron, %d %d %d (%10.8f)'%(v1a,v1b,v1c,vol1))
	else:
		pass
	counter1=0
	for i2 in range(len(obj)):
		if verbose>2:
			v2a,v2b,v2c=tetrahedron_volume_6d(obj[i2])
			vol2=tetrahedron_volume_6d_numerical(obj[i2])
			print('  %2d-th tetrahedron in obj, %d %d %d (%10.8f)'%(i2,v2a,v2b,v2c,vol2))
		else:
			pass
		#tmp4_common=intersection_two_tetrahedron_4(tetrahedron,obj[i2],verbose-1)
		tmp4_common=intersection_two_tetrahedron(tetrahedron,obj[i2],verbose-1)
		
		if tmp4_common.tolist()!=[[[[0]]]]: # if common part is not empty
			if counter1==0:
				tmp1c=tmp4_common.reshape(72*len(tmp4_common)) # 72=4*6*3
			else:
				tmp1c=np.append(tmp1c,tmp4_common)
			counter1+=1
		else:
			pass
		
	if counter1!=0:
		tmp4a=tmp1c.reshape(len(tmp1c)/72,4,6,3)
		# volume check
		v3a,v3b,v3c=obj_volume_6d(tmp4a)
		vol3=(v3a+v3b*TAU)/float(v3c)
		if v3a==v1a and v3b==v1b and v3c==v1c: # vol1==vol3 ＃ この場合、tetrahedron全体が、objに含まれている
			if verbose>1:
				print('                 common part (all), %d %d %d (%10.8f)'%(v3a,v3b,v3c,vol3))
			else:
				pass
			if counter3==0:
				tmp1b=tetrahedron.reshape(72)
			else:
				tmp1b=np.append(tmp1b,tetrahedron)
			counter3+=1
		else: # to avoid overflow
			vol4=obj_volume_6d_numerical(tmp4a)
			if abs(vol1-vol4)<=1e-8:
				if verbose>1:
					print('                 common part (all), %d %d %d (%10.8f)'%(v1a,v1b,v1c,vol4))
				else:
					pass
				if counter3==0:
					tmp1b=tetrahedron.reshape(72)
				else:
					tmp1b=np.append(tmp1b,tetrahedron)
				counter3+=1
			elif abs(vol1-vol4)>1e-8 and vol4>=0.0:
				if abs(vol4-vol3)<1e-8:
					if verbose>1:
						print('                 common part (partial_1), %d %d %d (%10.8f)'%(v3a,v3b,v3c,vol4))
					else:
						pass
					if counter3==0:
						tmp1b=tmp4a.reshape(len(tmp4a)*72)
					else:
						tmp1b=np.append(tmp1b,tmp4a)
					counter3+=1
				else:
					tmp4b=tmp4a
					vol4=obj_volume_6d_numerical(tmp4b)
					v3a,v3b,v3c=obj_volume_6d(tmp4b)
					vol3=(v3a+v3b*TAU)/float(v3c)
					if abs(vol4-vol3)<1e-8:
						if verbose>1:
							print('                 common part (partial_2), %d %d %d (%10.8f)'%(v3a,v3b,v3c,vol4))
						else:
							pass
						if counter3==0:
							tmp1b=tmp4b.reshape(len(tmp4b)*72)
						else:
							tmp1b=np.append(tmp1b,tmp4b)
						counter3+=1
					else:
						if verbose>1:
							print('                 common part (partial_3), %d %d %d (%10.8f)'%(v3a,v3b,v3c,vol4))
						else:
							pass
						if counter3==0:
							tmp1b=tmp4a.reshape(len(tmp4a)*72)
						else:
							tmp1b=np.append(tmp1b,tmp4a)
						counter3+=1
			else:
				if verbose>1:
					print('                 common part, %d %d %d (%10.8f) out of expected!'%(v3a,v3b,v3c,vol3))
					print('                 numerical value, %10.8f'%(vol4))
				else:
					pass
				pass
	else: # if common part (obj1_reduced and obj2) is NOT empty
		if verbose>1:
			print('                 common part, empty')
		else:
			pass
		pass
	if counter3!=0:
		tmp4_common=tmp1b.reshape(len(tmp1b)/72,4,6,3)
		v1a,v1b,v1c=obj_volume_6d(tmp4_common)
		vol2=obj_volume_6d_numerical(tmp4_common)
		if verbose>1:
			print('   common obj, volume = %d %d %d (%10.8f)'%(v1a,v1b,v1c,vol2))
		else:
			pass
		return tmp4_common
	else:
		if tmp4_common.tolist()!=[[[[0]]]]:
			return tmp4_common
		else:
			return np.array([[[[0]]]])

cdef np.ndarray intersection_two_tetrahedron_4(np.ndarray[np.int64_t, ndim=3] tetrahedron_1,\
												np.ndarray[np.int64_t, ndim=3] tetrahedron_2,\
												int verbose):
	cdef int i1,i2,counter1,counter2,num1,num2
	cdef long a1,b1,c1,a2,b2,c2
	cdef float vol1
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b,tmp1c,v1,v2,v3,v4,v5,v6
	cdef np.ndarray[np.int64_t,ndim=2] comb
	cdef np.ndarray[np.int64_t,ndim=2] tmp2a,tmp2b,tmp2c
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a
	
	if verbose>0:
		print('      intersection_two_tetrahedron_4()')
	else:
		pass
	#
	# -----------------
	# tetrahedron_1
	# -----------------
	# vertex 1: tetrahedron_1[0],  consist of (a1+b1*TAU)/c1, ... (a6+b6*TAU)/c6	a_i,b_i,c_i = tetrahedron_1[0][i:0~5][0],tetrahedron_1[0][i:0~5][1],tetrahedron_1[0][i:0~5][2]
	# vertex 2: tetrahedron_1[1]
	# vertex 3: tetrahedron_1[2]
	# vertex 4: tetrahedron_1[3]	
	#
	# 4 surfaces of tetrahedron_1
	# surface 1: v1,v2,v3
	# surface 2: v1,v2,v4
	# surface 3: v1,v3,v4
	# surface 4: v2,v3,v4
	#
	# 6 edges of tetrahedron_1
	# edge 1: v1,v2
	# edge 2: v1,v3
	# edge 3: v1,v4
	# edge 4: v2,v3
	# edge 5: v2,v4
	# edge 6: v3,v4
	#
	# -----------------
	# tetrahedron_2
	# -----------------
	# vertex 1: tetrahedron_2[0]
	# vertex 2: tetrahedron_2[1]
	# vertex 3: tetrahedron_2[2]
	# vertex 4: tetrahedron_2[3]	
	#
	# 4 surfaces of tetrahedron_2
	# surface 1: w1,w2,w3
	# surface 2: w1,w2,w4
	# surface 3: w1,w3,w4
	# surface 4: w2,w3,w4
	#
	# 6 edges of tetrahedron_2
	# edge 1: w1,w2
	# edge 2: w1,w3
	# edge 3: w1,w4
	# edge 4: w2,w3
	# edge 5: w2,w4
	# edge 6: w3,w4
	#
	# case 1: intersection between (edge of tetrahedron_1) and (surface of tetrahedron_2)
	# case 2: intersection between (edge of tetrahedron_2) and (surface of tetrahedron_1)
	#
	# combination_index
	# e.g. v1,v2,w1,w2,w3 (edge 1 and surface 1) ...
	comb=np.array([\
	[0,1,0,1,2],\
	[0,1,0,1,3],\
	[0,1,0,2,3],\
	[0,1,1,2,3],\
	[0,2,0,1,2],\
	[0,2,0,1,3],\
	[0,2,0,2,3],\
	[0,2,1,2,3],\
	[0,3,0,1,2],\
	[0,3,0,1,3],\
	[0,3,0,2,3],\
	[0,3,1,2,3],\
	[1,2,0,1,2],\
	[1,2,0,1,3],\
	[1,2,0,2,3],\
	[1,2,1,2,3],\
	[1,3,0,1,2],\
	[1,3,0,1,3],\
	[1,3,0,2,3],\
	[1,3,1,2,3],\
	[2,3,0,1,2],\
	[2,3,0,1,3],\
	[2,3,0,2,3],\
	[2,3,1,2,3]])
	
	tmp1a=np.array([0])
	tmp1b=np.array([0])
	tmp1c=np.array([0])

	counter1=0
	for i1 in range(len(comb)): # len(combination_index) = 24
		# case 1: intersection between
		# 6 edges of tetrahedron_1
		# 4 surfaces of tetrahedron_2
		segment_1=tetrahedron_1[comb[i1][0]] 
		segment_2=tetrahedron_1[comb[i1][1]]
		surface_1=tetrahedron_2[comb[i1][2]]
		surface_2=tetrahedron_2[comb[i1][3]]
		surface_3=tetrahedron_2[comb[i1][4]]
		tmp1c=intersection_segment_surface(segment_1,segment_2,surface_1,surface_2,surface_3)
		if len(tmp1c)!=1:
			if counter1==0 :
				tmp1a=tmp1c # intersection
			else:
				tmp1a=np.append(tmp1a,tmp1c) # intersecting points
			counter1+=1
		else:
			pass
		# case 2: intersection between
		# 6 edges of tetrahedron_2
		# 4 surfaces of tetrahedron_1
		segment_1=tetrahedron_2[comb[i1][0]]
		segment_2=tetrahedron_2[comb[i1][1]]
		surface_1=tetrahedron_1[comb[i1][2]]
		surface_2=tetrahedron_1[comb[i1][3]]
		surface_3=tetrahedron_1[comb[i1][4]]
		tmp1c=intersection_segment_surface(segment_1,segment_2,surface_1,surface_2,surface_3)
		if len(tmp1c)!=1:
			if counter1==0:
				tmp1a=tmp1c # intersection
			else:
				tmp1a=np.append(tmp1a,tmp1c) # intersecting points
			counter1+=1
		else:
			pass

	# get vertces of tetrahedron_1 which are inside tetrahedron_2
	for i1 in range(len(tetrahedron_1)):
		flag=inside_outside_tetrahedron(tetrahedron_1[i1],tetrahedron_2[0],tetrahedron_2[1],tetrahedron_2[2],tetrahedron_2[3])
		if flag==0:
			if counter1==0:
				tmp1a=tetrahedron_1[i1].reshape(18)
			else:
				tmp1a=np.append(tmp1a,tetrahedron_1[i1])
			counter1+=1
		else:
			pass
	# get vertces of tetrahedron_2 which are inside tetrahedron_1
	for i1 in range(len(tetrahedron_2)):
		flag=inside_outside_tetrahedron(tetrahedron_2[i1],tetrahedron_1[0],tetrahedron_1[1],tetrahedron_1[2],tetrahedron_1[3])
		if flag==0:
			if counter1==0:
				tmp1a=tetrahedron_2[i1].reshape(18)
			else:
				tmp1a=np.append(tmp1a,tetrahedron_2[i1])
			counter1+=1
		else:
			pass
	
	#tmp4a=np.array([0]).reshape(1,1,1,1)
	tmp4a=np.array([[[[0]]]])
	if counter1!=0:
		# remove doubling
		tmp3a=tmp1a.reshape(len(tmp1a)/18,6,3)
		tmp3a=remove_doubling_dim3_in_perp_space(tmp3a)
		
		if len(tmp3a)>=4:
			if verbose>2:
				print('       number of points for tetrahedralization, %d'%(len(tmp3a)))
			else:
				pass
			# Tetrahedralization
			if coplanar_check(tmp3a)==0:
				if len(tmp3a)==4:
					tmp4a=tmp3a.reshape(1,4,6,3)
				else:
					tmp4a=tetrahedralization_points(tmp3a)
				num1=len(tmp4a)
				if num1!=1:
					if verbose>2:
						print('       -> number of tetrahedron,  %d'%(len(tmp4a)))
					else:
						pass
					for i1 in range(num1):
						tmp2c=centroid(tmp4a[i1])
						# check tmp2c is inside both tetrahedron_1 and 2
						flag1=inside_outside_tetrahedron(tmp2c,tetrahedron_1[0],tetrahedron_1[1],tetrahedron_1[2],tetrahedron_1[3])
						flag2=inside_outside_tetrahedron(tmp2c,tetrahedron_2[0],tetrahedron_2[1],tetrahedron_2[2],tetrahedron_2[3])
						if verbose>2:
							print('         tetraheddron %d'%(i1))
							
							for i2 in range(4):
								v1,v2,v3,v4,v5,v6=projection(tmp4a[i1][i2][0],tmp4a[i1][i2][1],tmp4a[i1][i2][2],tmp4a[i1][i2][3],tmp4a[i1][i2][4],tmp4a[i1][i2][5])
								print('          Qq %8.6f %8.6f %8.6f'%((v4[0]+v4[1]*TAU)/float(v4[2]),(v5[0]+v5[1]*TAU)/float(v5[2]),(v6[0]+v6[1]*TAU)/float(v6[2])))
								#v1,v2,v3,v4,v5,v6=projection(tmp4a[i1][0][0],tmp4a[i1][0][1],tmp4a[i1][0][2],tmp4a[i1][0][3],tmp4a[i1][0][4],tmp4a[i1][0][5])
								#print('          Qq %8.6f %8.6f %8.6f'%((v4[0]+v4[1]*TAU)/float(v4[2]),(v5[0]+v5[1]*TAU)/float(v5[2]),(v6[0]+v6[1]*TAU)/float(v6[2])))
								#v1,v2,v3,v4,v5,v6=projection(tmp4a[i1][1][0],tmp4a[i1][1][1],tmp4a[i1][1][2],tmp4a[i1][1][3],tmp4a[i1][1][4],tmp4a[i1][1][5])
								#print('          Qq %8.6f %8.6f %8.6f'%((v4[0]+v4[1]*TAU)/float(v4[2]),(v5[0]+v5[1]*TAU)/float(v5[2]),(v6[0]+v6[1]*TAU)/float(v6[2])))
								#v1,v2,v3,v4,v5,v6=projection(tmp4a[i1][2][0],tmp4a[i1][2][1],tmp4a[i1][2][2],tmp4a[i1][2][3],tmp4a[i1][2][4],tmp4a[i1][2][5])
								#print('          Qq %8.6f %8.6f %8.6f'%((v4[0]+v4[1]*TAU)/float(v4[2]),(v5[0]+v5[1]*TAU)/float(v5[2]),(v6[0]+v6[1]*TAU)/float(v6[2])))
								#v1,v2,v3,v4,v5,v6=projection(tmp4a[i1][3][0],tmp4a[i1][3][1],tmp4a[i1][3][2],tmp4a[i1][3][3],tmp4a[i1][3][4],tmp4a[i1][3][5])
								#print('          Qq %8.6f %8.6f %8.6f'%((v4[0]+v4[1]*TAU)/float(v4[2]),(v5[0]+v5[1]*TAU)/float(v5[2]),(v6[0]+v6[1]*TAU)/float(v6[2])))
							
							# volume of tetrahedron
							a1,b1,c1=tetrahedron_volume_6d(tmp4a[i1])
							vol1=(a1+b1*TAU)/float(c1)
							print('         volume, %d %d %d (%8.6f)'%(a1,b1,c1,vol1))
							print('         volume (numerical value)',tetrahedron_volume_6d_numerical(tmp4a[i1]))
						else:
							pass
					
						if flag1==0 and flag2==0: # inside
							if verbose>2:
								print('              in')
							else:
								pass
							if len(tmp1b)==1:
								tmp1b=tmp4a[i1].reshape(72)
							else:
								tmp1b=np.append(tmp1b,tmp4a[i1])
						else:
							if verbose>2:
								print('              out (%d,%d)'%(flag1,flag2))
							else:
								pass
							pass
					num2=len(tmp1b)
					if num2!=1:
						if num2/72==num1: # 全体が入っている場合
							return tmp4a
						else:  # 一部が交差している場合
							return tmp1b.reshape(num2/72,4,6,3)
					else:
						return tmp4a
				else:
					return tmp4a
			else:
				return tmp4a
		else:
			return tmp4a
	else:
		return tmp4a

cpdef np.ndarray intersection_two_segment(np.ndarray[np.int64_t, ndim=2] segment_1_A,\
										np.ndarray[np.int64_t, ndim=2] segment_1_B,\
										np.ndarray[np.int64_t, ndim=2] segment_2_C,\
										np.ndarray[np.int64_t, ndim=2] segment_2_D):
	cdef double s,t
	cdef long ax1,ax2,ax3,ay1,ay2,ay3,az1,az2,az3
	cdef long bx1,bx2,bx3,by1,by2,by3,bz1,bz2,bz3
	cdef long cx1,cx2,cx3,cy1,cy2,cy3,cz1,cz2,cz3
	cdef long a1,a2,a3,b1,b2,b3,c1,c2,c3,d1,d2,d3,e1,e2,e3,f1,f2,f3,g1,g2,g3
	cdef long m1,m2,m3,n1,n2,n3,o1,o2,o3,p1,p2,p3
	cdef long z1,z2,z3
	cdef long ddx1,ddx2,ddx3
	cdef long h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16,h17,h18
	cdef long i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18
	cdef long j1,j2,j3,j4,j5,j6,j7,j8,j9,j10,j11,j12,j13,j14,j15,j16,j17,j18
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b,tmp1c
	cdef np.ndarray[np.int64_t,ndim=1] seg1ai1,seg1ai2,seg1ai3,seg1bi1,seg1bi2,seg1bi3
	cdef np.ndarray[np.int64_t,ndim=1] seg2ci1,seg2ci2,seg2ci3,seg2di1,seg2di2,seg2di3
	cdef int verbose
	
	verbose=0
	
	if verbose>=1:
		print('                        intersection_two_segment()')
	else:
		pass
		
	if verbose>=2:
		print('segment_1',segment_1_A)
		print('segment_1',segment_1_B)
		print('segment_2',segment_2_C)
		print('segment_2',segment_2_D)
	else:
		pass
	tmp1a,tmp1b,tmp1c,seg1ai1,seg1ai2,seg1ai3=projection(segment_1_A[0],segment_1_A[1],segment_1_A[2],segment_1_A[3],segment_1_A[4],segment_1_A[5])
	tmp1a,tmp1b,tmp1c,seg1bi1,seg1bi2,seg1bi3=projection(segment_1_B[0],segment_1_B[1],segment_1_B[2],segment_1_B[3],segment_1_B[4],segment_1_B[5])
	tmp1a,tmp1b,tmp1c,seg2ci1,seg2ci2,seg2ci3=projection(segment_2_C[0],segment_2_C[1],segment_2_C[2],segment_2_C[3],segment_2_C[4],segment_2_C[5])
	tmp1a,tmp1b,tmp1c,seg2di1,seg2di2,seg2di3=projection(segment_2_D[0],segment_2_D[1],segment_2_D[2],segment_2_D[3],segment_2_D[4],segment_2_D[5])
	
	# vec AB
	[ax1,ax2,ax3]=sub(seg1bi1[0],seg1bi1[1],seg1bi1[2],seg1ai1[0],seg1ai1[1],seg1ai1[2])
	[ay1,ay2,ay3]=sub(seg1bi2[0],seg1bi2[1],seg1bi2[2],seg1ai2[0],seg1ai2[1],seg1ai2[2])
	[az1,az2,az3]=sub(seg1bi3[0],seg1bi3[1],seg1bi3[2],seg1ai3[0],seg1ai3[1],seg1ai3[2])

	# vec AC
	[bx1,bx2,bx3]=sub(seg2ci1[0],seg2ci1[1],seg2ci1[2],seg1ai1[0],seg1ai1[1],seg1ai1[2])
	[by1,by2,by3]=sub(seg2ci2[0],seg2ci2[1],seg2ci2[2],seg1ai2[0],seg1ai2[1],seg1ai2[2])
	[bz1,bz2,bz3]=sub(seg2ci3[0],seg2ci3[1],seg2ci3[2],seg1ai3[0],seg1ai3[1],seg1ai3[2])
	
	# vec CD
	[cx1,cx2,cx3]=sub(seg2di1[0],seg2di1[1],seg2di1[2],seg2ci1[0],seg2ci1[1],seg2ci1[2])
	[cy1,cy2,cy3]=sub(seg2di2[0],seg2di2[1],seg2di2[2],seg2ci2[0],seg2ci2[1],seg2ci2[2])
	[cz1,cz2,cz3]=sub(seg2di3[0],seg2di3[1],seg2di3[2],seg2ci3[0],seg2ci3[1],seg2ci3[2])
	
	# dot_product(vecAC,vecCD)
	a1,a2,a3=dot_product(np.array([bx1,bx2,bx3]),np.array([by1,by2,by3]),np.array([bz1,bz2,bz3]),np.array([cx1,cx2,cx3]),np.array([cy1,cy2,cy3]),np.array([cz1,cz2,cz3]))
	# dot_product(vecCD,vecAB)
	b1,b2,b3=dot_product(np.array([cx1,cx2,cx3]),np.array([cy1,cy2,cy3]),np.array([cz1,cz2,cz3]),np.array([ax1,ax2,ax3]),np.array([ay1,ay2,ay3]),np.array([az1,az2,az3]))
	# dot_product(vecCD,vecCD)
	c1,c2,c3=dot_product(np.array([cx1,cx2,cx3]),np.array([cy1,cy2,cy3]),np.array([cz1,cz2,cz3]),np.array([cx1,cx2,cx3]),np.array([cy1,cy2,cy3]),np.array([cz1,cz2,cz3]))
	# dot_product(vecAC,vecAB)
	d1,d2,d3=dot_product(np.array([bx1,bx2,bx3]),np.array([by1,by2,by3]),np.array([bz1,bz2,bz3]),np.array([ax1,ax2,ax3]),np.array([ay1,ay2,ay3]),np.array([az1,az2,az3]))
	# dot_product(vecAB,vecCD)
	e1,e2,e3=dot_product(np.array([ax1,ax2,ax3]),np.array([ay1,ay2,ay3]),np.array([az1,az2,az3]),np.array([cx1,cx2,cx3]),np.array([cy1,cy2,cy3]),np.array([cz1,cz2,cz3]))
	# dot_product(vecAB,vecAB)
	f1,f2,f3=dot_product(np.array([ax1,ax2,ax3]),np.array([ay1,ay2,ay3]),np.array([az1,az2,az3]),np.array([ax1,ax2,ax3]),np.array([ay1,ay2,ay3]),np.array([az1,az2,az3]))

	#bunbo=dot_product(vecAB,vecCD)*dot_product(vecCD,vecAB)-dot_product(vecAB,vecAB)*dot_product(vecCD,vecCD)
	m1,m2,m3=mul(e1,e2,e3,b1,b2,b3)
	n1,n2,n3=mul(f1,f2,f3,c1,c2,c3)
	p1,p2,p3=sub(m1,m2,m3,n1,n2,n3)	
	tmp1a=np.array([0])
	if p1!=0 or p2!=0:
		# bunshi=dot_product(vecAC,vecCD)*dot_product(vecCD,vecAB)-dot_product(vecCD,vecCD)*dot_product(vecAC,vecAB)
		m1,m2,m3=mul(a1,a2,a3,b1,b2,b3)
		n1,n2,n3=mul(c1,c2,c3,d1,d2,d3)
		o1,o2,o3=sub(m1,m2,m3,n1,n2,n3)
		#
		# Numerical calc
		#
		# s=bunshi/bunbo
		s1,s2,s3=div(o1,o2,o3,p1,p2,p3)
		s=(s1+TAU*s2)/float(s3)
		#t=(-dot_product(vecAC,vecCD)+s*dot_product(vecAB,vecCD))/dot_product(vecCD,vecCD)
		# dot_product(vecAB,vecCD)
		g1,g2,g3=dot_product(np.array([ax1,ax2,ax3]),np.array([ay1,ay2,ay3]),np.array([az1,az2,az3]),np.array([cx1,cx2,cx3]),np.array([cy1,cy2,cy3]),np.array([cz1,cz2,cz3]))
		h1,h2,h3=mul(s1,s2,s3,g1,g2,g3)
		h1,h2,h3=sub(h1,h2,h3,a1,a2,a3)
		if verbose>=2:
			print('                           s = %8.6f'%(s))
		else:
			pass
		if c1!=0 or c2!=0:
			t1,t2,t3=div(h1,h2,h3,c1,c2,c3)
			t=(t1+TAU*t2)/float(t3)
			if verbose>=2:
				print('                           t = %8.6f'%(t))
			else:
				pass
			if s>=0.0 and s<=1.0 and t>=0.0 and t<=1.0:
				# ddx=(L2ax-L1ax)-s*(L1bx-L1ax)+t*(L2bx-L2ax)
				#bx1,bx2,bx3 # (L2ax-L1ax)
				#mul(s1,s2,s3,ax1,ax2,ax3) # s*(L1bx-L1ax)
				#mul(t1,t2,t3,cx1,cx2,cx3) # t*(L2bx-L2ax)
				ddx1,ddx2,ddx3=mul(s1,s2,s3,ax1,ax2,ax3)
				ddx1,ddx2,ddx3=sub(bx1,bx2,bx3,ddx1,ddx2,ddx3)
				z1,z2,z3=mul(t1,t2,t3,cx1,cx2,cx3)
				ddx1,ddx2,ddx3=add(ddx1,ddx2,ddx3,z1,z2,z3)
				# ddx**2
				ddx1,ddx2,ddx3=mul(ddx1,ddx2,ddx3,ddx1,ddx2,ddx3)
				if verbose>=2:
					print('      ddx1,ddx2 = %d %d'%(ddx1,ddx2))
				else:
					pass
				# ddy=(L2ay-L1ay)-s*(L1by-L1ay)+t*(L2by-L2ay)
				ddy1,ddy2,ddy3=mul(s1,s2,s3,ay1,ay2,ay3)
				ddy1,ddy2,ddy3=sub(by1,by2,by3,ddy1,ddy2,ddy3)
				z1,z2,z3=mul(t1,t2,t3,cy1,cy2,cy3)
				ddy1,ddy2,ddy3=add(ddy1,ddy2,ddy3,z1,z2,z3)
				# ddy**2
				ddy1,ddy2,ddy3=mul(ddy1,ddy2,ddy3,ddy1,ddy2,ddy3)
				if verbose>=2:
					print('      ddy1,ddy2 = %d %d'%(ddy1,ddy2))
				else:
					pass
				# ddz=(L2az-L1az)-s*(L1bz-L1az)+t*(L2bz-L2az)
				ddz1,ddz2,ddz3=mul(s1,s2,s3,az1,az2,az3)
				ddz1,ddz2,ddz3=sub(bz1,bz2,bz3,ddz1,ddz2,ddz3)
				z1,z2,z3=mul(t1,t2,t3,cz1,cz2,cz3)
				ddz1,ddz2,ddz3=add(ddz1,ddz2,ddz3,z1,z2,z3)
				# ddz**2
				ddz1,ddz2,ddz3=mul(ddz1,ddz2,ddz3,ddz1,ddz2,ddz3)
				if verbose>=2:
					print('      ddz1,ddz2 = %d %d'%(ddz1,ddz2))
				else:
					pass
					
				z1,z2,z3=add(ddx1,ddx2,ddx3,ddy1,ddy2,ddy3)
				z1,z2,z3=add(z1,z2,z3,ddz1,ddz2,ddz3)
				if verbose>=2:
					print('      z1,z2 = %d %d'%(z1,z2))
				else:
					pass
				#if ddx**2+ddy**2+ddz**2<EPS:
				#if ddx1==0 and ddx2==0 and ddy1==0 and ddy2==0 and ddz1==0 and ddz2==0:
				if z1==0 and z2==0:
					#
					#interval=line1a+s*(line1b-line1a)
					#
					# line1b-line1a
					[h1,h2,h3]=sub(segment_1_B[0][0],segment_1_B[0][1],segment_1_B[0][2],segment_1_A[0][0],segment_1_A[0][1],segment_1_A[0][2])
					[h4,h5,h6]=sub(segment_1_B[1][0],segment_1_B[1][1],segment_1_B[1][2],segment_1_A[1][0],segment_1_A[1][1],segment_1_A[1][2])
					[h7,h8,h9]=sub(segment_1_B[2][0],segment_1_B[2][1],segment_1_B[2][2],segment_1_A[2][0],segment_1_A[2][1],segment_1_A[2][2])
					[h10,h11,h12]=sub(segment_1_B[3][0],segment_1_B[3][1],segment_1_B[3][2],segment_1_A[3][0],segment_1_A[3][1],segment_1_A[3][2])
					[h13,h14,h15]=sub(segment_1_B[4][0],segment_1_B[4][1],segment_1_B[4][2],segment_1_A[4][0],segment_1_A[4][1],segment_1_A[4][2])
					[h16,h17,h18]=sub(segment_1_B[5][0],segment_1_B[5][1],segment_1_B[5][2],segment_1_A[5][0],segment_1_A[5][1],segment_1_A[5][2])
					#
					# line1a
					i1,i2,i3=segment_1_A[0][0],segment_1_A[0][1],segment_1_A[0][2]
					i4,i5,i6=segment_1_A[1][0],segment_1_A[1][1],segment_1_A[1][2]
					i7,i8,i9=segment_1_A[2][0],segment_1_A[2][1],segment_1_A[2][2]
					i10,i11,i12=segment_1_A[3][0],segment_1_A[3][1],segment_1_A[3][2]
					i13,i14,i15=segment_1_A[4][0],segment_1_A[4][1],segment_1_A[4][2]
					i16,i17,i18=segment_1_A[5][0],segment_1_A[5][1],segment_1_A[5][2]
					#
					[j1,j2,j3]=mul(s1,s2,s3,h1,h2,h3)
					[j1,j2,j3]=add(j1,j2,j3,i1,i2,i3)
					#
					[j4,j5,j6]=mul(s1,s2,s3,h4,h5,h6)
					[j4,j5,j6]=add(j4,j5,j6,i4,i5,i6)
					#
					[j7,j8,j9]=mul(s1,s2,s3,h7,h8,h9)
					[j7,j8,j9]=add(j7,j8,j9,i7,i8,i9)
					#
					[j10,j11,j12]=mul(s1,s2,s3,h10,h11,h12)
					[j10,j11,j12]=add(j10,j11,j12,i10,i11,i12)
					#
					[j13,j14,j15]=mul(s1,s2,s3,h13,h14,h15)
					[j13,j14,j15]=add(j13,j14,j15,i13,i14,i15)
					#
					[j16,j17,j18]=mul(s1,s2,s3,h16,h17,h18)
					[j16,j17,j18]=add(j16,j17,j18,i16,i17,i18)
					tmp1a=np.array([j1,j2,j3,j4,j5,j6,j7,j8,j9,j10,j11,j12,j13,j14,j15,j16,j17,j18])
					if verbose>=2:
						print tmp1a
					else:
						pass
				else:
					pass
			else:
				pass
		else:
			pass
	else:
		pass
	return tmp1a

cpdef np.ndarray tetrahedralization_points(np.ndarray[np.int64_t, ndim=3] points):
	cdef int i,num,counter,verbose
	cdef long t1,t2,t3,t4,v1,v2,v3,w1,w2,w3
	cdef double vx,vy,vz
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,xe,ye,ze,xi,yi,zi
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a
	cdef np.ndarray[DTYPE_t,ndim=1] tmp1v
	cdef np.ndarray[DTYPE_t,ndim=2] tmp2v
	cdef list ltmp
	
	verbose=0
	tmp3a=points
	#print '   N of points: %3d :'%len(intersecting_point)
	for i in range(len(tmp3a)):
		xe,ye,ze,xi,yi,zi=projection(tmp3a[i][0],tmp3a[i][1],tmp3a[i][2],tmp3a[i][3],tmp3a[i][4],tmp3a[i][5])
		vx=(xi[0]+xi[1]*TAU)/float(xi[2]) # numeric value of xi
		vy=(yi[0]+yi[1]*TAU)/float(yi[2])
		vz=(zi[0]+zi[1]*TAU)/float(zi[2])
		if i==0:
			tmp1v=np.array([vx,vy,vz])
			#print '     %8.6f %8.6f %8.6f'%(vx,vy,vz)
		else:
			tmp1v=np.append(tmp1v,[vx,vy,vz])
			#print '     %8.6f %8.6f %8.6f'%(vx,vy,vz)
	tmp2v=tmp1v.reshape(len(tmp1v)/3,3)
	ltmp=decomposition(tmp2v)
	
	if verbose==1:
		print('   -> N of tetrahedron: %3d'%(len(ltmp)))
	else:
		pass
	
	if ltmp!=[0]:
		w1,w2,w3=0,0,1
		counter=0
		for i in range(len(ltmp)):
			tmp3b=np.array([tmp3a[ltmp[i][0]],tmp3a[ltmp[i][1]],tmp3a[ltmp[i][2]],tmp3a[ltmp[i][3]]]).reshape(4,6,3)
			#print 'tmp3b',tmp3b
			[v1,v2,v3]=tetrahedron_volume_6d(tmp3b)
			if v1==0 and v2==0:
				if verbose==1:
					print('     %d-th tet, empty'%(i))
				else:
					pass
				pass
			#elif (v1+v2*TAU)/float(v3)<0.0:
			#	print '     %d-th tet, volume : %d %d %d (%8.6f) ignored!'%(i,v1,v2,v3,(v1+v2*TAU)/float(v3))
			#	pass
			else:
				if counter==0:
					tmp1a=tmp3b.reshape(72) # 4*6*3=72
					if verbose==1:
						print('     %d-th tet, volume : %d %d %d (%8.6f)'%(i,v1,v2,v3,(v1+v2*TAU)/float(v3)))
					else:
						pass
				else:
					tmp1a=np.append(tmp1a,tmp3b)
					if verbose==1:
						print('     %d-th tet, volume : %d %d %d (%8.6f)'%(i,v1,v2,v3,(v1+v2*TAU)/float(v3)))
					else:
						pass
				w1,w2,w3=add(v1,v2,v3,w1,w2,w3)
				counter+=1
		if counter!=0:
			tmp4a=tmp1a.reshape(len(tmp1a)/72,4,6,3) # 4*6*3=72	
			if verbose==1:
				print('     -> Total : %d %d %d (%8.6f)'%(w1,w2,w3,(w1+w2*TAU)/float(w3)))
			else:
				pass
		else:
			tmp4a=np.array([0],dtype=int).reshape(1,1,1,1)
		return tmp4a
	else:
		print('tmp2v',tmp2v)
		return np.array([0],dtype=int).reshape(1,1,1,1)

cpdef np.ndarray tetrahedralization(np.ndarray[np.int64_t, ndim=3] tetrahedron,\
									np.ndarray[np.int64_t, ndim=3] intersecting_point):
	cdef int i,num,counter
	cdef long t1,t2,t3,t4,v1,v2,v3,w1,w2,w3
	cdef double vx,vy,vz
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,xe,ye,ze,xi,yi,zi
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a
	cdef np.ndarray[DTYPE_t,ndim=1] tmp1v
	cdef np.ndarray[DTYPE_t,ndim=2] tmp2v
	cdef list ltmp
	# 
	#print ' tetrahedralization()'
	#[v1,v2,v3]=tetrahedron_volume_6d(tetrahedron) 	# check volume of 'tetrahedron'
	#print '   volume of tetrahedron base: (%d+%d*TAU)/%d'%(v1,v2,v3)
	tmp1a=np.append(tetrahedron,intersecting_point)
	tmp3a=tmp1a.reshape(len(tmp1a)/18,6,3)
	#print '   N of points: %3d :'%len(intersecting_point)
	for i in range(len(tmp3a)):
		xe,ye,ze,xi,yi,zi=projection(tmp3a[i][0],tmp3a[i][1],tmp3a[i][2],tmp3a[i][3],tmp3a[i][4],tmp3a[i][5])
		vx=(xi[0]+xi[1]*TAU)/float(xi[2]) # numeric value of xi
		vy=(yi[0]+yi[1]*TAU)/float(yi[2])
		vz=(zi[0]+zi[1]*TAU)/float(zi[2])
		if i==0:
			tmp1v=np.array([vx,vy,vz])
			#print '     %8.6f %8.6f %8.6f'%(vx,vy,vz)
		else:
			tmp1v=np.append(tmp1v,[vx,vy,vz])
			#print '     %8.6f %8.6f %8.6f'%(vx,vy,vz)
	tmp2v=tmp1v.reshape(len(tmp1v)/3,3)
	ltmp=decomposition(tmp2v)
	if ltmp!=[0]:
		#print '   -> N of tetrahedron: %3d'%(len(ltmp))
		w1,w2,w3=0,0,1
		counter=0
		for i in range(len(ltmp)):
			tmp3b=np.array([tmp3a[ltmp[i][0]],tmp3a[ltmp[i][1]],tmp3a[ltmp[i][2]],tmp3a[ltmp[i][3]]],dtype=int).reshape(4,6,3)
			[v1,v2,v3]=tetrahedron_volume_6d(tmp3b)
			if v1==0 and v2==0:
				pass
			else:
				if counter==0:
					tmp1a=tmp3b.reshape(72) # 4*6*3=72
					#print '     volume : (%d+%d*TAU)/%d'%(v1,v2,v3)
				else:
					tmp1a=np.append(tmp1a,tmp3b)
					#print '     volume : (%d+%d*TAU)/%d'%(v1,v2,v3)
					w1,w2,w3=add(v1,v2,v3,w1,w2,w3)
				counter+=1
		if counter!=0:
			tmp4a=tmp1a.reshape(len(tmp1a)/72,4,6,3) # 4*6*3=72	
			#print '     -> Total : (%d+%d*TAU)/%d'%(w1,w2,w3)
		else:
			tmp4a=np.array([[[[0]]]])
		return tmp4a
	else:
		return np.array([[[[0]]]])

cdef list decomposition(np.ndarray[DTYPE_t, ndim=2] tmp2v):
	cdef int i
	cdef list tmp=[]
	try:
		tri=Delaunay(tmp2v)
	except:
		print('error in decomposition()')
		tmp=[0]
	else:
		for i in range(len(tri.simplices)):
			tet=tri.simplices[i]
			tmp.append([tet[0],tet[1],tet[2],tet[3]])
	return tmp
	
cdef np.ndarray outer_product(np.ndarray[np.int64_t, ndim=2] v1,np.ndarray[np.int64_t, ndim=2] v2):
	cdef long a1,a2,a3,b1,b2,b3,c1,c2,c3,c4,c5,c6,c7,c8,c9
	[a1,a2,a3]=mul(v1[1][0],v1[1][1],v1[1][2],v2[2][0],v2[2][1],v2[2][2])
	[b1,b2,b3]=mul(v1[2][0],v1[2][1],v1[2][2],v2[1][0],v2[1][1],v2[1][2])
	[c1,c2,c3]=sub(a1,a2,a3,b1,b2,b3)
	#
	[a1,a2,a3]=mul(v1[2][0],v1[2][1],v1[2][2],v2[0][0],v2[0][1],v2[0][2])
	[b1,b2,b3]=mul(v1[0][0],v1[0][1],v1[0][2],v2[2][0],v2[2][1],v2[2][2])
	[c4,c5,c6]=sub(a1,a2,a3,b1,b2,b3)
	#
	[a1,a2,a3]=mul(v1[0][0],v1[0][1],v1[0][2],v2[1][0],v2[1][1],v2[1][2])
	[b1,b2,b3]=mul(v1[1][0],v1[1][1],v1[1][2],v2[0][0],v2[0][1],v2[0][2])
	[c7,c8,c9]=sub(a1,a2,a3,b1,b2,b3)	
	#
	return np.array([[c1,c2,c3],[c4,c5,c6],[c7,c8,c9]],dtype=int)

cdef list inner_product(np.ndarray[np.int64_t, ndim=2] v1,np.ndarray[np.int64_t, ndim=2] v2):
	cdef long a1,a2,a3,b1,b2,b3,c1,c2,c3
	[a1,a2,a3]=mul(v1[0][0],v1[0][1],v1[0][2],v2[0][0],v2[0][1],v2[0][2])
	[b1,b2,b3]=mul(v1[1][0],v1[1][1],v1[1][2],v2[1][0],v2[1][1],v2[1][2])
	[c1,c2,c3]=mul(v1[2][0],v1[2][1],v1[2][2],v2[2][0],v2[2][1],v2[2][2])
	[a1,a2,a3]=add(a1,a2,a3,b1,b2,b3)
	[a1,a2,a3]=add(a1,a2,a3,c1,c2,c3)
	return [a1,a2,a3]

cdef int coplanar_check(np.ndarray[np.int64_t, ndim=3] point):
	# coplanar check
	cdef np.ndarray[np.int64_t, ndim=1] x0e,y0e,z0e,x0i,y0i,z0i
	cdef np.ndarray[np.int64_t, ndim=1] x1e,y1e,z1e,x1i,y1i,z1i
	cdef np.ndarray[np.int64_t, ndim=1] x2e,y2e,z2e,x2i,y2i,z2i
	cdef np.ndarray[np.int64_t, ndim=1] x3e,y3e,z3e,x3i,y3i,z3i
	cdef long a1,a2,a3,b1,b2,b3,c1,c2,c3,d1,d2,d3
	cdef int i1,flag
	cdef np.ndarray[np.int64_t, ndim=2] v1,v2,v3,v4
	if len(point)>3:
		x0e,y0e,z0e,x0i,y0i,z0i=projection(point[0][0],point[0][1],point[0][2],point[0][3],point[0][4],point[0][5])
		x1e,y1e,z1e,x1i,y1i,z1i=projection(point[1][0],point[1][1],point[1][2],point[1][3],point[1][4],point[1][5])
		x2e,y2e,z2e,x2i,y2i,z2i=projection(point[2][0],point[2][1],point[2][2],point[2][3],point[2][4],point[2][5])
		[a1,a2,a3]=sub(x1i[0],x1i[1],x1i[2],x0i[0],x0i[1],x0i[2]) # e1
		[b1,b2,b3]=sub(y1i[0],y1i[1],y1i[2],y0i[0],y0i[1],y0i[2])
		[c1,c2,c3]=sub(z1i[0],z1i[1],z1i[2],z0i[0],z0i[1],z0i[2])
		v1=np.array([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]])
		[a1,a2,a3]=sub(x2i[0],x2i[1],x2i[2],x0i[0],x0i[1],x0i[2]) # e2
		[b1,b2,b3]=sub(y2i[0],y2i[1],y2i[2],y0i[0],y0i[1],y0i[2])
		[c1,c2,c3]=sub(z2i[0],z2i[1],z2i[2],z0i[0],z0i[1],z0i[2])
		v2=np.array([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]])
		v3=outer_product(v1,v2)
		flag=0
		for i1 in range(3,len(point)):
			x3e,y3e,z3e,x3i,y3i,z3i=projection(point[i1][0],point[i1][1],point[i1][2],point[i1][3],point[i1][4],point[i1][5])
			[a1,a2,a3]=sub(x3i[0],x3i[1],x3i[2],x0i[0],x0i[1],x0i[2])
			[b1,b2,b3]=sub(y3i[0],y3i[1],y3i[2],y0i[0],y0i[1],y0i[2])
			[c1,c2,c3]=sub(z3i[0],z3i[1],z3i[2],z0i[0],z0i[1],z0i[2])
			v4=np.array([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]])
			[d1,d2,d3]=inner_product(v3,v4)
			if d1==0 and d2==0:
				flag+=0
			else:
				flag+=1
				break
		if flag==0:
			return 1 # coplanar
		else:
			return 0
	else:
		return 1 # coplanar

cpdef np.ndarray centroid(np.ndarray[np.int64_t, ndim=3] tetrahedron):
	#  geometric center, centroid of tetrahedron
	cdef int i1,i2
	cpdef long v1,v2,v3
	
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a
	tmp1a=np.array([0])
	for i2 in range(6):
		v1,v2,v3=0,0,1
		for i1 in range(4):
			v1,v2,v3=add(v1,v2,v3,tetrahedron[i1][i2][0],tetrahedron[i1][i2][1],tetrahedron[i1][i2][2])
		v1,v2,v3=mul(v1,v2,v3,1,0,4)
		if len(tmp1a)!=1:
			tmp1a=np.append(tmp1a,[v1,v2,v3])
		else:
			tmp1a=np.array([v1,v2,v3])
	return tmp1a.reshape(6,3)
"""
cpdef np.ndarray centroid_obj(np.ndarray[np.int64_t, ndim=4] obj):
	#  geometric center, centroid of OBJ
	cdef int i1,i2,i3
	cpdef long v1,v2,v3
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a
	tmp1a=np.array([0])
	for i3 in range(6):
		v1,v2,v3=0,0,1
		for i1 in range(len(obj)):
			for i2 in range(4):
				v1,v2,v3=add(v1,v2,v3,obj[i1][i2][i3][0],obj[i1][i2][i3][1],obj[i1][i2][i3][2])
			v1,v2,v3=mul(v1,v2,v3,1,0,4)
		v1,v2,v3=mul(v1,v2,v3,1,0,len(obj))
		if len(tmp1a)!=1:
			tmp1a=np.append(tmp1a,[v1,v2,v3])
		else:
			tmp1a=np.array([v1,v2,v3])
	return tmp1a.reshape(6,3)
"""
cpdef np.ndarray centroid_obj(np.ndarray[np.int64_t, ndim=4] obj):
	#  geometric center, centroid of OBJ
	cdef int i1,i2,i3,num
	cpdef long v1,v2,v3
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a
	tmp1a=np.array([0])
	num=len(obj)
	for i3 in range(6):
		v1,v2,v3=0,0,1
		for i1 in range(num):
			for i2 in range(4):
				v1,v2,v3=add(v1,v2,v3,obj[i1][i2][i3][0],obj[i1][i2][i3][1],obj[i1][i2][i3][2])
		v1,v2,v3=mul(v1,v2,v3,1,0,num*4)
		if i3!=0:
			tmp1a=np.append(tmp1a,[v1,v2,v3])
		else:
			tmp1a=np.array([v1,v2,v3])
	return tmp1a.reshape(6,3)

cdef list dot_product(np.ndarray[np.int64_t, ndim=1] a1,np.ndarray[np.int64_t, ndim=1] a2,np.ndarray[np.int64_t, ndim=1] a3,np.ndarray[np.int64_t, ndim=1] b1,np.ndarray[np.int64_t, ndim=1] b2,np.ndarray[np.int64_t, ndim=1] b3):
	# product of vectors A and B
	#
	# vector A
	# Ax=(a1[0]+a1[1]*tau)/a1[2]
	# Ay=(a2[0]+a2[1]*tau)/a2[2]
	# Az=(a3[0]+a3[1]*tau)/a3[2]
	#
	# vector B
	# Bx=(b1[0]+b1[1]*tau)/b1[2]
	# By=(b2[0]+b2[1]*tau)/b2[2]
	# Bz=(b3[0]+b3[1]*tau)/b3[2]	
	#	
	# return:
	# A*B = Ax*Bx + Ay*By + Az*Bz
	#	 = (t1+t2*TAU)/t3
	cdef long t1,t2,t3,t4,t5,t6
	[t1,t2,t3]=mul(a1[0],a1[1],a1[2],b1[0],b1[1],b1[2])
	[t4,t5,t6]=mul(a2[0],a2[1],a2[2],b2[0],b2[1],b2[2])
	[t1,t2,t3]=add(t1,t2,t3,t4,t5,t6)
	[t4,t5,t6]=mul(a3[0],a3[1],a3[2],b3[0],b3[1],b3[2])
	[t1,t2,t3]=add(t1,t2,t3,t4,t5,t6)
	return [t1,t2,t3]

cpdef list obj_volume_6d(np.ndarray[np.int64_t, ndim=4] obj):
	cdef int i
	cdef long v1,v2,v3,w1,w2,w3
	cdef np.ndarray[np.int64_t,ndim=3] tmp3
	w1,w2,w3=0,0,1
	for i in range(len(obj)):
		tmp3=obj[i]
		[v1,v2,v3]=tetrahedron_volume_6d(tmp3)
		w1,w2,w3=add(w1,w2,w3,v1,v2,v3)
	return [w1,w2,w3]

cpdef np.ndarray intersection_segment_surface(np.ndarray[np.int64_t, ndim=2] segment_1,np.ndarray[np.int64_t, ndim=2] segment_2,np.ndarray[np.int64_t, ndim=2] surface_1,np.ndarray[np.int64_t, ndim=2] surface_2,np.ndarray[np.int64_t, ndim=2] surface_3):
#def intersection_segment_surface(np.ndarray[np.int64_t, ndim=2] segment_1,np.ndarray[np.int64_t, ndim=2] segment_2,np.ndarray[np.int64_t, ndim=2] surface_1,np.ndarray[np.int64_t, ndim=2] surface_2,np.ndarray[np.int64_t, ndim=2] surface_3):
	#
	cdef np.ndarray[np.int64_t,ndim=1] tmp1,tmp1a
	cdef np.ndarray[np.int64_t,ndim=1] seg1e1,seg1e2,seg1e3,seg1i1,seg1i2,seg1i3
	cdef np.ndarray[np.int64_t,ndim=1] seg2e1,seg2e2,seg2e3,seg2i1,seg2i2,seg2i3
	cdef np.ndarray[np.int64_t,ndim=1] sur1e1,sur1e2,sur1e3,sur1i1,sur1i2,sur1i3
	cdef np.ndarray[np.int64_t,ndim=1] sur2e1,sur2e2,sur2e3,sur2i1,sur2i2,sur2i3
	cdef np.ndarray[np.int64_t,ndim=1] sur3e1,sur3e2,sur3e3,sur3i1,sur3i2,sur3i3
	cdef np.ndarray[np.int64_t,ndim=2] vec1,vecBA,vecCD,vecCE,vecCA
	cdef long bx1,bx2,bx3,by1,by2,by3,bz1,bz2,bz3
	cdef long cx1,cx2,cx3,cy1,cy2,cy3,cz1,cz2,cz3
	cdef long dx1,dx2,dx3,dy1,dy2,dy3,dz1,dz2,dz3
	cdef long ex1,ex2,ex3,ey1,ey2,ey3,ez1,ez2,ez3
	cdef long f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12
	cdef long g1,g2,g3,g4,g5,g6,g7,g8,g9
	cdef long h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16,h17,h18
	cdef long i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18
	cdef long j1,j2,j3,j4,j5,j6,j7,j8,j9,j10,j11,j12,j13,j14,j15,j16,j17,j18
	cdef double val1,val2,val3,val4
	cdef int verbose
	
	verbose=0
	
	if verbose>=1:
		print('              intersection_segment_surface()')
	else:
		pass
	#
	seg1e1,seg1e2,seg1e3,seg1i1,seg1i2,seg1i3=projection(segment_1[0],segment_1[1],segment_1[2],segment_1[3],segment_1[4],segment_1[5])
	seg2e1,seg2e2,seg2e3,seg2i1,seg2i2,seg2i3=projection(segment_2[0],segment_2[1],segment_2[2],segment_2[3],segment_2[4],segment_2[5])
	sur1e1,sur1e2,sur1e3,sur1i1,sur1i2,sur1i3=projection(surface_1[0],surface_1[1],surface_1[2],surface_1[3],surface_1[4],surface_1[5])
	sur2e1,sur2e2,sur2e3,sur2i1,sur2i2,sur2i3=projection(surface_2[0],surface_2[1],surface_2[2],surface_2[3],surface_2[4],surface_2[5])
	sur3e1,sur3e2,sur3e3,sur3i1,sur3i2,sur3i3=projection(surface_3[0],surface_3[1],surface_3[2],surface_3[3],surface_3[4],surface_3[5])
	#
	# Origin: seg1i1,seg1i2,seg1i3
	# line segment
	# segment line A-B: seg3i1,seg3i2,seg3i3
	#
	#ax1,ax2,ax3=seg1i1[0],seg1i1[1],seg1i1[2]
	#ay1,ay2,ay3=seg1i2[0],seg1i2[1],seg1i2[2]
	#az1,az2,az3=seg1i3[0],seg1i3[1],seg1i3[2]
	# AB
	#bx1,bx2,bx3=sub(seg2i1[0],seg2i1[1],seg2i1[2],seg1i1[0],seg1i1[1],seg1i1[2])
	#by1,by2,by3=sub(seg2i2[0],seg2i2[1],seg2i2[2],seg1i2[0],seg1i2[1],seg1i2[2])
	#bz1,bz2,bz3=sub(seg2i3[0],seg2i3[1],seg2i3[2],seg1i3[0],seg1i3[1],seg1i3[2])
	# BA
	[bx1,bx2,bx3]=sub(seg1i1[0],seg1i1[1],seg1i1[2],seg2i1[0],seg2i1[1],seg2i1[2])
	[by1,by2,by3]=sub(seg1i2[0],seg1i2[1],seg1i2[2],seg2i2[0],seg2i2[1],seg2i2[2])
	[bz1,bz2,bz3]=sub(seg1i3[0],seg1i3[1],seg1i3[2],seg2i3[0],seg2i3[1],seg2i3[2])
	# plane CDE
	# AC
	#cx1,cx2,cx3=sub(sur1i1[0],sur1i1[1],sur1i1[2],seg1i1[0],seg1i1[1],seg1i1[2])
	#cy1,cy2,cy3=sub(sur1i2[0],sur1i2[1],sur1i2[2],seg1i2[0],seg1i2[1],seg1i2[2])
	#cz1,cz2,cz3=sub(sur1i3[0],sur1i3[1],sur1i3[2],seg1i3[0],seg1i3[1],seg1i3[2])
	# CA
	[cx1,cx2,cx3]=sub(seg1i1[0],seg1i1[1],seg1i1[2],sur1i1[0],sur1i1[1],sur1i1[2])
	[cy1,cy2,cy3]=sub(seg1i2[0],seg1i2[1],seg1i2[2],sur1i2[0],sur1i2[1],sur1i2[2])
	[cz1,cz2,cz3]=sub(seg1i3[0],seg1i3[1],seg1i3[2],sur1i3[0],sur1i3[1],sur1i3[2])
	# CD
	[dx1,dx2,dx3]=sub(sur2i1[0],sur2i1[1],sur2i1[2],sur1i1[0],sur1i1[1],sur1i1[2])
	[dy1,dy2,dy3]=sub(sur2i2[0],sur2i2[1],sur2i2[2],sur1i2[0],sur1i2[1],sur1i2[2])
	[dz1,dz2,dz3]=sub(sur2i3[0],sur2i3[1],sur2i3[2],sur1i3[0],sur1i3[1],sur1i3[2])
	# CE
	[ex1,ex2,ex3]=sub(sur3i1[0],sur3i1[1],sur3i1[2],sur1i1[0],sur1i1[1],sur1i1[2])
	[ey1,ey2,ey3]=sub(sur3i2[0],sur3i2[1],sur3i2[2],sur1i2[0],sur1i2[1],sur1i2[2])
	[ez1,ez2,ez3]=sub(sur3i3[0],sur3i3[1],sur3i3[2],sur1i3[0],sur1i3[1],sur1i3[2])
	#
	#vecBA=np.array([[-bx1,-bx2, bx3],[-by1,-by2, by3],[-bz1,-bz2, bz3]]) # line segment BA = -AB
	vecBA=np.array([[ bx1, bx2, bx3],[ by1, by2, by3],[ bz1, bz2, bz3]]) # line segment BA
	vecCD=np.array([[ dx1, dx2, dx3],[ dy1, dy2, dy3],[ dz1, dz2, dz3]]) # edge segment of triangle CDE, CD
	vecCE=np.array([[ ex1, ex2, ex3],[ ey1, ey2, ey3],[ ez1, ez2, ez3]]) # edge segment of triangle CDE, CE
	#vecCA=np.array([[-cx1,-cx2, cx3],[-cy1,-cy2, cy3],[-cz1,-cz2, cz3]]) # CA = -AC
	vecCA=np.array([[ cx1, cx2, cx3],[ cy1, cy2, cy3],[ cz1, cz2, cz3]]) # CA
	#
	# below part consists of numerica calculations....
	#
	tmp1=np.array([0])
	f1,f2,f3=det_matrix(vecCD,vecCE,vecBA)
	val1=(f1+f2*TAU)/float(f3)
	if f1==0 and f2==0:
		if verbose>=2:
			print('   line segment and triangle are parallel')
		else:
			pass
		tmp1a=intersection_two_segment(segment_1,segment_2,surface_1,surface_2)
		if len(tmp1a)!=1:
			if len(tmp1)==1:
				tmp1=tmp1a
			else:
				tmp1=np.append(tmp1a,tmp1)
		else:
			pass
		tmp1a=intersection_two_segment(segment_1,segment_2,surface_1,surface_3)
		if len(tmp1a)!=1:
			if len(tmp1)==1:
				tmp1=tmp1a
			else:
				tmp1=np.append(tmp1a,tmp1)
		else:
			pass
		tmp1a=intersection_two_segment(segment_1,segment_2,surface_2,surface_3)
		if len(tmp1a)!=1:
			if len(tmp1)==1:
				tmp1=tmp1a
			else:
				tmp1=np.append(tmp1a,tmp1)
		else:
			pass
		if verbose>=2:
			print('   Intersectiong point:',tmp1)
		else:
			pass
		return tmp1
	else:
		f4,f5,f6=det_matrix(vecCA,vecCE,vecBA)
		val2=(f4+f5*TAU)/float(f6)
		#
		#   u = val2/val1:
		#  g4,g5,g6 = div(f4,f5,f6,f1,f2,f3)
		#
		#   v = val3/val1:
		#  g7,g8,g9 = div(f7,f8,f9,f1,f2,f3)
		#
		#   t = val4/val1:
		#  g1,g2,g3 = div(f10,f11,f12,f1,f2,f3)
		if val2/val1>=0.0 and val2/val1<=1.0:
			f7,f8,f9=det_matrix(vecCD,vecCA,vecBA)
			val3=(f7+f8*TAU)/float(f9)
			if val3/val1>=0.0 and (val2+val3)/val1<=1.0:
				f10,f11,f12=det_matrix(vecCD,vecCE,vecCA)
				val4=(f10+f11*TAU)/float(f12)
				if val4/val1>=0.0 and val4/val1<=1.0: # t = val4/val1
					g1,g2,g3=div(f10,f11,f12,f1,f2,f3) # t in TAU-style
					#
					#interval=line1a+t*(line1b-line1a)
					#
					# line1b-line1a
					[h1,h2,h3]=sub(segment_2[0][0],segment_2[0][1],segment_2[0][2],segment_1[0][0],segment_1[0][1],segment_1[0][2])
					[h4,h5,h6]=sub(segment_2[1][0],segment_2[1][1],segment_2[1][2],segment_1[1][0],segment_1[1][1],segment_1[1][2])
					[h7,h8,h9]=sub(segment_2[2][0],segment_2[2][1],segment_2[2][2],segment_1[2][0],segment_1[2][1],segment_1[2][2])
					[h10,h11,h12]=sub(segment_2[3][0],segment_2[3][1],segment_2[3][2],segment_1[3][0],segment_1[3][1],segment_1[3][2])
					[h13,h14,h15]=sub(segment_2[4][0],segment_2[4][1],segment_2[4][2],segment_1[4][0],segment_1[4][1],segment_1[4][2])
					[h16,h17,h18]=sub(segment_2[5][0],segment_2[5][1],segment_2[5][2],segment_1[5][0],segment_1[5][1],segment_1[5][2])
					#
					# line1a
					i1,i2,i3=segment_1[0][0],segment_1[0][1],segment_1[0][2]
					i4,i5,i6=segment_1[1][0],segment_1[1][1],segment_1[1][2]
					i7,i8,i9=segment_1[2][0],segment_1[2][1],segment_1[2][2]
					i10,i11,i12=segment_1[3][0],segment_1[3][1],segment_1[3][2]
					i13,i14,i15=segment_1[4][0],segment_1[4][1],segment_1[4][2]
					i16,i17,i18=segment_1[5][0],segment_1[5][1],segment_1[5][2]
					#
					[j1,j2,j3]=mul(g1,g2,g3,h1,h2,h3)
					[j1,j2,j3]=add(j1,j2,j3,i1,i2,i3)
					#
					[j4,j5,j6]=mul(g1,g2,g3,h4,h5,h6)
					[j4,j5,j6]=add(j4,j5,j6,i4,i5,i6)
					#
					[j7,j8,j9]=mul(g1,g2,g3,h7,h8,h9)
					[j7,j8,j9]=add(j7,j8,j9,i7,i8,i9)
					#
					[j10,j11,j12]=mul(g1,g2,g3,h10,h11,h12)
					[j10,j11,j12]=add(j10,j11,j12,i10,i11,i12)
					#
					[j13,j14,j15]=mul(g1,g2,g3,h13,h14,h15)
					[j13,j14,j15]=add(j13,j14,j15,i13,i14,i15)
					#
					[j16,j17,j18]=mul(g1,g2,g3,h16,h17,h18)
					[j16,j17,j18]=add(j16,j17,j18,i16,i17,i18)

					tmp1=np.array([j1,j2,j3,j4,j5,j6,j7,j8,j9,j10,j11,j12,j13,j14,j15,j16,j17,j18])
				else:
					pass
			else:
				pass
		else:
			pass
		return tmp1

cpdef np.ndarray remove_doubling_dim4(np.ndarray[np.int64_t, ndim=4] obj):
	cdef int i1,i2,j,counter,num
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b
	cdef np.ndarray[np.int64_t,ndim=2] tmp2a
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a,tmp4b	
	num=len(obj[0])
	tmp1a=np.append(obj[0][0],obj[0][1])
	for i1 in range(2,num):
		tmp1a=np.append(tmp1a,obj[0][i1])
	tmp3a=tmp1a.reshape(len(tmp1a)/18,6,3) # 18=6*3
	for i1 in range(1,len(obj)):
		for i2 in range(num):
			tmp2a=obj[i1][i2]
			counter=0
			for j in range(0,len(tmp3a)):
				if np.all(tmp2a==tmp3a[j]):
					counter+=1
					break
				else:
					counter+=0
			if counter==0:
				tmp1b=np.append(tmp3a,tmp2a)
				tmp3a=tmp1b.reshape(len(tmp1b)/18,6,3) # 18=6*3
			else:
				pass
	return tmp3a

cpdef np.ndarray remove_doubling_dim3(np.ndarray[np.int64_t, ndim=3] obj):
	cdef int i,j,counter
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a
	cdef np.ndarray[np.int64_t,ndim=2] tmp2a
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a
	tmp2a=obj[0]
	tmp3a=tmp2a.reshape(1,6,3)
	for i in range(1,len(obj)):
		tmp2a=obj[i]
		counter=0
		for j in range(0,len(tmp3a)):
			if np.all(tmp2a==tmp3a[j]):
				counter+=1
				break
			else:
				counter+=0
		if counter==0:
			tmp1a=np.append(tmp3a,tmp2a)
			tmp3a=tmp1a.reshape(len(tmp1a)/18,6,3) # 18=6*3
		else:
			pass
	return tmp3a

cpdef np.ndarray remove_doubling_dim4_in_perp_space(np.ndarray[np.int64_t, ndim=4] obj):
	# remove 6d coordinates which is doubled in perpendicular space
	cdef int num
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b
	num=len(obj[0])
	tmp3a=obj.reshape(len(obj)*num,6,3)
	tmp3b=remove_doubling_dim3_in_perp_space(tmp3a)
	return tmp3b

cpdef np.ndarray remove_doubling_dim3_in_perp_space(np.ndarray[np.int64_t, ndim=3] obj):
	# remove 6d coordinates which is doubled in perpendicular space
	cdef int i1,i2,counter1,counter2
	cdef np.ndarray[np.int64_t,ndim=1] v1,v2,v3,v4,v5,v6
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b
	cdef np.ndarray[np.int64_t,ndim=2] tmp2b
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b
	counter2=0
	if len(obj)>1:
		tmp2b=obj[0]
		v1,v2,v3,v4,v5,v6=projection(tmp2b[0],tmp2b[1],tmp2b[2],tmp2b[3],tmp2b[4],tmp2b[5])
		tmp1a=np.array([v4,v5,v6]).reshape(9)
		tmp1b=tmp2b.reshape(18)
		tmp3a=tmp1a.reshape(1,3,3) # perpendicular components
		tmp3b=tmp1b.reshape(1,6,3) # 6d
		for i1 in range(1,len(obj)):
			tmp2b=obj[i1]
			v1,v2,v3,v4,v5,v6=projection(tmp2b[0],tmp2b[1],tmp2b[2],tmp2b[3],tmp2b[4],tmp2b[5])
			counter1=0
			for i2 in range(len(tmp3a)):
				if np.all(v4==tmp3a[i2][0]) and np.all(v5==tmp3a[i2][1]) and np.all(v6==tmp3a[i2][2]):
					counter1+=1
					break
				else:
					counter1+=0
			if counter1==0:
				tmp1a=np.append(tmp1a,[v4,v5,v6])
				tmp1b=np.append(tmp1b,[tmp2b[0],tmp2b[1],tmp2b[2],tmp2b[3],tmp2b[4],tmp2b[5]])
			else:
				pass
			tmp3a=tmp1a.reshape(len(tmp1a)/9,3,3)
		return tmp1b.reshape(len(tmp1b)/18,6,3)
	else:
		return obj

cpdef int generator_xyz_dim4(np.ndarray[np.int64_t, ndim=4] obj,filename):
	cdef int i1
	cdef np.ndarray[np.int64_t,ndim=1] a1,a2,a3,a4,a5,a6
	cdef np.ndarray[np.int64_t,ndim=2] tmp2
	cdef np.ndarray[np.int64_t,ndim=3] tmp3
	f=open('%s'%(filename),'w')
	tmp3=remove_doubling_dim4(obj)
	f.write('%d\n'%(len(tmp3)))
	f.write('%s\n'%(filename))
	for i1 in range(len(tmp3)):
		a1,a2,a3,a4,a5,a6=projection(tmp3[i1][0],tmp3[i1][1],tmp3[i1][2],tmp3[i1][3],tmp3[i1][4],tmp3[i1][5])
		f.write('Xx %8.6f %8.6f %8.6f # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
		((a4[0]+a4[1]*TAU)/float(a4[2]),(a5[0]+a5[1]*TAU)/float(a5[2]),(a6[0]+a6[1]*TAU)/float(a6[2]),\
		tmp3[i1][0][0],tmp3[i1][0][1],tmp3[i1][0][2],\
		tmp3[i1][1][0],tmp3[i1][1][1],tmp3[i1][1][2],\
		tmp3[i1][2][0],tmp3[i1][2][1],tmp3[i1][2][2],\
		tmp3[i1][3][0],tmp3[i1][3][1],tmp3[i1][3][2],\
		tmp3[i1][4][0],tmp3[i1][4][1],tmp3[i1][4][2],\
		tmp3[i1][5][0],tmp3[i1][5][1],tmp3[i1][5][2]))
	f.closed
	return 0

cpdef int generator_xyz_dim4_triangle(np.ndarray[np.int64_t, ndim=4] obj,filename):
	cdef int i1,i2
	cdef np.ndarray[np.int64_t,ndim=1] a1,a2,a3,a4,a5,a6
	f=open('%s'%(filename),'w')
	f.write('%d\n'%(len(obj)*3))
	f.write('%s\n'%(filename))
	for i1 in range(len(obj)):
		for i2 in range(3):
			a1,a2,a3,a4,a5,a6=projection(obj[i1][i2][0],obj[i1][i2][1],obj[i1][i2][2],obj[i1][i2][3],obj[i1][i2][4],obj[i1][i2][5])
			f.write('Xx %8.6f %8.6f %8.6f # %3d-the tetrahedron %d-th vertex # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
			((a4[0]+a4[1]*TAU)/float(a4[2]),\
			(a5[0]+a5[1]*TAU)/float(a5[2]),\
			(a6[0]+a6[1]*TAU)/float(a6[2]),\
			i1,i2,\
			obj[i1][i2][0][0],obj[i1][i2][0][1],obj[i1][i2][0][2],\
			obj[i1][i2][1][0],obj[i1][i2][1][1],obj[i1][i2][1][2],\
			obj[i1][i2][2][0],obj[i1][i2][2][1],obj[i1][i2][2][2],\
			obj[i1][i2][3][0],obj[i1][i2][3][1],obj[i1][i2][3][2],\
			obj[i1][i2][4][0],obj[i1][i2][4][1],obj[i1][i2][4][2],\
			obj[i1][i2][5][0],obj[i1][i2][5][1],obj[i1][i2][5][2]))
	f.closed
	return 0

cpdef int generator_xyz_dim4_tetrahedron(np.ndarray[np.int64_t, ndim=4] obj,filename,int option):
	cdef int i1,i2
	cdef long w1,w2,w3
	cdef np.ndarray[np.int64_t,ndim=1] a1,a2,a3,a4,a5,a6
	
	if option==0:
		f=open('%s.xyz'%(filename),'w')
		f.write('%d\n'%(len(obj)*4))
		f.write('%s\n'%(filename))
		for i1 in range(len(obj)):
			for i2 in range(4):
				a1,a2,a3,a4,a5,a6=projection(obj[i1][i2][0],obj[i1][i2][1],obj[i1][i2][2],obj[i1][i2][3],obj[i1][i2][4],obj[i1][i2][5])
				f.write('Xx %8.6f %8.6f %8.6f # %3d-the tetrahedron %d-th vertex # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
				((a4[0]+a4[1]*TAU)/float(a4[2]),\
				(a5[0]+a5[1]*TAU)/float(a5[2]),\
				(a6[0]+a6[1]*TAU)/float(a6[2]),\
				i1,i2,\
				obj[i1][i2][0][0],obj[i1][i2][0][1],obj[i1][i2][0][2],\
				obj[i1][i2][1][0],obj[i1][i2][1][1],obj[i1][i2][1][2],\
				obj[i1][i2][2][0],obj[i1][i2][2][1],obj[i1][i2][2][2],\
				obj[i1][i2][3][0],obj[i1][i2][3][1],obj[i1][i2][3][2],\
				obj[i1][i2][4][0],obj[i1][i2][4][1],obj[i1][i2][4][2],\
				obj[i1][i2][5][0],obj[i1][i2][5][1],obj[i1][i2][5][2]))
		w1,w2,w3=obj_volume_6d(obj)
		f.write('volume = %d %d %d (%8.6f)\n'%(w1,w2,w3,(w1+TAU*w2)/float(w3)))
		for i1 in range(len(obj)):
			[v1,v2,v3]=tetrahedron_volume_6d(obj[i1])
			f.write('%3d-the tetrahedron, %d %d %d (%8.6f)\n'%(i1,v1,v2,v3,(v1+TAU*v2)/float(v3)))
	elif option==1:
		for i1 in range(len(obj)):
			f=open('%s_%d.xyz'%(filename,i1),'w')
			f.write('%d\n'%(4))
			f.write('%s_%d\n'%(filename,i1))
			for i2 in range(4):
				a1,a2,a3,a4,a5,a6=projection(obj[i1][i2][0],obj[i1][i2][1],obj[i1][i2][2],obj[i1][i2][3],obj[i1][i2][4],obj[i1][i2][5])
				f.write('Xx %8.6f %8.6f %8.6f # %3d-the tetrahedron %d-th vertex # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
				((a4[0]+a4[1]*TAU)/float(a4[2]),\
				(a5[0]+a5[1]*TAU)/float(a5[2]),\
				(a6[0]+a6[1]*TAU)/float(a6[2]),\
				i1,i2,\
				obj[i1][i2][0][0],obj[i1][i2][0][1],obj[i1][i2][0][2],\
				obj[i1][i2][1][0],obj[i1][i2][1][1],obj[i1][i2][1][2],\
				obj[i1][i2][2][0],obj[i1][i2][2][1],obj[i1][i2][2][2],\
				obj[i1][i2][3][0],obj[i1][i2][3][1],obj[i1][i2][3][2],\
				obj[i1][i2][4][0],obj[i1][i2][4][1],obj[i1][i2][4][2],\
				obj[i1][i2][5][0],obj[i1][i2][5][1],obj[i1][i2][5][2]))
				[v1,v2,v3]=tetrahedron_volume_6d(obj[i1])
			f.write('%3d-the tetrahedron, %d %d %d (%8.6f)\n'%(i1,v1,v2,v3,(v1+TAU*v2)/float(v3)))
	f.closed
	return 0

cpdef int generator_xyz_dim4_tetrahedron_select(np.ndarray[np.int64_t, ndim=4] obj,filename,int num):
	cdef int i2
	cdef long w1,w2,w3
	cdef np.ndarray[np.int64_t,ndim=1] a1,a2,a3,a4,a5,a6
	# option=0 # generate single .xyz file
	# option=1 # generate single .xyz file
	f=open('%s_%d.xyz'%(filename,num),'w')
	f.write('%d\n'%(4))
	f.write('%s_%d\n'%(filename,num))
	for i2 in range(4):
		a1,a2,a3,a4,a5,a6=projection(obj[num][i2][0],obj[num][i2][1],obj[num][i2][2],obj[num][i2][3],obj[num][i2][4],obj[num][i2][5])
		print('Xx %8.6f %8.6f %8.6f # %3d-the tetrahedron %d-th vertex # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
		((a4[0]+a4[1]*TAU)/float(a4[2]),\
		(a5[0]+a5[1]*TAU)/float(a5[2]),\
		(a6[0]+a6[1]*TAU)/float(a6[2]),\
		num,i2,\
		obj[num][i2][0][0],obj[num][i2][0][1],obj[num][i2][0][2],\
		obj[num][i2][1][0],obj[num][i2][1][1],obj[num][i2][1][2],\
		obj[num][i2][2][0],obj[num][i2][2][1],obj[num][i2][2][2],\
		obj[num][i2][3][0],obj[num][i2][3][1],obj[num][i2][3][2],\
		obj[num][i2][4][0],obj[num][i2][4][1],obj[num][i2][4][2],\
		obj[num][i2][5][0],obj[num][i2][5][1],obj[num][i2][5][2]))
		[v1,v2,v3]=tetrahedron_volume_6d(obj[num])
	f.write('%3d-the tetrahedron, %d %d %d (%8.6f)\n'%(num,v1,v2,v3,(v1+TAU*v2)/float(v3)))
	f.close
	return 0

cpdef int generator_xyz_dim3(np.ndarray[np.int64_t, ndim=3] obj,filename):
	cdef int i1
	cdef np.ndarray[np.int64_t,ndim=1] a1,a2,a3,a4,a5,a6
	cdef np.ndarray[np.int64_t,ndim=2] tmp2
	cdef np.ndarray[np.int64_t,ndim=3] tmp3
	f=open('%s'%(filename),'w')
	tmp3=remove_doubling_dim3(obj)
	f.write('%d\n'%(len(tmp3)))
	f.write('%s\n'%(filename))
	for i1 in range(len(tmp3)):
		a1,a2,a3,a4,a5,a6=projection(tmp3[i1][0],tmp3[i1][1],tmp3[i1][2],tmp3[i1][3],tmp3[i1][4],tmp3[i1][5])
		f.write('Xx %8.6f %8.6f %8.6f # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n'%\
		((a4[0]+a4[1]*TAU)/float(a4[2]),(a5[0]+a5[1]*TAU)/float(a5[2]),(a6[0]+a6[1]*TAU)/float(a6[2]),\
		tmp3[i1][0][0],tmp3[i1][0][1],tmp3[i1][0][2],\
		tmp3[i1][1][0],tmp3[i1][1][1],tmp3[i1][1][2],\
		tmp3[i1][2][0],tmp3[i1][2][1],tmp3[i1][2][2],\
		tmp3[i1][3][0],tmp3[i1][3][1],tmp3[i1][3][2],\
		tmp3[i1][4][0],tmp3[i1][4][1],tmp3[i1][4][2],\
		tmp3[i1][5][0],tmp3[i1][5][1],tmp3[i1][5][2]))
	return 0

cdef int on_out_surface(np.ndarray[np.int64_t, ndim=2] point,np.ndarray[np.int64_t, ndim=3] triangle):
	
	cdef np.ndarray[np.int64_t, ndim=1] m1,m2,m3,m4,m5,m6
	cdef np.ndarray[np.int64_t, ndim=2] p1,p2,p3
	cdef double volume0,volume1

	m1,m2,m3,m4,m5,m6=projection(point[0],point[1],point[2],point[3],point[4],point[5])
	p0=np.array([m4,m5,m6])
	m1,m2,m3,m4,m5,m6=projection(triangle[0][0],triangle[0][1],triangle[0][2],triangle[0][3],triangle[0][4],triangle[0][5])
	p1=np.array([m4,m5,m6])
	m1,m2,m3,m4,m5,m6=projection(triangle[1][0],triangle[1][1],triangle[1][2],triangle[1][3],triangle[1][4],triangle[1][5])
	p2=np.array([m4,m5,m6])
	m1,m2,m3,m4,m5,m6=projection(triangle[2][0],triangle[2][1],triangle[2][2],triangle[2][3],triangle[2][4],triangle[2][5])
	p3=np.array([m4,m5,m6])
	volume0=triangle_area(p1,p2,p3)
	volume1=triangle_area(p0,p2,p3)+triangle_area(p1,p0,p3)+triangle_area(p1,p2,p0)
	if abs(volume0-volume1)< EPS:
		return 0
	else:
		return 1

cdef double length_3d(np.ndarray[np.int64_t, ndim=2] vec):
	cdef long a1,b1,c1,a2,b2,c2
	cdef double val
	a1,b1,c1=mul(vec[0][0],vec[0][1],vec[0][2],vec[0][0],vec[0][1],vec[0][2])
	a2,b2,c2=mul(vec[1][0],vec[1][1],vec[1][2],vec[1][0],vec[1][1],vec[1][2])
	a1,b1,c1=add(a1,b1,c1,a2,b2,c2)
	a2,b2,c2=mul(vec[2][0],vec[2][1],vec[2][2],vec[2][0],vec[2][1],vec[2][2])
	a1,b1,c1=add(a1,b1,c1,a2,b2,c2)
	val=(a1+b1*TAU)/float(c1)
	return np.sqrt(val)

# Numerical version
cdef double inner_product_numerical(np.ndarray[np.float64_t, ndim=1] vector_1, np.ndarray[np.float64_t, ndim=1] vector_2):
	return vector_1[0]*vector_2[0]+vector_1[1]*vector_2[1]+vector_1[2]*vector_2[2]

cdef double length_numerical(np.ndarray[np.float64_t, ndim=1] vector):
	return np.sqrt(vector[0]**2+vector[1]**2+vector[2]**2)

# Numerical version
cdef int point_on_segment(np.ndarray[np.int64_t, ndim=2] point, np.ndarray[np.int64_t, ndim=2] lineA, np.ndarray[np.int64_t, ndim=2] lineB):
	# judge whether a point is on a line segment, A-B, or not.
	# http://marupeke296.com/COL_2D_No2_PointToLine.html
	cdef list tmp
	cdef double lPA,lBA,s
	cdef double x0,y0,z0,x1,y1,z1,x2,y2,z2
	cdef np.ndarray[np.float64_t, ndim=1] vecPA,vecBA
	#
	tmp=projection_numerical((point[0][0]+TAU*point[0][1])/float(point[0][2]),\
							(point[1][0]+TAU*point[1][1])/float(point[1][2]),\
							(point[2][0]+TAU*point[2][1])/float(point[2][2]),\
							(point[3][0]+TAU*point[3][1])/float(point[3][2]),\
							(point[4][0]+TAU*point[4][1])/float(point[4][2]),\
							(point[5][0]+TAU*point[5][1])/float(point[5][2]))
	x0,y0,z0=tmp[3],tmp[4],tmp[5]
	tmp=projection_numerical((lineA[0][0]+TAU*lineA[0][1])/float(lineA[0][2]),\
							(lineA[1][0]+TAU*lineA[1][1])/float(lineA[1][2]),\
							(lineA[2][0]+TAU*lineA[2][1])/float(lineA[2][2]),\
							(lineA[3][0]+TAU*lineA[3][1])/float(lineA[3][2]),\
							(lineA[4][0]+TAU*lineA[4][1])/float(lineA[4][2]),\
							(lineA[5][0]+TAU*lineA[5][1])/float(lineA[5][2]))
	x1,y1,z1=tmp[3],tmp[4],tmp[5]
	tmp=projection_numerical((lineB[0][0]+TAU*lineB[0][1])/float(lineB[0][2]),\
							(lineB[1][0]+TAU*lineB[1][1])/float(lineB[1][2]),\
							(lineB[2][0]+TAU*lineB[2][1])/float(lineB[2][2]),\
							(lineB[3][0]+TAU*lineB[3][1])/float(lineB[3][2]),\
							(lineB[4][0]+TAU*lineB[4][1])/float(lineB[4][2]),\
							(lineB[5][0]+TAU*lineB[5][1])/float(lineB[5][2]))
	x2,y2,z2=tmp[3],tmp[4],tmp[5]
	vecPA=np.array([x0-x1,y0-y1,z0-z1])
	vecBA=np.array([x2-x1,y2-y1,z2-z1])
	lPA=length_numerical(vecPA)
	lBA=length_numerical(vecBA)
	if lBA>0.0 and abs(inner_product_numerical(vecPA,vecBA)-lPA*lBA)<EPS:
		s=lPA/lBA
		if s>=0.0 and s<=1.0:
			return 0
		elif s>1.0:
			return 1 #       A==B P
		else:
			return -1 #    P A==B
	else:
		return 2

cdef double triangle_area(np.ndarray[np.int64_t, ndim=2] v1,\
							np.ndarray[np.int64_t, ndim=2] v2,\
							np.ndarray[np.int64_t, ndim=2] v3):
	cdef double vx0,vx1,vx2,vy0,vy1,vy2,vz0,vz1,vz2,area
	cdef np.ndarray[np.float64_t, ndim=1] vec1,vec2,vec3
	cdef np.ndarray[np.int64_t, ndim=1] x0,x1,x2,y0,y1,y2,z0,z1,z2
	x0=v1[0]
	y0=v1[1]
	z0=v1[2]
	#
	x1=v2[0]
	y1=v2[1]
	z1=v2[2]
	#
	x2=v3[0]
	y2=v3[1]
	z2=v3[2]
	#
	vx0=(x0[0]+x0[1]*TAU)/float(x0[2])
	vx1=(x1[0]+x1[1]*TAU)/float(x1[2])
	vx2=(x2[0]+x2[1]*TAU)/float(x2[2])
	vy0=(y0[0]+y0[1]*TAU)/float(y0[2])
	vy1=(y1[0]+y1[1]*TAU)/float(y1[2])
	vy2=(y2[0]+y2[1]*TAU)/float(y2[2])
	vz0=(z0[0]+z0[1]*TAU)/float(z0[2])
	vz1=(z1[0]+z1[1]*TAU)/float(z1[2])
	vz2=(z2[0]+z2[1]*TAU)/float(z2[2])
	vec1=np.array([vx1-vx0,vy1-vy0,vz1-vz0])
	vec2=np.array([vx2-vx0,vy2-vy0,vz2-vz0])
	
	vec3=np.cross(vec2,vec1) # cross product
	area=np.sqrt(vec3[0]**2+vec3[1]**2+vec3[2]**2)/2.0
	area=abs(area)
	return area
						
cdef list tetrahedron_volume(np.ndarray[np.int64_t, ndim=2] v1,\
							np.ndarray[np.int64_t, ndim=2] v2,\
							np.ndarray[np.int64_t, ndim=2] v3,\
							np.ndarray[np.int64_t, ndim=2] v4):
	# This function returns volume of a tetrahedron
	# input: vertex coordinates of the tetrahedron (x0,y0,z0),(x1,y1,z1),(x2,y2,z2),(x3,y3,z3)
	cdef long a1,a2,a3,b1,b2,b3,c1,c2,c3
	cdef np.ndarray[np.int64_t, ndim=1] x0,x1,x2,x3,y0,y1,y2,y3,z0,z1,z2,z3
	cdef np.ndarray[np.int64_t, ndim=2] a,b,c
	#
	x0=v1[0]
	y0=v1[1]
	z0=v1[2]
	#
	x1=v2[0]
	y1=v2[1]
	z1=v2[2]
	#
	x2=v3[0]
	y2=v3[1]
	z2=v3[2]
	#
	x3=v4[0]
	y3=v4[1]
	z3=v4[2]
	#
	[a1,a2,a3]=sub(x1[0],x1[1],x1[2],x0[0],x0[1],x0[2])
	[b1,b2,b3]=sub(y1[0],y1[1],y1[2],y0[0],y0[1],y0[2])
	[c1,c2,c3]=sub(z1[0],z1[1],z1[2],z0[0],z0[1],z0[2])
	a=np.array([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]])
	#
	[a1,a2,a3]=sub(x2[0],x2[1],x2[2],x0[0],x0[1],x0[2])
	[b1,b2,b3]=sub(y2[0],y2[1],y2[2],y0[0],y0[1],y0[2])
	[c1,c2,c3]=sub(z2[0],z2[1],z2[2],z0[0],z0[1],z0[2])
	b=np.array([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]])
	#
	[a1,a2,a3]=sub(x3[0],x3[1],x3[2],x0[0],x0[1],x0[2])
	[b1,b2,b3]=sub(y3[0],y3[1],y3[2],y0[0],y0[1],y0[2])
	[c1,c2,c3]=sub(z3[0],z3[1],z3[2],z0[0],z0[1],z0[2])
	c=np.array([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]])
	#
	[a1,a2,a3]=det_matrix(a,b,c) # determinant of 3x3 matrix
	#
	# avoid a negative value
	#
	if a1+a2*TAU<0.0: # to avoid negative volume...
		[a1,a2,a3]=mul(a1,a2,a3,-1,0,6)
	else:
		[a1,a2,a3]=mul(a1,a2,a3,1,0,6)
	return [a1,a2,a3]

cdef list det_matrix(np.ndarray[np.int64_t, ndim=2] a, np.ndarray[np.int64_t, ndim=2] b, np.ndarray[np.int64_t, ndim=2] c):
	# determinant of 3x3 matrix in TAU style
	cdef long t1,t2,t3,t4,t5,t6,t7,t8,t9
	#
	[t1,t2,t3]=mul(a[0][0],a[0][1],a[0][2],b[1][0],b[1][1],b[1][2])
	[t1,t2,t3]=mul(t1,t2,t3,c[2][0],c[2][1],c[2][2])
	#		
	[t4,t5,t6]=mul(a[2][0],a[2][1],a[2][2],b[0][0],b[0][1],b[0][2])
	[t4,t5,t6]=mul(t4,t5,t6,c[1][0],c[1][1],c[1][2])
	#
	[t7,t8,t9]=add(t1,t2,t3,t4,t5,t6)
	#
	[t1,t2,t3]=mul(a[1][0],a[1][1],a[1][2],b[2][0],b[2][1],b[2][2])
	[t1,t2,t3]=mul(t1,t2,t3,c[0][0],c[0][1],c[0][2])
	#
	[t7,t8,t9]=add(t7,t8,t9,t1,t2,t3)
	#
	[t1,t2,t3]=mul(a[2][0],a[2][1],a[2][2],b[1][0],b[1][1],b[1][2])
	[t1,t2,t3]=mul(t1,t2,t3,c[0][0],c[0][1],c[0][2])
	#
	[t7,t8,t9]=sub(t7,t8,t9,t1,t2,t3)
	#
	[t1,t2,t3]=mul(a[1][0],a[1][1],a[1][2],b[0][0],b[0][1],b[0][2])
	[t1,t2,t3]=mul(t1,t2,t3,c[2][0],c[2][1],c[2][2])
	#
	[t7,t8,t9]=sub(t7,t8,t9,t1,t2,t3)
	#
	[t1,t2,t3]=mul(a[0][0],a[0][1],a[0][2],b[2][0],b[2][1],b[2][2])
	[t1,t2,t3]=mul(t1,t2,t3,c[1][0],c[1][1],c[1][2])

	[t7,t8,t9]=sub(t7,t8,t9,t1,t2,t3)
	#
	return [t7,t8,t9]

cdef double tetrahedron_volume_numerical(double x0, double y0, double z0,\
										double x1, double y1, double z1,\
										double x2, double y2, double z2,\
										double x3, double y3, double z3):
	# This function returns volume of a tetrahedron
	# parameters: vertex coordinates of the tetrahedron, (x0,y0,z0),(x1,y1,z1),(x2,y2,z2),(x3,y3,z3)
	cdef double detm,vol
	cdef np.ndarray[np.float64_t, ndim=2] m
	m = np.array([[x1-x0,y1-y0,z1-z0],\
				  [x2-x0,y2-y0,z2-z0],\
				  [x3-x0,y3-y0,z3-z0]])
	detm = np.linalg.det(m)
	vol = abs(detm)/6.0
	return vol

# numerical version
cdef int inside_outside_tetrahedron(np.ndarray[np.int64_t, ndim=2] point,\
									np.ndarray[np.int64_t, ndim=2] tetrahedron_v1,\
									np.ndarray[np.int64_t, ndim=2] tetrahedron_v2,\
									np.ndarray[np.int64_t, ndim=2] tetrahedron_v3,\
									np.ndarray[np.int64_t, ndim=2] tetrahedron_v4):
	# this function judges whether the point is inside a traiangle or not
	# input:
	# (1) vertex coordinates of the triangle,xyz0, xyz1, xyz2, xyz3
	# (2) coordinate of the point,xyz4
	cdef np.ndarray[np.int64_t, ndim=1] m1,m2,m3,m4,m5,m6
	cdef double volume0,volume1,volume2,volume3,volume4,volume_sum
	cdef double x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4
	#
	tetrahedron0=np.append(tetrahedron_v1,tetrahedron_v2)
	tetrahedron0=np.append(tetrahedron0,tetrahedron_v3)
	tetrahedron0=np.append(tetrahedron0,tetrahedron_v4)
	tetrahedron0=tetrahedron0.reshape(4,6,3)
	volume0=tetrahedron_volume_6d_numerical(tetrahedron0)
	#
	tetrahedron1=np.append(point,tetrahedron_v2)
	tetrahedron1=np.append(tetrahedron1,tetrahedron_v3)
	tetrahedron1=np.append(tetrahedron1,tetrahedron_v4)
	tetrahedron1=tetrahedron1.reshape(4,6,3)
	volume1=tetrahedron_volume_6d_numerical(tetrahedron1)
	#
	tetrahedron2=np.append(tetrahedron_v1,point)
	tetrahedron2=np.append(tetrahedron2,tetrahedron_v3)
	tetrahedron2=np.append(tetrahedron2,tetrahedron_v4)
	tetrahedron2=tetrahedron2.reshape(4,6,3)
	volume2=tetrahedron_volume_6d_numerical(tetrahedron2)
	#
	tetrahedron3=np.append(tetrahedron_v1,tetrahedron_v2)
	tetrahedron3=np.append(tetrahedron3,point)
	tetrahedron3=np.append(tetrahedron3,tetrahedron_v4)
	tetrahedron3=tetrahedron3.reshape(4,6,3)
	volume3=tetrahedron_volume_6d_numerical(tetrahedron3)
	#
	tetrahedron4=np.append(tetrahedron_v1,tetrahedron_v2)
	tetrahedron4=np.append(tetrahedron4,tetrahedron_v3)
	tetrahedron4=np.append(tetrahedron4,point)
	tetrahedron4=tetrahedron4.reshape(4,6,3)
	volume4=tetrahedron_volume_6d_numerical(tetrahedron4)
	#
	if abs(volume0-volume1-volume2-volume3-volume4)<EPS*volume0:
		return 0 # inside
	else:
		return 1 # outside

cdef np.ndarray matrix_dot(np.ndarray array_1, np.ndarray array_2):
	cdef Py_ssize_t mx1,my1,mx2,my2
	cdef Py_ssize_t x,y,z	
	mx1 = array_1.shape[0]
	my1 = array_1.shape[1]
	mx2 = array_2.shape[0]
	my2 = array_2.shape[1]
	array_3 = np.zeros((mx1,my2), dtype=int)
	for x in range(array_1.shape[0]):
		for y in range(array_2.shape[1]):
			for z in range(array_1.shape[1]):
				array_3[x][y] += array_1[x][z] * array_2[z][y]
	return array_3

cdef np.ndarray matrix_pow(np.ndarray array_1, int n):
	cdef Py_ssize_t mx
	cdef Py_ssize_t my
	cdef Py_ssize_t x,y,z
	cdef int i
	mx = array_1.shape[0]
	my = array_1.shape[1]
	array_2 = np.identity(mx, dtype=int)
	if mx == my:
		if n == 0:
			return np.identity(mx, dtype=int)
		elif n<0:
			return np.zeros((mx,mx), dtype=int)
		else:
			for i in range(n):
				tmp = np.zeros((6, 6), dtype=int)
				for x in range(array_2.shape[0]):
					for y in range(array_1.shape[1]):
						for z in range(array_2.shape[1]):
							tmp[x][y] += array_2[x][z] * array_1[z][y]
				array_2 = tmp
			return array_2
	else:
		print('ERROR: matrix has not regular shape')
		return 

cdef icosasymop():
	# icosahedral symmetry operations
	cdef int i,j,k,l,m
	cdef np.ndarray[np.int64_t, ndim=2] s1,s2,s3,s4,s5,m1,m2,m3,m4,m5
	cdef list symop
	m1=np.array([[ 1, 0, 0, 0, 0, 0],\
				[ 0, 0, 1, 0, 0, 0],\
				[ 0, 0, 0, 1, 0, 0],\
				[ 0, 0, 0, 0, 1, 0],\
				[ 0, 0, 0, 0, 0, 1],\
				[ 0, 1, 0, 0, 0, 0]])
	# mirror
	m2=np.array([[-1, 0, 0, 0, 0, 0],\
				[ 0,-1, 0, 0, 0, 0],\
				[ 0, 0, 0, 0, 0,-1],\
				[ 0, 0, 0, 0,-1, 0],\
				[ 0, 0, 0,-1, 0, 0],\
				[ 0, 0,-1, 0, 0, 0]])
	# c2
	m3=np.array([[ 0, 0, 0, 0, 0,-1],\
				[ 0,-1, 0, 0, 0, 0],\
				[ 0, 0, 0, 1, 0, 0],\
				[ 0, 0, 1, 0, 0, 0],\
				[ 0, 0, 0, 0,-1, 0],\
				[-1, 0, 0, 0, 0, 0]])
	# c3
	m4=np.array([[ 0, 1, 0, 0, 0, 0],\
				[ 0, 0, 1, 0, 0, 0],\
				[ 1, 0, 0, 0, 0, 0],\
				[ 0, 0, 0, 0, 0, 1],\
				[ 0, 0, 0,-1, 0, 0],\
				[ 0, 0, 0, 0,-1, 0]])
	# inversion
	m5=np.array([[-1, 0, 0, 0, 0, 0],\
				[ 0,-1, 0, 0, 0, 0],\
				[ 0, 0,-1, 0, 0, 0],\
				[ 0, 0, 0,-1, 0, 0],\
				[ 0, 0, 0, 0,-1, 0],\
				[ 0, 0, 0, 0, 0,-1]])
	symop=[]
	for m in range(2):
		for l in range(3):
			for k in range(2):
				for j in range(2):
					for i in range(5):
						s1=matrix_pow(m1,i) # c5
						s2=matrix_pow(m2,j) # mirror
						s3=matrix_pow(m3,k) # c2
						s4=matrix_pow(m4,l) # c3
						s5=matrix_pow(m5,m) # inversion
						tmp=matrix_dot(s5,s4)
						tmp=matrix_dot(tmp,s3)
						tmp=matrix_dot(tmp,s2)
						tmp=matrix_dot(tmp,s1)
						symop.append(tmp)
	return symop

cpdef list projection(np.ndarray[np.int64_t, ndim=1] h1,\
					np.ndarray[np.int64_t, ndim=1] h2,\
					np.ndarray[np.int64_t, ndim=1] h3,\
					np.ndarray[np.int64_t, ndim=1] h4,\
					np.ndarray[np.int64_t, ndim=1] h5,\
					np.ndarray[np.int64_t, ndim=1] h6):
	# projection of a 6d vector onto Epar and Eperp, using "TAU-style"
	#
	# NOTE: coefficient (alpha) of the projection matrix is set to be 1.
	# alpha = a/np.sqrt(2.0+TAU)
	# see Yamamoto ActaCrystal (1997)
	cdef np.ndarray[np.int64_t, ndim=1] m1,m2,m3,m4
	cdef np.ndarray[np.int64_t, ndim=1] v1e,v2e,v3e,v1i,v2i,v3i
	m0=np.array([ 0, 0, 1]) #  0 in 'TAU-style'
	m1=np.array([ 1, 0, 1]) #  1
	m2=np.array([-1, 0, 1]) # -1
	m3=np.array([ 0, 1, 1]) #  tau
	m4=np.array([ 0,-1, 1]) # -tau
	v1e=mtrixcal(m1,m3,m3,m0,m2,m0,h1,h2,h3,h4,h5,h6) # 1,tau,tau,0,-1,0
	v2e=mtrixcal(m3,m0,m0,m1,m3,m1,h1,h2,h3,h4,h5,h6) # tau,0,0,1,TAU,1
	v3e=mtrixcal(m0,m1,m2,m4,m0,m3,h1,h2,h3,h4,h5,h6) # 0,1,-1,-tau,0,tau
	v1i=mtrixcal(m3,m2,m2,m0,m4,m0,h1,h2,h3,h4,h5,h6) # tau,-1,-1,0,-tau,0
	v2i=mtrixcal(m2,m0,m0,m3,m2,m3,h1,h2,h3,h4,h5,h6) # -1,0,0,tau,-1,tau
	v3i=mtrixcal(m0,m3,m4,m1,m0,m2,h1,h2,h3,h4,h5,h6) # 0,tau,-tau,1,0,-1
	return [v1e,v2e,v3e,v1i,v2i,v3i]

cdef np.ndarray mtrixcal(np.ndarray[np.int64_t, ndim=1] v1,
						np.ndarray[np.int64_t, ndim=1] v2,
						np.ndarray[np.int64_t, ndim=1] v3,
						np.ndarray[np.int64_t, ndim=1] v4,
						np.ndarray[np.int64_t, ndim=1] v5,
						np.ndarray[np.int64_t, ndim=1] v6,
						np.ndarray[np.int64_t, ndim=1] n1,
						np.ndarray[np.int64_t, ndim=1] n2,
						np.ndarray[np.int64_t, ndim=1] n3,
						np.ndarray[np.int64_t, ndim=1] n4,
						np.ndarray[np.int64_t, ndim=1] n5,
						np.ndarray[np.int64_t, ndim=1] n6):
	cdef long a1,a2,a3,a4,a5,a6
	cdef np.ndarray[np.int64_t,ndim=1] val
	
	[a1,a2,a3]=mul(v1[0],v1[1],v1[2],n1[0],n1[1],n1[2])
	[a4,a5,a6]=mul(v2[0],v2[1],v2[2],n2[0],n2[1],n2[2])
	[a1,a2,a3]=add(a1,a2,a3,a4,a5,a6)
	[a4,a5,a6]=mul(v3[0],v3[1],v3[2],n3[0],n3[1],n3[2])
	[a1,a2,a3]=add(a1,a2,a3,a4,a5,a6)
	[a4,a5,a6]=mul(v4[0],v4[1],v4[2],n4[0],n4[1],n4[2])
	[a1,a2,a3]=add(a1,a2,a3,a4,a5,a6)
	[a4,a5,a6]=mul(v5[0],v5[1],v5[2],n5[0],n5[1],n5[2])
	[a1,a2,a3]=add(a1,a2,a3,a4,a5,a6)
	[a4,a5,a6]=mul(v6[0],v6[1],v6[2],n6[0],n6[1],n6[2])
	[a1,a2,a3]=add(a1,a2,a3,a4,a5,a6)
	val=np.array([a1,a2,a3])
	return val

cdef list projection_perp(np.ndarray[np.int64_t, ndim=1] n1,
						np.ndarray[np.int64_t, ndim=1] n2,
						np.ndarray[np.int64_t, ndim=1] n3,
						np.ndarray[np.int64_t, ndim=1] n4,
						np.ndarray[np.int64_t, ndim=1] n5,
						np.ndarray[np.int64_t, ndim=1] n6):
	# This returns 6D indeces of a projection of 6D vector (n1,n2,n3,n4,n5,n6) onto Eperp
	# Direct lattice vector
	cdef np.ndarray[np.int64_t, ndim=1] m1,m2,m3
	cdef np.ndarray[np.int64_t, ndim=1] h1,h2,h3,h4,h5,h6
	m1=np.array([ 1, 0, 2]) #  (TAU+2)/2/(2+TAU)=1/2 in 'TAU-style'
	m2=np.array([-1, 2,10]) #  TAU/2/(2+TAU)
	m3=np.array([ 1,-2,10]) # -TAU/2/(2+TAU)
	h1=mtrixcal(m1,m3,m3,m3,m3,m3,n1,n2,n3,n4,n5,n6) # (tau+2,-tau,-tau,-tau,-tau,-tau)/2
	h2=mtrixcal(m3,m1,m3,m2,m2,m3,n1,n2,n3,n4,n5,n6) # (-tau,tau+2,-tau,tau,tau,-tau)/2
	h3=mtrixcal(m3,m3,m1,m3,m2,m2,n1,n2,n3,n4,n5,n6) # (-tau,-tau,tau+2,-tau,tau,tau)/2
	h4=mtrixcal(m3,m2,m3,m1,m3,m2,n1,n2,n3,n4,n5,n6) # (-tau,tau,-tau,tau+2,-tau,tau)/2
	h5=mtrixcal(m3,m2,m2,m3,m1,m3,n1,n2,n3,n4,n5,n6) # (-tau,tau,tau,-tau,tau+2,-tau)/2
	h6=mtrixcal(m3,m3,m2,m2,m3,m1,n1,n2,n3,n4,n5,n6) # (-tau,-tau,tau,tau,-tau,tau+2)/2
	return [h1,h2,h3,h4,h5,h6]
	#const=1/(2.0+TAU)
	#m1 =((TAU+2)*n1 -     TAU*n2 -     TAU*n3 -     TAU*n4 -     TAU*n5 -     TAU*n6)/2.0*const
	#m2 = ( - TAU*n1 + (TAU+2)*n2 -     TAU*n3 +     TAU*n4 +     TAU*n5 -     TAU*n6)/2.0*const
	#m3 = ( - TAU*n1 -     TAU*n2 + (TAU+2)*n3 -     TAU*n4 +     TAU*n5 +     TAU*n6)/2.0*const
	#m4 = ( - TAU*n1 +     TAU*n2 -     TAU*n3 + (TAU+2)*n4 -     TAU*n5 +     TAU*n6)/2.0*const
	#m5 = ( - TAU*n1 +     TAU*n2 +     TAU*n3 -     TAU*n4 + (TAU+2)*n5 -     TAU*n6)/2.0*const
	#m6 = ( - TAU*n1 -     TAU*n2 +     TAU*n3 +     TAU*n4 -     TAU*n5 + (TAU+2)*n6)/2.0*const
	#return m1,m2,m3,m4,m5,m6

cdef list add(long p1,long p2,long p3,long q1,long q2,long q3): # A+B
	cdef long c1,c2,c3,gcd
	cdef np.ndarray[np.int64_t,ndim=1] x
	c1=p1*q3+q1*p3
	c2=p2*q3+q2*p3
	c3=p3*q3
	x=np.array([c1,c2,c3])
	gcd=np.gcd.reduce(x) # c1,c2,c3の最大公約数
	if c3/gcd<0:
		return [-c1/gcd,-c2/gcd,-c3/gcd]
	else:
		return [c1/gcd,c2/gcd,c3/gcd]

cdef list sub(long p1,long p2,long p3,long q1,long q2,long q3): # A-B
	cdef long c1,c2,c3,d1,d2,d3,gcd
	cdef np.ndarray[np.int64_t,ndim=1] x
	c1=p1*q3-q1*p3
	c2=p2*q3-q2*p3
	c3=p3*q3
	x=np.array([c1,c2,c3])
	gcd=np.gcd.reduce(x) # c1,c2,c3の最大公約数
	if c3/gcd<0:
		return [-c1/gcd,-c2/gcd,-c3/gcd]
	else:
		return [c1/gcd,c2/gcd,c3/gcd]

cdef list mul(long p1,long p2,long p3,long q1,long q2,long q3): # A*B
	cdef long c1,c2,c3,d1,d2,d3,gcd
	cdef np.ndarray[np.int64_t,ndim=1] x
	c1=p1*q1+p2*q2
	c2=p1*q2+p2*q1+p2*q2
	c3=p3*q3
	x=np.array([c1,c2,c3])
	gcd=np.gcd.reduce(x) # c1,c2,c3の最大公約数
	if c3/gcd<0:
		return [-c1/gcd,-c2/gcd,-c3/gcd]
	else:
		return [c1/gcd,c2/gcd,c3/gcd]

cdef list div(long p1,long p2,long p3,long q1,long q2,long q3): # A/B
	cdef long gcd,c1,c2,c3
	cdef np.ndarray[np.int64_t,ndim=1] x
	if q1==0 and q2==0:
		print('ERROR_1:division error')
		return 1
	else:
		if p1==0 and p2==0:
			return [0,0,1]
		else:
			if q2!=0:
				if q1!=0:
					c1=(p1*q1 + p1*q2 - p2*q2)*q3
					c2=(p2*q1 - p1*q2)*q3
					c3=(q1*q1 - q2*q2 + q1*q2)*p3
				else:
					c1=(-p1+p2)*q3
					c2=p1*q3
					c3=q2*p3
			elif q1!=0:
				c1=p1*q3
				c2=p2*q3
				c3=q1*p3
			x=np.array([c1,c2,c3])
			gcd=np.gcd.reduce(x) # c1,c2,c3の最大公約数
			if gcd!=0:
				if c3/gcd<0:
					return [-c1/gcd,-c2/gcd,-c3/gcd]
				else:
					return [c1/gcd,c2/gcd,c3/gcd]
			else:
				print('ERROR_2:division error',c1,c2,c3,p1,p2,p3,q1,q2,q3)
				return 1

cpdef list symop_vec(np.ndarray[np.int64_t,ndim=2] symop,np.ndarray[np.int64_t,ndim=2] vec,np.ndarray[np.int64_t, ndim=2] centre):
	cdef long j,k,a1,a2,a3,b1,b2,b3
	cdef list vec1
	vec1=[]
	for k in range(6):
		b1,b2,b3=0,0,1
		for j in range(6):
			a1=vec[j][0]*symop[k][j]
			a2=vec[j][1]*symop[k][j]
			a3=vec[j][2]
			#print a1,a2,a3,b1,b2,b3
			[b1,b2,b3]=add(b1,b2,b3,a1,a2,a3)
		[b1,b2,b3]=add(b1,b2,b3,centre[k][0],centre[k][1],centre[k][2])
		vec1.extend([b1,b2,b3])	
	return vec1

cpdef list symop_obj(np.ndarray[np.int64_t,ndim=2] symop,np.ndarray[np.int64_t, ndim=3] obj,np.ndarray[np.int64_t, ndim=2] centre):
	cdef int i
	cdef list obj1
	obj1=[]
	for i in range(len(obj)):
		obj1.extend(symop_vec(symop,obj[i],centre))
	return obj1

cpdef np.ndarray generator_obj_symmetric_surface(np.ndarray[np.int64_t, ndim=3] obj,np.ndarray[np.int64_t, ndim=2] centre):
	cdef int i
	cdef list od,mop
	cdef np.ndarray[np.int64_t,ndim=4] tmp4
	od=[]
	mop=icosasymop()
	for i in range(len(mop)):
		od.extend(symop_obj(mop[i],obj,centre))
	tmp4=np.array(od).reshape(len(od)/54,3,6,3) # 54=3*6*3
	print(' Number of triangles on POD surface: %d'%(len(tmp4)))
	return tmp4	

cpdef np.ndarray generator_obj_symmetric_tetrahedron(np.ndarray[np.int64_t, ndim=3] obj,np.ndarray[np.int64_t, ndim=2] centre):
	cdef int i
	cdef list od,mop
	cdef np.ndarray[np.int64_t,ndim=4] tmp4
	print('Generating symmetric POD')
	od=[]
	mop=icosasymop()
	for i in range(len(mop)):
		od.extend(symop_obj(mop[i],obj,centre))
	tmp4=np.array(od).reshape(len(od)/72,4,6,3) # 72=4*6*3
	print(' Number of tetrahedron: %d'%(len(tmp4)))
	return tmp4	

cpdef np.ndarray generator_obj_symmetric_tetrahedron_specific_symop(np.ndarray[np.int64_t, ndim=3] obj,np.ndarray[np.int64_t, ndim=2] centre,list symmetry_operation):
	# using specific symmetry operations
	cdef int i
	cdef list od
	cdef np.ndarray[np.int64_t,ndim=4] tmp4
	print('Generating symmetric POD')
	od=[]
	for i in range(len(symmetry_operation)):
		od.extend(symop_obj(symmetry_operation[i],obj,centre))
	tmp4=np.array(od).reshape(len(od)/72,4,6,3) # 72=4*6*3
	print(' Number of tetrahedron: %d'%(len(tmp4)))
	return tmp4	

cdef list tetrahedron_volume_6d(np.ndarray[np.int64_t, ndim=3] tetrahedron):
	cdef long v1,v2,v3
	cdef np.ndarray[np.int64_t,ndim=1] x1e,y1e,z1e,x1i,y1i,z1i,x2i,y2i,z2i,x3i,y3i,z3i,x4i,y4i,z4i
	x1e,y1e,z1e,x1i,y1i,z1i=projection(tetrahedron[0][0],tetrahedron[0][1],tetrahedron[0][2],tetrahedron[0][3],tetrahedron[0][4],tetrahedron[0][5])
	x1e,y1e,z1e,x2i,y2i,z2i=projection(tetrahedron[1][0],tetrahedron[1][1],tetrahedron[1][2],tetrahedron[1][3],tetrahedron[1][4],tetrahedron[1][5])
	x1e,y1e,z1e,x3i,y3i,z3i=projection(tetrahedron[2][0],tetrahedron[2][1],tetrahedron[2][2],tetrahedron[2][3],tetrahedron[2][4],tetrahedron[2][5])
	x1e,y1e,z1e,x4i,y4i,z4i=projection(tetrahedron[3][0],tetrahedron[3][1],tetrahedron[3][2],tetrahedron[3][3],tetrahedron[3][4],tetrahedron[3][5])
	[v1,v2,v3]=tetrahedron_volume(np.array([x1i,y1i,z1i]),np.array([x2i,y2i,z2i]),np.array([x3i,y3i,z3i]),np.array([x4i,y4i,z4i]))
	return [v1,v2,v3]

###########################
#	 Numerical Calc	  #
###########################

cpdef double obj_volume_6d_numerical(np.ndarray[np.int64_t, ndim=4] obj):
	cdef int i
	cdef double volume,vol
	volume=0.0
	for i in range(len(obj)):
		vol=tetrahedron_volume_6d_numerical(obj[i])
		volume+=vol	
	return volume

cpdef double tetrahedron_volume_6d_numerical(np.ndarray[np.int64_t, ndim=3] tetrahedron):
	cdef double x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3,z3
	cdef np.ndarray[np.int64_t,ndim=1] x1e,y1e,z1e,x1i,y1i,z1i,x2i,y2i,z2i,x3i,y3i,z3i,x4i,y4i,z4i
	x0,y0,z0=get_internal_component_numerical(tetrahedron[0])
	x1,y1,z1=get_internal_component_numerical(tetrahedron[1])
	x2,y2,z2=get_internal_component_numerical(tetrahedron[2])
	x3,y3,z3=get_internal_component_numerical(tetrahedron[3])
	return tetrahedron_volume_numerical(x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3,z3)

cpdef list get_internal_component_numerical(np.ndarray[np.int64_t, ndim=2] vec6d):
	cdef double n1,n2,n3,n4,n5,n6
	cdef double v1,v2,v3,v4,v5,v6
	#cdef np.ndarray[np.float64_t, ndim=1] a1,a2,a3,a4,a5,a6
	#print 'a=',vec6d[0][2]
	#print 'b=',float(vec6d[0][2])
	n1=(vec6d[0][0]+TAU*vec6d[0][1])/float(vec6d[0][2])
	#print 'n1=',n1
	n2=(vec6d[1][0]+TAU*vec6d[1][1])/float(vec6d[1][2])
	n3=(vec6d[2][0]+TAU*vec6d[2][1])/float(vec6d[2][2])
	n4=(vec6d[3][0]+TAU*vec6d[3][1])/float(vec6d[3][2])
	n5=(vec6d[4][0]+TAU*vec6d[4][1])/float(vec6d[4][2])
	n6=(vec6d[5][0]+TAU*vec6d[5][1])/float(vec6d[5][2])
	v1,v2,v3,v4,v5,v6=projection_numerical(n1,n2,n3,n4,n5,n6)
	return [v4,v5,v6]

cpdef list projection_numerical(double n1, double n2, double n3, double n4, double n5, double n6):
	#	parallel and perpendicular components of a 6D lattice vector in direct space.
	cdef double const,v1,v2,v3,v4,v5,v6
	cdef np.ndarray[np.float64_t, ndim=2] n,m1,v
	
	n = np.array([[n1],[n2],[n3],[n4],[n5],[n6]])
	#const = lattice_a
	#const = 1.0/np.sqrt(2.0+TAU)
	const = 1.0
	m1 = const*np.array([[ 1.0,  TAU,  TAU,  0.0, -1.0,  0.0],\
						[ TAU,  0.0,  0.0,  1.0,  TAU,  1.0],\
						[ 0.0,  1.0, -1.0, -TAU,  0.0,  TAU],\
						[ TAU, -1.0, -1.0,  0.0, -TAU,  0.0],\
						[-1.0,  0.0,  0.0,  TAU, -1.0,  TAU],\
						[ 0.0,  TAU, -TAU,  1.0,  0.0, -1.0]])
	v = matrix_dot_cy(m1,n)
	v1 = v[0][0] # x in Epar
	v2 = v[1][0] # y in Epar
	v3 = v[2][0] # z in Epar
	v4 = v[3][0] # x in Eperp
	v5 = v[4][0] # y in Eperp
	v6 = v[5][0] # z in Eperp
	return [v1,v2,v3,v4,v5,v6]

cpdef np.ndarray matrix_dot_cy(np.ndarray array_1, np.ndarray array_2):
#def matrix_dot_cy(np.ndarray array_1, np.ndarray array_2):
	cdef Py_ssize_t mx1,my1,mx2,my2
	cdef Py_ssize_t x,y,z
	
	mx1 = array_1.shape[0]
	my1 = array_1.shape[1]
	mx2 = array_2.shape[0]
	my2 = array_2.shape[1]
		
	cdef np.ndarray[DTYPE_t, ndim=2] array_3
	
	array_3 = np.zeros((mx1,my2), dtype=np.float64)
	for x in range(array_1.shape[0]):
		for y in range(array_2.shape[1]):
			for z in range(array_1.shape[1]):
				array_3[x][y] += array_1[x][z] * array_2[z][y]
	return array_3

cpdef det_matrix_cy(np.ndarray a, np.ndarray b, np.ndarray c):
	cdef double a1,a2,a3,b1,b2,b3,c1,c2,c3
	return a[0]*b[1]*c[2]+a[2]*b[0]*c[1]+a[1]*b[2]*c[0]-a[2]*b[1]*c[0]-a[1]*b[0]*c[2]-a[0]*b[2]*c[1]
