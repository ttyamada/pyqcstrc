# -*- coding: utf-8 -*-
#
# PyQC - Python tools for Quasi-Crystallography
# Copyright (c) 2020 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
#
# python setup.py build_ext --inplace
#
import sys
import numpy as np
cimport numpy as np
cimport cython
from scipy.spatial import Delaunay

DTYPE = np.int
#DTYPE = np.int64
#DOUBLE = np.float64
#ctypedef np.int64_t DTYPE_t
#ctypedef np.float64_t DOUBLE_t
ctypedef np.float64_t DTYPE_t

#cdef np.float64_t TAU=1.618033988749895 # (1.0+np.sqrt(5.0))/2.0
cdef np.float64_t TAU=(1.0+np.sqrt(5.0))/2.0
#cdef np.float64_t BVAL=np.sqrt(3)/2.0
cdef np.float64_t EPS=1e-6 # tolerance


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
		print 'shift_object()'
		if vorbose>1:
			if vorbose>0:
				print ' volume: %d %d %d (%10.8f)'%(v0,v1,v2,vol1)
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
			print ' succeeded'
		else:
			pass
		return obj_new
	else:
		print ' fail'
		return np.array([0]).reshape(1,1,1,1)

cdef np.ndarray two_segment_into_one(np.ndarray[np.int64_t, ndim=3] line_segment_1,np.ndarray[np.int64_t, ndim=3] line_segment_2):
	cdef int i,flag,counter,verbose
	cdef list comb
	cdef np.ndarray[np.int64_t, ndim=1] tmp1
	cdef np.ndarray[np.int64_t, ndim=1] edge1,removed_vtrx
	cdef np.ndarray[np.int64_t, ndim=2] edge1a,edge1b,edge2a,edge2b
	
	comb=[[0,1,0,1],[0,1,1,0],[1,0,0,1],[1,0,1,0]]
	#edge1=np.array([0])
	#removed_vtrx=np.array([0])
	counter=0
	
	verbose=0
	
	for i1 in range(len(comb)):
		edge1a=line_segment_1[comb[i1][0]]
		edge1b=line_segment_1[comb[i1][1]]
		edge2a=line_segment_2[comb[i1][2]]
		edge2b=line_segment_2[comb[i1][3]]
		#a1,a2,a3,a4,a5,a6=projection(edge1a[0],edge1a[1],edge1a[2],edge1a[3],edge1a[4],edge1a[5])
		#a1,a2,a3,b4,b5,b6=projection(edge2a[0],edge2a[1],edge2a[2],edge2a[3],edge2a[4],edge2a[5])
		#if np.all(a4==b4) and np.all(a5==b5) and np.all(a6==b6):
		flag=check_two_vertices(edge1a,edge2a,verbose)
		if flag==1: # equivalent
			flag=point_on_segment(edge1a,edge1b,edge2b)
			if flag==0:
				tmp1=np.append(edge1b,edge2b)
				tmp1=np.append(tmp1,edge1a)
				#edge1=np.append(edge1b,edge2b)
				#removed_vtrx=edge1a
				counter+=1
				break
			else:
				pass
		else:
			pass
	#return edge1,removed_vtrx
	if counter!=0:
		return tmp1.reshape(3,6,3)
	else:
		return np.array([0]).reshape(1,1,1)

cdef np.ndarray check_two_triangles(np.ndarray[np.int64_t, ndim=3] triange_1,np.ndarray[np.int64_t, ndim=3] triange_2, int verbose):
	
	# ２つの三角形をチェックする
	# 同一の場合、
	# １つの頂点を共有し、同一平面にある場合
	# ２つの頂点を共有し、同一平面にある場合
	#
	cdef int i,i1,i2,i3,i4,i5,i6,i7,i8,counter
	cdef np.ndarray[np.int64_t,ndim=1] t1,t2,t3,a1,a2,a3,b1,b2,b3,c1,c2,c3,d1,d2,d3,e1,e2,e3,f1,f2,f3
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a
	cdef np.ndarray[np.int64_t,ndim=2] tmp2a,tmp2b,tmp2c,tmp2d,tmp2e,tmp2f
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b
	cdef list comb1
	
	if verbose>0:
		print '            check_two_triangles()'
	else:
		pass
	tmp3a=triange_1
	tmp3b=triange_2
	
	tmp1a=np.append(triange_1,triange_2)
	tmp3b=remove_doubling_dim3_in_perp_space(tmp1a.reshape(len(tmp1a)/18,6,3))
	if coplanar_check(tmp3b)==1: # 2つの三角形は同一平面上
		if len(tmp3b)==3: # 2つの三角形は同一
			if verbose>0:
				print '              two triangles are identical'
			else:
				pass
				return triange_1
		elif len(tmp3b)==4: # 2つの三角形は２頂点共有
			comb1=[[0,1,3],[0,2,1],[1,2,0]]
			for i1 in range(len(comb1)):
				[i3,i4,i7]=comb1[i1]
				t1,t2,t3,a1,a2,a3=projection(tmp3a[i3][0],tmp3a[i3][1],tmp3a[i3][2],tmp3a[i3][3],tmp3a[i3][4],tmp3a[i3][5])
				t1,t2,t3,c1,c2,c3=projection(tmp3a[i4][0],tmp3a[i4][1],tmp3a[i4][2],tmp3a[i4][3],tmp3a[i4][4],tmp3a[i4][5])
				tmp2a=np.array([a1,a2,a3])
				tmp2c=np.array([c1,c2,c3])
				counter=0
				for i2 in range(len(comb1)):
					[i5,i6,i8]=comb1[i2]
					t1,t2,t3,b1,b2,b3=projection(tmp3b[i5][0],tmp3b[i5][1],tmp3b[i5][2],tmp3b[i5][3],tmp3b[i5][4],tmp3b[i5][5])
					t1,t2,t3,d1,d2,d3=projection(tmp3b[i6][0],tmp3b[i6][1],tmp3b[i6][2],tmp3b[i6][3],tmp3b[i6][4],tmp3b[i6][5])
					tmp2b=np.array([b1,b2,b3])
					tmp2d=np.array([d1,d2,d3])
					if (np.all(tmp2a==tmp2b) and np.all(tmp2c==tmp2d)) or (np.all(tmp2a==tmp2d) and np.all(tmp2c==tmp2b)):
						tmp1a=np.append(tmp3a[i7],tmp3b[i8],tmp3a[i3],tmp3a[i4])
						counter=1
						break
					else:
						pass
				if counter==1:
					break
				else:
					print 'ERROR'
					pass
			# 2つの三角形が一つの三角形としてまとめられる場合をチェック
			# (case 1)
			tmp3a=tmp1a.reshape(4,6,3)
			tmp3b=np.append(tmp3a[0],tmp3a[2])
			tmp3c=np.append(tmp3a[1],tmp3a[2])
			tmp3d=two_segment_into_one(tmp3b,tmp3c)
			if len(tmp3d)!=1: # 2つの三角形が一つの三角形としてまとめられる
				tmp1a=np.append(tmp3a[i7],tmp3b[i8],tmp3a[i4])
				tmp3a=tmp1a.reshape(3,6,3)
				if verbose>1:
					print '              two triangles merged into one'
				else:
					pass
			# (case 2)
			else:
				tmp3b=np.append(tmp3a[0],tmp3a[3])
				tmp3c=np.append(tmp3a[1],tmp3a[3])
				tmp3d=two_segment_into_one(tmp3b,tmp3c)
				if len(tmp3d)!=1: # 2つの三角形が一つの三角形としてまとめられる
					tmp1a=np.append(tmp3a[i7],tmp3b[i8],tmp3a[i3])
					tmp3a=tmp1a.reshape(3,6,3)
					if verbose>1:
						print '              two triangles merged into one'
					else:
						pass
				else: # まとめられない
					if verbose>1:
						print '              two triangles are sharing one edge'
					else:
						pass
			return tmp3a

		elif len(tmp3b)==5: # 2つの三角形が1頂点共有
			comb1=[[0,1,3],[0,2,1],[1,2,0]]
			for i1 in range(len(comb1)):
				[i3,i5,i6]=comb1[i1]
				t1,t2,t3,a1,a2,a3=projection(tmp3a[i3][0],tmp3a[i3][1],tmp3a[i3][2],tmp3a[i3][3],tmp3a[i3][4],tmp3a[i3][5])
				tmp2a=np.array([a1,a2,a3])
				counter=0
				for i2 in [0,1,2]:
					[i4,i7,i8]=comb1[i2]
					t1,t2,t3,b1,b2,b3=projection(tmp3b[i4][0],tmp3b[i4][1],tmp3b[i4][2],tmp3b[i4][3],tmp3b[i4][4],tmp3b[i4][5])
					tmp2b=np.array([b1,b2,b3])
					if np.all(tmp2a==tmp2b):
						tmp1a=np.append(tmp3a[i5],tmp3a[i6],tmp3b[i7],tmp3b[i8],tmp3a[i3])
						counter=1
						break
					else:
						pass
				if counter==1:
					break
				else:
					print 'ERROR'
					pass
			if verbose>1:
				print '              two triangles are sharing one vertex'
			else:
				pass
			return tmp1a.reshape(5,6,3)
		else:
			if verbose>1:
				print '              not coplaner'
			else:
				pass
				return np.array([0]).reshape(1,1,1)

cdef int coplanar_check_two_triangles(np.ndarray[np.int64_t, ndim=3] triange1,np.ndarray[np.int64_t, ndim=3] triange2, int verbose):
	cdef int i1
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a
	tmp1a=np.append(triange1,triange2)
	tmp3a=remove_doubling_dim3_in_perp_space(tmp1a.reshape(len(tmp1a)/18,6,3))
	if verbose>=0:
		print '        coplanar_check_two_triangles()'
	else:
		pass
	if coplanar_check(tmp3a)==1:
		if verbose>=0:
			print '         coplanar'
		else:
			pass
		return 1 # coplanar
	else:
		if verbose>=0:
			print '         not coplanar'
		else:
			pass
			return 0

cpdef int check_two_vertices(np.ndarray[np.int64_t, ndim=2] vertex1,np.ndarray[np.int64_t, ndim=2] vertex2, int verbose):
	"""
	# TAU-style calc
	cdef np.ndarray[np.int64_t,ndim=1] t1,t2,t3,a1,a2,a3,b1,b2,b3
	t1,t2,t3,a1,a2,a3=projection(vertex1[0],vertex1[1],vertex1[2],vertex1[3],vertex1[4],vertex1[5])
	t1,t2,t3,b1,b2,b3=projection(vertex2[0],vertex2[1],vertex2[2],vertex2[3],vertex2[4],vertex2[5])
	
	if verbose>3:
		print '          check_two_vertices()'
	else:
		pass

	if (np.all(a1==b1) and np.all(a2==b2) and np.all(a3==b3)):
		if verbose>3:
			print '           equivalent'
		else:
			pass
		return 1
	else:
		if verbose>3:
			print '           inequivalent'
		else:
			pass
		return 0
	"""
	# numerical calc
	cdef double x0,y0,z0,x1,y1,z1
	cdef list tmp1a,tmp1b
	cdef np.ndarray[np.int64_t,ndim=1] t1,t2,t3,a1,a2,a3,b1,b2,b3
	if verbose>0:
		print '          check_two_vertices()'
	else:
		pass
	tmp1a=projection_numerical((vertex1[0][0]+TAU*vertex1[0][1])/float(vertex1[0][2]),\
								(vertex1[1][0]+TAU*vertex1[1][1])/float(vertex1[1][2]),\
								(vertex1[2][0]+TAU*vertex1[2][1])/float(vertex1[2][2]),\
								(vertex1[3][0]+TAU*vertex1[3][1])/float(vertex1[3][2]),\
								(vertex1[4][0]+TAU*vertex1[4][1])/float(vertex1[4][2]),\
								(vertex1[5][0]+TAU*vertex1[5][1])/float(vertex1[5][2]))
	x0,y0,z0=tmp1a[3],tmp1a[4],tmp1a[5]
	tmp1b=projection_numerical((vertex2[0][0]+TAU*vertex2[0][1])/float(vertex2[0][2]),\
								(vertex2[1][0]+TAU*vertex2[1][1])/float(vertex2[1][2]),\
								(vertex2[2][0]+TAU*vertex2[2][1])/float(vertex2[2][2]),\
								(vertex2[3][0]+TAU*vertex2[3][1])/float(vertex2[3][2]),\
								(vertex2[4][0]+TAU*vertex2[4][1])/float(vertex2[4][2]),\
								(vertex2[5][0]+TAU*vertex2[5][1])/float(vertex2[5][2]))
	x1,y1,z1=tmp1b[3],tmp1b[4],tmp1b[5]
	if abs(x0-x1)<EPS and abs(y0-y1)<EPS and abs(z0-z1)<EPS:
		if verbose>1:
			print '           equivalent'
		else:
			pass
		return 1
	else:
		if verbose>1:
			print '           inequivalent'
		else:
			pass
		return 0

cdef int check_two_edges(np.ndarray[np.int64_t, ndim=3] edge1,np.ndarray[np.int64_t, ndim=3] edge2, int verbose):
	cdef int flag1,flag2
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a
	
	if verbose>0:
		print '         check_two_edges()'
	else:
		pass
	
	flag1=0
	for i1 in range(2):
		for i2 in range(2):
			flag1+=check_two_vertices(edge1[i1],edge2[i2],verbose-1)
	if flag1==2:
		if verbose>1:
			print '          equivalent'
		else:
			pass
		return 0
	else:
		# edge1   A==B
		# edge2   C==D
		# check, point C and A==B
		flag1=point_on_segment(edge2[0],edge1[0],edge1[1])
		if   flag1==2:  # C is not on a line passing through A and B.
			if verbose>1:
				print '          inequivalent'
			else:
				pass
			return 2
		else:
			# check, point D and A==B
			flag2=point_on_segment(edge2[0],edge1[0],edge1[1])
			if   flag2==2:  # C is not on a line passing through A and B.
				if verbose>1:
					print '          inequivalent'
				else:
					pass
				return 2
			else:
				if   flag1== 0 and flag2== 0: #   A==CD=B
					if verbose>1:
						print '          A==CD=B'
					else:
						pass
					return 1
				elif flag1==-1 and flag2== 0: # C A==D==B
					if verbose>1:
						print '          C A==D==B'
					else:
						pass
					return 2
				elif flag1==-1 and flag2== 1: # C A=====B D
					if verbose>1:
						print '          C A=====B D'
					else:
						pass
					return -1
				elif flag1== 0 and flag2== 1: #   A==C==B D
					if verbose>1:
						print '          A==C==B D'
					else:
						pass
					return 2
				elif flag1== 0 and flag2==-1: # D A==C==B
					if verbose>1:
						print '          D A==C==B'
					else:
						pass
					return 2
				elif flag1== 1 and flag2==-1: # D A=====B C
					if verbose>1:
						print '          D A=====B C'
					else:
						pass
					return 2
				elif flag1== 1 and flag2== 0: #   A==D==B C
					if verbose>1:
						print '          A==D==B C'
					else:
						pass
					return 2
				else:
					return 2
				
cdef np.ndarray merge_two_edges(np.ndarray[np.int64_t, ndim=3] edge1,np.ndarray[np.int64_t, ndim=3] edge2, int verbose):
	cdef int flag1,flag2
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a

	if verbose>0:
		print '           merge_two_edges()'
	else:
		pass
	
	flag1=0
	for i1 in range(2):
		for i2 in range(2):
			flag1+=check_two_vertices(edge1[i1],edge1[i2],verbose)
	if flag1==2:
		if verbose>1:
			print '           merged'
		else:
			pass
		return edge1
	else:
		# edge1   A==B
		# edge2   C==D
		# check, point C and A==B
		flag1=point_on_segment(edge2[0],edge1[0],edge1[1])
		if   flag1==2:  # C is not on a line passing through A and B.
			if verbose>1:
				print '           cannot merged'
			else:
				pass
			return np.array([0]).reshape(1,1,1)
		else:
			# check, point D and A==B
			flag2=point_on_segment(edge2[0],edge1[0],edge1[1])
			if   flag1==2:  # C is not on a line passing through A and B.
				if verbose>1:
					print '           cannot merged'
				else:
					pass
				return np.array([0]).reshape(1,1,1)
			else:
				if verbose>1:
					print '           merged'
				else:
					pass
				if   flag1== 0 and flag2== 0: #   A==CD=B
					return edge1
				elif flag1==-1 and flag2== 0: # C A==D==B
					tmp1a=np.append(edge2[0],edge1[1])
					return tmp1a.reshape(2,6,3)
				elif flag1==-1 and flag2== 1: # C A=====B D
					tmp1a=np.append(edge2[0],edge2[1])
					return tmp1a.reshape(2,6,3)
				elif flag1== 0 and flag2== 1: #   A==C==B D
					tmp1a=np.append(edge1[0],edge2[1])
					return tmp1a.reshape(2,6,3)
				elif flag1== 0 and flag2==-1: # D A==C==B
					tmp1a=np.append(edge2[1],edge1[1])
					return tmp1a.reshape(2,6,3)
				elif flag1== 1 and flag2==-1: # D A=====B C
					tmp1a=np.append(edge2[1],edge2[0])
					return tmp1a.reshape(2,6,3)
				elif flag1== 1 and flag2== 0: #   A==D==B C
					tmp1a=np.append(edge1[0],edge1[0])
					return tmp1a.reshape(2,6,3)
					
cdef np.ndarray triangles_to_edges(np.ndarray[np.int64_t, ndim=4] triangles,int verbose):
	# parameter, set of triangles
	# returns, set of vertices
	cdef int i1,i2,ji,j2
	cdef list combination
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a
	
	if verbose>0:
		print '       triangles_to_edges()'
	else:
		pass
	
	tmp1b=np.array([0])
	combination=[[0,1],[0,2],[1,2]]
	for i1 in range(len(triangles)):
		for i2 in range(3):
			j1=combination[i2][0]
			j2=combination[i2][1]
			tmp1a=np.append(triangles[i1][j1],triangles[i1][j2])
			if len(tmp1b)==1:
				tmp1b=tmp1a
			else:
				tmp1b=np.append(tmp1b,tmp1a)
	return tmp1b.reshape(len(tmp1b)/36,2,6,3)

cpdef list surface_cleaner(np.ndarray[np.int64_t, ndim=4] surface,int num_cycle,int verbose):
	
	# 同一平面上にある三角形を求め、グループ分けする
	# 各グループにおいて、以下を行う．
	# 各グループにおいて、三角形の３辺が、他のどの三角形とも共有していない辺を求める
	# そして、２つの辺が１つの辺にまとめられるのであれば、まとめる
	# 辺の集合をアウトプット
	
	cdef int flag,counter1,counter2
	cdef int i0,i1,i2,i3,i4,i5
	cdef list list_0,list_1,list_2,skip_list,combination
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b,tmp1c,tmp1d,tmp1e
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b,tmp3c
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a,tmp4b,tmp4c
	
	if verbose>0:
		print '       surface_cleaner()'
	else:
		pass
	
	obj_edge_all=np.array([0]).reshape(1,1,1,1)
	combination=[[0,1],[0,2],[1,2]]

	# 同一平面上にある三角形を求め、集合Aとする
	list_0=[]
	list_2=[]
	skip_list=[-1]
	for i1 in range(len(surface)-1):
		tmp1a=np.array([0])
		tmp1b=np.array([0])
		tmp1c=np.array([0])
		counter2=0
		list_1=[]
		counter2=0
		for i2 in range(i1+1,len(surface)):
			counter1=0
			for i3 in skip_list:
				if i1==i3 or i2==i3:
					counter1+=1
					break
				else:
					pass
			if counter1==0:
				flag=coplanar_check_two_triangles(surface[i1],surface[i2],verbose-3)
				if verbose>1:
					print '         %3d %3d %3d'%(i1,i2,flag)
				else:
					pass
				if flag==1: # coplanar
					if len(list_1)==0:
						list_1.append(i1)
					else:
						pass
					skip_list.append(i2)
					list_1.append(i2)
				counter2+=1
		if counter2!=0:
			if len(list_1)!=0:
				list_0.append(list_1)
			else:
				list_0.append([i1])
		else:
			pass
	# check the last triangle in 'surface'
	counter1=0
	for i1 in skip_list:
		if i1==len(surface)-1:
			counter1+=1
			break
		else:
			pass
	if counter1==0:
		list_0.append([len(surface)-1])
	else:
		pass
	
	if verbose>1:
		print '        number of set of coplanar triangles, %d'%(len(list_0))
	else:
		pass
	
	tmp1d=np.array([0])
	for i1 in range(len(list_0)):
		tmp1a=np.array([0])
		tmp1c=np.array([0])
		tmp1e=np.array([0])
		if verbose>1:
			print '          %d-th set of triangles'%(i1)
			print '            number of trianges, %d'%(len(list_0[i1]))
			#print '            ',list_0[i1]
		for i2 in list_0[i1]:
			if len(tmp1a)==1:
				tmp1a=surface[i2].reshape(54) #3*6*3
			else:
				tmp1a=np.append(tmp1a,surface[i2])
		
		#集合Aに含まれる三角形のそれぞれの三辺について、どの三角形とも共有していない辺を求める．
		tmp4a=tmp1a.reshape(len(tmp1a)/54,3,6,3) # set of triangles
		if len(list_0[i1])!=1:
			tmp4b=triangles_to_edges(tmp4a,verbose-1)
			if verbose>1:
				print '            number of edges, %d'%(len(tmp4b))
			else:
				pass
			for i2 in range(len(tmp4a)):
				for i3 in range(len(combination)):
					j1=combination[i3][0]
					j2=combination[i3][1]
					tmp1a=np.append(tmp4a[i2][j1],tmp4a[i2][j2])
					tmp3a=tmp1a.reshape(2,6,3)
					counter1=0
					for i4 in range(len(tmp4b)):
						flag=check_two_edges(tmp3a,tmp4b[i4],verbose-1)
						if flag==0: # equivalent
							counter1+=1
						else:
							pass
						if counter1==2:
							break
						else:
							pass
					if counter1==1:
						if len(tmp1c)==1:
							tmp1c=tmp1a
						else:
							tmp1c=np.append(tmp1c,tmp1a)
					else:
						pass
			if verbose>1:
				print '            number of independent edges, %d'%(len(tmp1c)/36)
			else:
				pass
			# ２つの辺が１つの辺にまとめられるのであれば、まとめる
			if len(tmp1c)!=1:
				tmp4c=tmp1c.reshape(len(tmp1c)/36,2,6,3)
				for i0 in range(num_cycle):
					tmp1e=np.array([0])
					skip_list=[-1]
					for i2 in range(len(tmp4c)-1):
						for i3 in range(i2+1,len(tmp4c)):
							counter1=0
							for i4 in skip_list:
								if i2==i4 or i3==i4:
									counter1+=1
									break
								else:
									pass
							if counter1==0:
								tmp3a=two_segment_into_one(tmp4c[i2],tmp4c[i3])
								#tmp3a=merge_two_edges(tmp4c[i2],tmp4c[i3],verbose)
								if len(tmp3a)!=1:
									#if len(skip_list)==1:
									#	skip_list.append(i2)
									#else:
									#	pass
									skip_list.append(i2)
									skip_list.append(i3)
									if len(tmp1e)==1:
										tmp1e=np.append(tmp3a[0],tmp3a[1])
									else:
										tmp1e=np.append(tmp1e,tmp3a[0])
										tmp1e=np.append(tmp1e,tmp3a[1])
									break
								else:
									pass
							else:
								pass
					for i2 in range(len(tmp4c)):
						counter1=0
						for i3 in skip_list:
							if i2==i3:
								counter1+=1
								break
							else:
								pass
						if counter1==0:
							if len(tmp1e)==1:
								tmp1e=tmp4c[i2].reshape(36) # 2*6*3=36
							else:
								tmp1e=np.append(tmp1e,tmp4c[i2])
						else:
							pass
					if verbose>1:
						print '            %d cycle %d -> %d'%(i0,len(tmp4c),len(tmp1e)/36)
					else:
						pass
					if len(tmp4c)==len(tmp1e)/36:
						break
					else:
						pass
					tmp4c=tmp1e.reshape(len(tmp1e)/36,2,6,3)
		
		else: # 同一平面に三角形が一つだけある場合
			if verbose>1:
				print '            number of independent edges, 3'
			else:
				pass
			tmp4c=triangles_to_edges(tmp4a,verbose-1)
		
		# Merge
		list_2.append(tmp4c)
		
	return list_2
	
cdef np.ndarray chech_four_edges_forms_tetrahedron(np.ndarray[np.int64_t, ndim=3] edges1,\
											np.ndarray[np.int64_t, ndim=3] edges2,\
											np.ndarray[np.int64_t, ndim=3] edges3,\
											np.ndarray[np.int64_t, ndim=3] edges4,\
											np.ndarray[np.int64_t, ndim=3] edges5,\
											np.ndarray[np.int64_t, ndim=3] edges6,\
											int verbose):
	# 与えられた4つの辺が4面体を成すかを判定
	# Judgement: whether given four edges form a tetrahedron or not.
	if verbose>0:
		print 'chech_four_edges_forms_tetrahedron()'
	else:
		pass
	cdef np.ndarray[np.int64_t, ndim=1] tmp1a
	cdef np.ndarray[np.int64_t, ndim=3] tmp3a,tmp3b
	cdef np.ndarray[np.int64_t, ndim=4] tmp4a
	tmp1a=np.append(edges1,edges2)
	tmp1a=np.append(tmp1a,edges3)
	tmp1a=np.append(tmp1a,edges4)
	tmp1a=np.append(tmp1a,edges5)
	tmp1a=np.append(tmp1a,edges6)
	tmp3b=tmp1a.reshape(len(tmp1a)/18,6,3)
	tmp3a=remove_doubling_dim3_in_perp_space(tmp3b)
	if len(tmp3a)==4: 
		# 与えられた4つの辺の各頂点について、独立なものが4つだけの場合、4面体を成す
		# Given four edges form a tetrahedron
		tmp4a=tetrahedralization_points(tmp3a)
		if verbose>1:
			print ' found'
		else:
			pass
		return tmp4a
	else:
		if verbose>1:
			print ' not found'
		else:
			pass
		return np.array([0]).reshape(1,1,1,1)
		
cdef np.ndarray search_polyhedron_in_edges(np.ndarray[np.int64_t, ndim=4] edges,int verbose):
	# 与えられた面から、4面体を探す
	return 0

cdef np.ndarray search_tetrahedron_in_edges(np.ndarray[np.int64_t, ndim=4] edges,int verbose):
	# 与えられた辺の中で、4面体を探す
	cdef int flag,i1,i2,i3,i4,i5,i6
	cdef double vol1
	cdef np.ndarray[np.int64_t, ndim=1] tmp1a
	cdef np.ndarray[np.int64_t, ndim=4] tmp4a
	
	if verbose>0:
		print '          search_tetrahedron_in_edges()'
	else:
		pass

	if verbose>1:
		print '           number of egses, %d'%(len(edges))
	else:
		pass
	
	tmp1b=np.array([0])
	if len(edges)>5:
		for i1 in range(len(edges)-5):
			for i2 in range(i1+1,len(edges)-4):
				for i3 in range(i1+2,len(edges)-3):
					for i4 in range(i1+3,len(edges)-2):
						for i5 in range(i1+3,len(edges)-1):
							for i6 in range(i1+3,len(edges)):
								tmp4a=chech_four_edges_forms_tetrahedron(edges[i1],edges[i2],edges[i3],edges[i4],edges[i5],edges[i6],verbose+1)
								if len(tmp4a)!=1:
									if verbose>1:
										vol1=tetrahedron_volume_6d_numerical(tmp4a)
										print '            volume, %10.8f'%(vol1)
									else:
										pass
									if len(tmp1b)==1:
										tmp1b=tmp4a.reshape(72)
									else: 
										tmp1b=np.append(tmp1b,tmp4a)
								else:
									pass
		if len(tmp1b)!=1:
			tmp4a=tmp1b.reshape(len(tmp1b)/72,4,6,3)
			if verbose>1:
				print '          number of tetrahedron, %d'%(len(tmp4a))
				vol1=obj_volume_6d_numerical(tmp4a)
				print '            total volume, %10.8f'%(vol1)
			else:
				pass
			return tmp1b.reshape(len(tmp1b)/72,4,6,3)
		else:
			return np.array([0]).reshape(1,1,1,1)
	else:
		return np.array([0]).reshape(1,1,1,1)
	
cpdef np.ndarray extract_common_vertex(np.ndarray[np.int64_t, ndim=4] obj,int verbose):
	# 全ての四面体に共通する頂点を得る
	cdef int flag1,counter1,counter2
	cdef int i1,i2,i3
	cdef np.ndarray[np.int64_t, ndim=1] tmp1a
	cdef np.ndarray[np.int64_t, ndim=3] tmp3a
	
	if verbose>0:
		print '      extract_common_vertex()'
	else:
		pass
	
	tmp1a=np.array([0])
	tmp3a=remove_doubling_dim4_in_perp_space(obj)
	for i1 in range(len(tmp3a)):
		counter2=0
		for i2 in range(len(obj)):
			counter1=0
			for i3 in range(4):
				flag1=check_two_vertices(tmp3a[i1],obj[i2][i3],verbose-1)
				if flag1==1: # equivalent
					counter1+=1
					break
				else:
					pass
			if counter1!=0:
				pass
			else:
				counter2+=1
		if counter2==0:
			if len(tmp1a)==1:
				tmp1a=tmp3a[i1].reshape(18)
			else:
				tmp1a=np.append(tmp1a,tmp3a[i1])
		else:
			pass
	if len(tmp1a)!=1:
		if verbose>0:
			print '       number of common vertex, %d'%(len(tmp1a)/18)
		else:
			pass
		return tmp1a.reshape(len(tmp1a)/18,6,3)
	else:
		if verbose>0:
			print '       number of common vertex, 0'
		else:
			pass
		
		return np.array([0]).reshape(1,1,1)

cdef np.ndarray extract_surface_without_specific_vertx(np.ndarray[np.int64_t, ndim=4] triangles ,np.ndarray[np.int64_t, ndim=3] points ,int verbose):	
	# 三角形の集まりから、特定の頂点を含まない三角形を取り出す．
	cdef int i1,i2,i3,flag,counter1
	cdef np.ndarray[np.int64_t, ndim=1] tmp1a
	cdef np.ndarray[np.int64_t, ndim=3] tmp3a
	
	if verbose>0:
		print '      extract_surface_without_specific_vertx()'
	else:
		pass
		
	tmp1a=np.array([0])
	for i1 in range(len(points)):
		for i2 in range(len(triangles)):
			counter1=0
			for i3 in range(3): 
				flag=check_two_vertices(points[i1],triangles[i2][i3],verbose-1)
				if flag==1: # equivalent
					counter1+=1
					break
				else:
					pass
			if counter1==0:
				if len(tmp1a)==1:
					tmp1a=triangles[i2].reshape(54) # 3*6*3
				else:
					tmp1a=np.append(tmp1a,triangles[i2])
			else:
				pass
	if len(tmp1a)!=1:
		if verbose>0:
			print '       number of triangles, %d'%(len(tmp1a)/54)
		else:
			pass
		return tmp1a.reshape(len(tmp1a)/54,3,6,3)
	else:
		if verbose>0:
			print '       number of triangles, 0'
		else:
			pass
		return np.array([0]).reshape(1,1,1,1)

cpdef np.ndarray simplification_convex_polyhedron(np.ndarray[np.int64_t, ndim=4] obj,int num_cycle,int verbose):
	
	# tetrahedralization for convex polyhedron
	cdef int i1
	cdef long v0,v1,v2,v3,v4,v5
	cdef double vol0,vol1,vol2
	cdef list surface_list
	cdef np.ndarray[np.int64_t, ndim=1] tmp1a
	cdef np.ndarray[np.int64_t, ndim=3] tmp3a
	cdef np.ndarray[np.int64_t, ndim=4] tmp4a

	if verbose>0:
		print '      simplification_convex_polyhedron()'
	else:
		pass
	
	v0,v1,v2=obj_volume_6d(obj)
	#vol0=(v0+TAU*v1)/float(v2)
	vol1=obj_volume_6d_numerical(obj)
	if verbose>1:
		#print '       initial volume: %d %d %d (%10.8f) (%10.8f)'%(v0,v1,v2,vol0,vol1)
		print '       initial volume: %d %d %d (%10.8f)'%(v0,v1,v2,vol1)
	else:
		pass
	
	tmp1a=np.array([0])
	tmp4a=np.array([0]).reshape(1,1,1,1)
	surface_list=generate_obj_surface(obj,num_cycle,verbose-1)
	for i1 in range(len(surface_list)):
		if len(tmp1a)==1:
			tmp4a=surface_list[i1]
			tmp1a=tmp4a.reshape(len(tmp4a)*36) # 2*6*3
		else:
			tmp1a=np.append(tmp1a,surface_list[i1])

	tmp3a=tmp1a.reshape(len(tmp1a)/18,6,3)
	tmp3a=remove_doubling_dim3_in_perp_space(tmp3a)
	tmp4a=tetrahedralization_points(tmp3a)
	
	v3,v4,v5=obj_volume_6d(tmp4a)
	#vol0=(v0+TAU*v1)/float(v2)
	vol2=obj_volume_6d_numerical(tmp4a)
		
	if v3==v0 and v4==v1 and v5==v2 or abs(vol1-vol2)<vol2*EPS:
		if verbose>1:
			print '       succdeded, simplified volume: %d %d %d (%10.8f)'%(v0,v1,v2,vol2)
		else:
			pass
		return tmp4a
		
	else:
		if verbose>1:
			print '         fail, initial obj returned'
		else:
			pass
		return obj
	
cpdef list generate_obj_surface(np.ndarray[np.int64_t, ndim=4] obj,int num_cycle,int verbose):	

	cdef long v0,v1,v2,v3,v4,v5
	cdef double vol0,vol1
	cdef list surface_list
	cdef np.ndarray[np.int64_t, ndim=4] obj_surface
	
	if verbose>0:
		print '      generate_obj_surface()'
	else:
		pass
		
	v0,v1,v2=obj_volume_6d(obj)
	#vol0=(v0+TAU*v1)/float(v2)
	vol1=obj_volume_6d_numerical(obj)
	if verbose>1:
		#print '       initial volume: %d %d %d (%10.8f) (%10.8f)'%(v0,v1,v2,vol1)
		print '       initial volume: %d %d %d (%10.8f)'%(v0,v1,v2,vol1)
	else:
		pass
	
	#obj_surface=generator_surface(obj)
	obj_surface=generator_surface_1(obj)
	surface_list=surface_cleaner(obj_surface,num_cycle,verbose-1)
	
	return surface_list

cpdef np.ndarray simplification_obj_smart(np.ndarray[np.int64_t, ndim=4] obj,int num_cycle,int verbose):	
	# 複数の四面体が一つの頂点を共有して集まっている場合の四面体分割
	cdef int i1
	cdef long v0,v1,v2,v3,v4,v5
	cdef double vol0,vol1,vol2
	cdef list surface_list
	cdef np.ndarray[np.int64_t, ndim=1] tmp1a,tmp1b
	cdef np.ndarray[np.int64_t, ndim=3] vertex_common
	cdef np.ndarray[np.int64_t, ndim=3] tmp3a
	cdef np.ndarray[np.int64_t, ndim=4] tmp4a
	cdef np.ndarray[np.int64_t, ndim=4] obj_surface,obj_surface_new,obj_tetrahedron
	
	if verbose>0:
		print '      simplification_obj_smart()'
	else:
		pass
		
	v0,v1,v2=obj_volume_6d(obj)
	vol0=(v0+TAU*v1)/float(v2)
	vol1=obj_volume_6d_numerical(obj)
	if verbose>=2:
		print '       initial volume: %d %d %d (%10.8f) (%10.8f)'%(v0,v1,v2,vol0,vol1)
	else:
		pass
	
	obj_surface=generator_surface(obj)
	
	vertex_common=extract_common_vertex(obj,verbose-1)
	
	obj_surface_new=extract_surface_without_specific_vertx(obj_surface,vertex_common,verbose-1)
		
	surface_list=surface_cleaner(obj_surface_new,num_cycle,verbose-1)
	
	tmp1a=np.array([0])
	if len(vertex_common)==1:
		for i1 in range(len(surface_list)):
			tmp4a=surface_list[i1]
			tmp1b=np.append(tmp4a,vertex_common[0])
			tmp3a=tmp1b.reshape(len(tmp1b)/18,6,3)
			tmp3a=remove_doubling_dim3_in_perp_space(tmp3a)
			tmp4a=tetrahedralization_points(tmp3a)
			if len(tmp1a)==1:
				tmp1a=tmp4a.reshape(len(tmp4a)*72)
			else:
				tmp1a=np.append(tmp1a,tmp4a)
		if len(tmp1a)!=1:
			if verbose>0:
				print '        number of tetrahedron in reduced obj, %d'%(len(tmp1a)/72)
			return tmp1a.reshape(len(tmp1a)/72,4,6,3)
		else:
			if verbose>0:
				print '         fail, initial obj returned'
			else:
				pass
				return obj
	else:
		if verbose>0:
			print '         fail, initial obj returned'
		else:
			pass
		return obj
	
	# ここまで
	"""
	# return obj_edge
	tmp3a=remove_doubling_dim4_in_perp_space(obj_edge)
	if len(tmp3a)>=4:
		tmp4a=tetrahedralization_points(tmp3a)
		v3,v4,v5=obj_volume_6d(tmp4a)
		vol2=(v3+TAU*v4)/float(v5)
		vol3=obj_volume_6d_numerical(tmp4a)
		if verbose>=2:
			print '         volume of reduced obj: %d %d %d (%10.8f) (%10.8f)'%(v3,v4,v5,vol2,vol3)
		else:
			pass
		if v3==v0 and v4==v1 and v5==v2 or abs(vol1-vol3)<vol3*EPS:
			if verbose>=2:
				print '         succdeded'
			else:
				pass
			return tmp4a
		else:
			print '         fail'
			if verbose>=1:
				print '         try to get correct obj'
			else:
				pass
			return tmp4a
	else:
		if verbose>=2:
			print '         fail, initial obj returned'
		else:
			pass
		return obj
	"""

cpdef np.ndarray simplification_obj_edges_using_parents(np.ndarray[np.int64_t, ndim=4] obj, np.ndarray[np.int64_t, ndim=4] obj2, np.ndarray[np.int64_t, ndim=4] obj3, int num_cycle, int verbose_level):
	
	cdef int i1,i2,i3,i4,num,counter1
	cdef long v0,v1,v2,v3,v4,v5,w0,w1,w2,w3,w4,w5
	cdef double vol0,vol1,vol2,vol3,vol4
	cdef list skip_list
	cdef np.ndarray[np.int64_t, ndim=1] tmp1a,tmp1b
	cdef np.ndarray[np.int64_t, ndim=1] edge_new,removed_vertices
	cdef np.ndarray[np.int64_t, ndim=1] a1,a2,a3,a4,a5,a6,b4,b5,b6,c4,c5,c6
	cdef np.ndarray[np.int64_t, ndim=3] tmp3a,tmp3b,tmp3c
	cdef np.ndarray[np.int64_t, ndim=4] obj_surface,obj_edge,tmp4a,tmp4b,tmp4c,tmp4d
	
	if verbose_level>=1:
		print '      simplification_obj_edges_using_parents()'
	else:
		pass
		
	v0,v1,v2=obj_volume_6d(obj)
	vol0=(v0+TAU*v1)/float(v2)
	vol4=obj_volume_6d_numerical(obj)
	if verbose_level>=2:
		print '         initial volume: %d %d %d (%10.8f)(%10.8f)'%(v0,v1,v2,vol0,vol4)
	else:
		pass
	
	obj_surface=generator_surface(obj)
	obj_edge=generator_edge(obj_surface)
	
	for i4 in range(num_cycle):
		skip_list=[-1] # initialize
		
		edge_new=np.array([0])
		removed_vertices=np.array([0])
		for i1 in range(len(obj_edge)-1):
			for i2 in range(i1,len(obj_edge)):
				for i3 in skip_list:
					if i1!=i3 and i2!=i3:
						#tmp1a,tmp1b=two_segment_into_one(obj_edge[i1],obj_edge[i2])
						tmp3c=two_segment_into_one(obj_edge[i1],obj_edge[i2])
						if len(tmp3c)!=1:
							skip_list.append(i1)
							skip_list.append(i2)
							if len(edge_new)==1:
								#edge_new=tmp1a
								edge_new=np.append(tmp3c[0],tmp3c[1])
							else:
								#edge_new=np.append(edge_new,tmp1a)
								edge_new=np.append(edge_new,tmp3c[0])
								edge_new=np.append(edge_new,tmp3c[1])
							if len(removed_vertices)==1:
								#removed_vertices=tmp1b
								removed_vertices=tmp3c[2].reshape(18) # 6*3=36
							else:
								#removed_vertices=np.append(removed_vertices,tmp1b)
								removed_vertices=np.append(removed_vertices,tmp3c[2])
							break
						else:
							pass
					else:
						pass
		
		for i1 in range(len(obj_edge)):
			counter1=0
			for i2 in skip_list:
				if i1==i2:
					counter1+=1
					break
				else:
					pass
			if counter1==0:
				if len(edge_new)==1:
					edge_new=obj_edge[i1].reshape(36) # 2*6*3=36
				else:
					edge_new=np.append(edge_new,obj_edge[i1])
			else:
				pass
		if len(edge_new)!=1: 
			num=len(obj_edge)
			if len(removed_vertices)!=1:
				obj_edge=edge_new.reshape(len(edge_new)/36,2,6,3)
				tmp3a=removed_vertices.reshape(len(removed_vertices)/18,6,3)
				edge_new=np.array([0])
				# remove unnecessary line segments
				for i1 in range(len(obj_edge)):
					counter1=0
					for i2 in range(len(tmp3a)):
						a1,a2,a3,a4,a5,a6=projection(obj_edge[i1][0][0],obj_edge[i1][0][1],obj_edge[i1][0][2],obj_edge[i1][0][3],obj_edge[i1][0][4],obj_edge[i1][0][5])
						a1,a2,a3,b4,b5,b6=projection(obj_edge[i1][1][0],obj_edge[i1][1][1],obj_edge[i1][1][2],obj_edge[i1][1][3],obj_edge[i1][1][4],obj_edge[i1][1][5])
						a1,a2,a3,c4,c5,c6=projection(tmp3a[i2][0],tmp3a[i2][1],tmp3a[i2][2],tmp3a[i2][3],tmp3a[i2][4],tmp3a[i2][5])
						if (np.all(a4==c4) and np.all(a5==c5) and np.all(a6==c6)) or (np.all(b4==c4) and np.all(b5==c5) and np.all(b6==c6)):
							counter1+=1
							break
						else:
							pass
					if counter1==0:
						if len(edge_new)==1:
							edge_new=obj_edge[i1].reshape(36)
						else:
							edge_new=np.append(edge_new,obj_edge[i1])
					else:
						pass
				tmp4b=edge_new.reshape(len(edge_new)/36,2,6,3)
				print '       cycle %d, %d -> %d'%(i4,num,len(tmp4b))
				obj_edge=tmp4b
			else:
				break
			if num==len(tmp4b):
				break
			else:
				pass
		else:
			pass
	# return obj_edge
	tmp3b=remove_doubling_dim4_in_perp_space(obj_edge)
	if len(tmp3b)>=4:
		tmp4a=tetrahedralization_points(tmp3b)
		v3,v4,v5=obj_volume_6d(tmp4a)
		vol1=(v3+TAU*v4)/float(v5)
		vol3=obj_volume_6d_numerical(tmp4a)
		if verbose_level>=2:
			print '         volume of reduced obj: %d %d %d (%10.8f)(%10.8f)'%(v3,v4,v5,vol1,vol3)
		else:
			pass
	
		if v3==v0 and v4==v1 and v5==v2 or abs(vol3-vol4)<vol3*EPS:
			print '         succdeded'
			return tmp4a
		else:
			print '         fail'
			if verbose_level>=1:
				print '         try to get correct obj'
			else:
				pass
			tmp1a=np.array([0])
			for i1 in range(len(tmp4a)):
				tmp4b=tmp4a[i1].reshape(1,4,6,3)
				w0,w1,w2=obj_volume_6d(tmp4b)
				tmp4c=intersection_using_tetrahedron_4(tmp4b,obj2,1,1,1)
				if tmp4c.tolist()!=[[[[0]]]]:
					tmp4c=intersection_using_tetrahedron_4(tmp4c,obj3,1,1,1)
				else:
					pass
				if tmp4c.tolist()!=[[[[0]]]]:
					w3,w4,w5=obj_volume_6d(tmp4c)
					if w0==w3 and w1==w4 and w2==w5:
						if len(tmp1a)==1:
							tmp1a=tmp4b.reshape(72)
						else:
							tmp1a=np.append(tmp1a,tmp4b)
					else:
						#tmp4c=simplification(tmp4c,5,0,0,2,0,0,1)
						tmp4d=object_subtraction_2(tmp4b,tmp4c,verbose_level)
						tmp4c=object_subtraction_2(tmp4b,tmp4d,verbose_level)
						if len(tmp1a)==1:
							tmp1a=tmp4c.reshape(len(tmp4c)*72)
						else:
							tmp1a=np.append(tmp1a,tmp4c)
				else:
					pass
			tmp4a=tmp1a.reshape(len(tmp1a)/72,4,6,3)
			v3,v4,v5=obj_volume_6d(tmp4a)
			vol2=(v3+TAU*v4)/float(v5)
			vol3=obj_volume_6d_numerical(tmp4a)
			print '         volume of reduced obj: %d %d %d (%10.8f)(%10.8f)'%(v3,v4,v5,vol2,vol3)
			if v3==v0 and v4==v1 and v5==v2 or abs(vol3-vol4)<vol3*EPS:
				print '         succdeded'
				return tmp4a
			else:
				#if abs(vol1-vol3)<1e-8:
				if abs(vol1-vol3)<vol1*EPS:
					print '         succdeded'
					return tmp4a
				else: 
					print '         fail, initial obj returned'
					return obj
					#return tmp4a
					#return np.array([0]).reshape(1,1,1,1)
	else:
		print '         fail, initial obj returned'
		return obj

cpdef np.ndarray simplification_obj_edges(np.ndarray[np.int64_t, ndim=4] obj, int num_cycle, int verbose_level):
	
	cdef int i1,i2,i3,i4,num,counter1
	cdef long v0,v1,v2,v3,v4,v5,w0,w1,w2,w3,w4,w5
	cdef double vol0,vol1,vol2,vol3,vol4
	cdef list skip_list
	cdef np.ndarray[np.int64_t, ndim=1] tmp1a,tmp1b
	cdef np.ndarray[np.int64_t, ndim=1] edge_new,removed_vertices
	cdef np.ndarray[np.int64_t, ndim=1] a1,a2,a3,a4,a5,a6,b4,b5,b6,c4,c5,c6
	cdef np.ndarray[np.int64_t, ndim=3] tmp3a,tmp3b,tmp3c
	cdef np.ndarray[np.int64_t, ndim=4] obj_surface,obj_edge,tmp4a,tmp4b,tmp4c,tmp4d
	
	if verbose_level>0:
		print '      simplification_obj_edges()'
	else:
		pass
		
	v0,v1,v2=obj_volume_6d(obj)
	vol0=(v0+TAU*v1)/float(v2)
	vol4=obj_volume_6d_numerical(obj)
	if verbose_level>1:
		print '         initial volume: %d %d %d (%10.8f)(%10.8f)'%(v0,v1,v2,vol0,vol4)
	else:
		pass
	
	#obj_surface=generator_surface(obj)
	obj_surface=generator_surface_1(obj)
	
	obj_edge=generator_edge(obj_surface)
	
	for i4 in range(num_cycle):
		skip_list=[-1] # initialize
		
		edge_new=np.array([0])
		removed_vertices=np.array([0])
		for i1 in range(len(obj_edge)-1):
			for i2 in range(i1,len(obj_edge)):
				for i3 in skip_list:
					if i1!=i3 and i2!=i3:
						tmp3c=two_segment_into_one(obj_edge[i1],obj_edge[i2])
						if len(tmp3c)!=1:
							skip_list.append(i1)
							skip_list.append(i2)
							if len(edge_new)==1:
								edge_new=np.append(tmp3c[0],tmp3c[1])
							else:
								edge_new=np.append(edge_new,tmp3c[0])
								edge_new=np.append(edge_new,tmp3c[1])
							if len(removed_vertices)==1:
								removed_vertices=tmp3c[2].reshape(18) # 6*3=36
							else:
								removed_vertices=np.append(removed_vertices,tmp3c[2])
							break
						else:
							pass
					else:
						pass
		
		for i1 in range(len(obj_edge)):
			counter1=0
			for i2 in skip_list:
				if i1==i2:
					counter1+=1
					break
				else:
					pass
			if counter1==0:
				if len(edge_new)==1:
					edge_new=obj_edge[i1].reshape(36) # 2*6*3=36
				else:
					edge_new=np.append(edge_new,obj_edge[i1])
			else:
				pass
		if len(edge_new)!=1: 
			num=len(obj_edge)
			if len(removed_vertices)!=1:
				obj_edge=edge_new.reshape(len(edge_new)/36,2,6,3)
				tmp3a=removed_vertices.reshape(len(removed_vertices)/18,6,3)
				edge_new=np.array([0])
				# remove unnecessary line segments
				for i1 in range(len(obj_edge)):
					counter1=0
					for i2 in range(len(tmp3a)):
						a1,a2,a3,a4,a5,a6=projection(obj_edge[i1][0][0],obj_edge[i1][0][1],obj_edge[i1][0][2],obj_edge[i1][0][3],obj_edge[i1][0][4],obj_edge[i1][0][5])
						a1,a2,a3,b4,b5,b6=projection(obj_edge[i1][1][0],obj_edge[i1][1][1],obj_edge[i1][1][2],obj_edge[i1][1][3],obj_edge[i1][1][4],obj_edge[i1][1][5])
						a1,a2,a3,c4,c5,c6=projection(tmp3a[i2][0],tmp3a[i2][1],tmp3a[i2][2],tmp3a[i2][3],tmp3a[i2][4],tmp3a[i2][5])
						if (np.all(a4==c4) and np.all(a5==c5) and np.all(a6==c6)) or (np.all(b4==c4) and np.all(b5==c5) and np.all(b6==c6)):
							counter1+=1
							break
						else:
							pass
					if counter1==0:
						if len(edge_new)==1:
							edge_new=obj_edge[i1].reshape(36)
						else:
							edge_new=np.append(edge_new,obj_edge[i1])
					else:
						pass
				tmp4b=edge_new.reshape(len(edge_new)/36,2,6,3)
				if verbose_level>2:
					print '       cycle %d, %d -> %d'%(i4,num,len(tmp4b))
				else:
					pass
				obj_edge=tmp4b
			else:
				break
			if num==len(tmp4b):
				break
			else:
				pass
		else:
			pass
	# return obj_edge
	tmp3b=remove_doubling_dim4_in_perp_space(obj_edge)
	if len(tmp3b)>=4:
		tmp4a=tetrahedralization_points(tmp3b)
		v3,v4,v5=obj_volume_6d(tmp4a)
		vol1=(v3+TAU*v4)/float(v5)
		vol3=obj_volume_6d_numerical(tmp4a)
		if verbose_level>2:
			print '         volume of reduced obj: %d %d %d (%10.8f)(%10.8f)'%(v3,v4,v5,vol1,vol3)
		else:
			pass
	
		if v3==v0 and v4==v1 and v5==v2 or abs(vol3-vol4)<vol3*EPS:
			if verbose_level>2:
				print '         succdeded'
			else:
				pass
			return tmp4a
		else:
			print '         fail'
			if verbose_level>2:
				print '         try to get correct obj'
			else:
				pass
			tmp1a=np.array([0])
			for i1 in range(len(tmp4a)):
				tmp4b=tmp4a[i1].reshape(1,4,6,3)
				w0,w1,w2=obj_volume_6d(tmp4b)
				tmp4c=intersection_using_tetrahedron_4(tmp4b,obj,1,verbose_level-1,1)
				if tmp4c.tolist()!=[[[[0]]]]:
					w3,w4,w5=obj_volume_6d(tmp4c)
					if w0==w3 and w1==w4 and w2==w5:
						if len(tmp1a)==1:
							tmp1a=tmp4b.reshape(72)
						else:
							tmp1a=np.append(tmp1a,tmp4b)
					else:
						#tmp4c=simplification(tmp4c,5,0,0,2,0,0,1)
						tmp4d=object_subtraction_2(tmp4b,tmp4c,verbose_level-1)
						tmp4c=object_subtraction_2(tmp4b,tmp4d,verbose_level-1)
						if len(tmp1a)==1:
							tmp1a=tmp4c.reshape(len(tmp4c)*72)
						else:
							tmp1a=np.append(tmp1a,tmp4c)
				else:
					pass
			tmp4a=tmp1a.reshape(len(tmp1a)/72,4,6,3)
			v3,v4,v5=obj_volume_6d(tmp4a)
			vol2=(v3+TAU*v4)/float(v5)
			vol3=obj_volume_6d_numerical(tmp4a)
			print '         volume of reduced obj: %d %d %d (%10.8f)(%10.8f)'%(v3,v4,v5,vol2,vol3)
			if v3==v0 and v4==v1 and v5==v2 or abs(vol3-vol4)<vol3*EPS:
				if verbose_level>2:
					print '         succdeded'
				else:
					pass
				return tmp4a
			else:
				if abs(vol1-vol2)<1e-8:
					if verbose_level>2:
						print '         succdeded'
					else:
						pass
					return tmp4a
				else: 
					if verbose_level>2:
						print '         fail, initial obj returned'
					else:
						pass
					return obj
					#return tmp4a
					#return np.array([0]).reshape(1,1,1,1)
	else: 
		if verbose_level>1:
			print '         fail, initial obj returned'
		else:
			pass
		return obj

# NEW
cpdef np.ndarray simplification_obj_edges_1(np.ndarray[np.int64_t, ndim=4] obj,int num_cycle,int verbose):
	
	# tetrahedralization for convex polyhedron
	cdef int i1
	cdef long v0,v1,v2,v3,v4,v5
	cdef long w0,w1,w2,w3,w4,w5,
	cdef double vol0,vol1,vol2
	cdef list surface_list
	cdef np.ndarray[np.int64_t, ndim=1] tmp1a
	cdef np.ndarray[np.int64_t, ndim=3] tmp3a
	cdef np.ndarray[np.int64_t, ndim=4] tmp4a,tmp4b,tmp4c,tmp4d

	if verbose>0:
		print '      simplification_obj_edges_1()'
	else:
		pass
	
	v0,v1,v2=obj_volume_6d(obj)
	#vol0=(v0+TAU*v1)/float(v2)
	vol1=obj_volume_6d_numerical(obj)
	if verbose>1:
		#print '       initial volume: %d %d %d (%10.8f) (%10.8f)'%(v0,v1,v2,vol0,vol1)
		print '       initial volume: %d %d %d (%10.8f)'%(v0,v1,v2,vol1)
	else:
		pass
	
	tmp1a=np.array([0])
	tmp4a=np.array([0]).reshape(1,1,1,1)
	
	surface_list=generate_obj_surface(obj,num_cycle,verbose-1)
	
	for i1 in range(len(surface_list)):
		if len(tmp1a)==1:
			tmp4a=surface_list[i1]
			tmp1a=tmp4a.reshape(len(tmp4a)*36) # 2*6*3
		else:
			tmp1a=np.append(tmp1a,surface_list[i1])

	tmp3a=tmp1a.reshape(len(tmp1a)/18,6,3)
	tmp3a=remove_doubling_dim3_in_perp_space(tmp3a)
	tmp4a=tetrahedralization_points(tmp3a)
	
	v3,v4,v5=obj_volume_6d(tmp4a)
	#vol0=(v0+TAU*v1)/float(v2)
	vol2=obj_volume_6d_numerical(tmp4a)
		
	if v3==v0 and v4==v1 and v5==v2 or abs(vol1-vol2)<vol2*EPS:
		if verbose>0:
			#print '       succdeded, simplified volume: %d %d %d (%10.8f) (%10.8f)'%(v0,v1,v2,vol0,vol2)
			print '       succdeded, simplified volume: %d %d %d (%10.8f)'%(v0,v1,v2,vol2)
		else:
			pass
		return tmp4a
		
	else:
		if verbose>0:
			print '         fail'
		else:
			pass
		
		vol3=0.0
		tmp1a=np.array([0])
		for i1 in range(len(tmp4a)):
			tmp4b=tmp4a[i1].reshape(1,4,6,3)
			w0,w1,w2=obj_volume_6d(tmp4b)
			tmp4c=intersection_using_tetrahedron_4(tmp4b,obj,1,verbose-1,1) # option=1
			#tmp4c=intersection_using_tetrahedron_4(tmp4b,obj,0,verbose-1,1) # option=0
			if tmp4c.tolist()!=[[[[0]]]]:
				w3,w4,w5=obj_volume_6d(tmp4c)
				#
				vol3+=obj_volume_6d_numerical(tmp4c)
				#
				if w0==w3 and w1==w4 and w2==w5:
					if len(tmp1a)==1:
						tmp1a=tmp4b.reshape(72)
					else:
						tmp1a=np.append(tmp1a,tmp4b)
				else:
					#tmp4c=simplification(tmp4c,5,0,0,2,0,0,1)
					#tmp4d=object_subtraction_2(tmp4b,tmp4c,verbose_level)
					#tmp4c=object_subtraction_2(tmp4b,tmp4d,verbose_level)
					#
					#tmp4d=object_subtraction_dev(tmp4b,tmp4c,obj,verbose)
					#tmp4c=object_subtraction_dev(tmp4b,tmp4d,obj,verbose)
					if len(tmp1a)==1:
						tmp1a=tmp4c.reshape(len(tmp4c)*72)
					else:
						tmp1a=np.append(tmp1a,tmp4c)
			else:
				pass
		tmp4a=tmp1a.reshape(len(tmp1a)/72,4,6,3)
		v3,v4,v5=obj_volume_6d(tmp4a)
		#vol2=(v3+TAU*v4)/float(v5)
		#vol3=obj_volume_6d_numerical(tmp4a)
		print '         volume of reduced obj: %d %d %d (%10.8f)'%(v3,v4,v5,vol3)
		if v3==v0 and v4==v1 and v5==v2 or abs(vol3-vol1)<vol3*EPS:
			print '         succdeded'
			return tmp4a
		else:
			if abs(vol1-vol2)<1e-8:
				print '         succdeded'
				return tmp4a
			else: 
				print '         fail, initial obj returned'
				return obj

cdef list read_file(file):
	cdef list line
	try:
		f=open(file,'r')
	except IOError, e:
		print e
		sys.exit(0)
	line=[]
	while 1:
		a=f.readline()
		if not a:
			break
		line.append(a[:-1])
	return line

cpdef int make_tmp_pod(np.ndarray[np.int64_t, ndim=4] pod, int number):
	generator_xyz_dim4_tetrahedron(pod,'tmp%d'%(number),0)
	return 0

cpdef np.ndarray read_tmp_pod(int number):
	cdef int a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5,a6,b6,c6
	cdef int i,num
	cdef np.ndarray[np.int64_t, ndim=1] tmp1
		
	file_name='./tmp%d.xyz'%(number)
	f1=read_file(file_name)
	f0=f1[0].split()
	num=int(f0[0])
	for i in range(2,num+2):
		fi=f1[i]
		fi=fi.split()
		a1=int(fi[10])
		b1=int(fi[11])
		c1=int(fi[12])
		a2=int(fi[13])
		b2=int(fi[14])
		c2=int(fi[15])
		a3=int(fi[16])
		b3=int(fi[17])
		c3=int(fi[18])
		a4=int(fi[19])
		b4=int(fi[20])
		c4=int(fi[21])
		a5=int(fi[22])
		b5=int(fi[23])
		c5=int(fi[24])
		a6=int(fi[25])
		b6=int(fi[26])
		c6=int(fi[27])
		if i==2:
			tmp1=np.array([a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5,a6,b6,c6])
		else:
			tmp1=np.append(tmp1,[a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5,a6,b6,c6])
	return tmp1.reshape(len(tmp1)/72,4,6,3)

cdef np.ndarray do_merge_tetrahedra_in_obj(np.ndarray[np.int64_t, ndim=4] obj, int numbre_of_cycle, int numbre_of_tetrahedron, int verbose):
	cdef int i1
	cdef np.ndarray[np.int64_t, ndim=4] tmp4a,tmp4b
	
	tmp4a=np.array([0]).reshape(1,1,1,1)
	tmp4a=obj
	if len(obj)>=numbre_of_tetrahedron:
		# merging two tetrahedra
		for i1 in range(numbre_of_cycle):
			#
			if numbre_of_tetrahedron==2:
				tmp4b=merge_2_tetrahedra_in_obj(tmp4a,verbose)
			elif numbre_of_tetrahedron==3:
				tmp4b=merge_3_tetrahedra_in_obj(tmp4a,verbose)
			elif  numbre_of_tetrahedron==4:
				tmp4b=merge_4_tetrahedra_in_obj(tmp4a,verbose)
			else:
				pass
			#
			if tmp4b.tolist()!=[[[[0]]]]:
				if verbose>=2:
					print '		simplification: %d -> %d'%(len(tmp4a),len(tmp4b))
				else:
					pass
				if len(obj)==len(tmp4b) or len(tmp4b)<numbre_of_tetrahedron:
					tmp4a=tmp4b
					break
				else:
					tmp4a=tmp4b
			else:
				if verbose>=2:
					print '	  simplification: fail'
					print '	  return previous obj'
				else:
					pass
				break
		if len(tmp4a)==len(tmp4b) or len(tmp4b)<numbre_of_tetrahedron:
			if verbose>=1:
				print '		succeed: %d to %d'%(len(obj),len(tmp4b))
			else:
				pass
			tmp4a=tmp4b
		else:
			pass
		return tmp4a
	else:
		return np.array([0]).reshape(1,1,1,1)

cdef np.ndarray shuffle_obj(np.ndarray[np.int64_t, ndim=4] obj):	
	cdef int i1
	cdef list a
	cdef np.ndarray[np.int64_t, ndim=1] index_list,tmp1
	cdef np.ndarray[np.int64_t, ndim=4] obj_new
	
	index_list = np.arange(len(obj))
	np.random.shuffle(index_list)
	a=index_list.tolist()
	
	for i1 in range(len(a)):
		if i1==0:
			tmp1=obj[a[0]].reshape(72)
		else:
			tmp1=np.append(tmp1,obj[a[i1]])
	obj_new=tmp1.reshape(len(tmp1)/72,4,6,3)
	return obj_new

cpdef np.ndarray simplification(np.ndarray[np.int64_t, ndim=4] obj, int num2_of_cycle, int num3_of_cycle, int num4_of_cycle, int num2_of_shuffle, int num3_of_shuffle, int num4_of_shuffle, int verbose_level):
	cdef int i1,index_min
	cdef list b
	cdef np.ndarray[np.int64_t, ndim=1] index_list,tmp1
	cdef np.ndarray[np.int64_t, ndim=4] obj1,obj2,obj_new
	
	# verbose_level  0 > 4 (no comment, verbose)
	if verbose_level>=1:
		print '	simplification()'
	else:
		pass
	
	# objに収納されている4面体をまとめる操作は、4面体の順番に大きく影響される.
	# 順番をランダムに変え、最も単純化できたものを採用する.
	
	if num2_of_shuffle>0:
		pass
	else:
		num2_of_shuffle=1
	
	b=[]
	for i1 in range(num2_of_shuffle):
		obj1=shuffle_obj(obj)
		obj2=do_merge_tetrahedra_in_obj(obj1,num2_of_cycle,2,verbose_level)
		if len(obj2)!=1:
			obj_new=obj2
		else:
			obj_new=obj1
		make_tmp_pod(obj_new,i1)
		b.append(len(obj_new))
	index_min=b.index(min(b))
	obj_min=read_tmp_pod(index_min)
	
	if num3_of_shuffle>0:
		b=[]
		for i1 in range(num3_of_shuffle):
			#obj1=shuffle_obj(obj)
			obj1=shuffle_obj(obj_min)
			obj2=do_merge_tetrahedra_in_obj(obj1,num3_of_cycle,3,verbose_level)
			if len(obj2)!=1:
				obj_new=obj2
			else:
				obj_new=obj1
			make_tmp_pod(obj_new,i1)
			b.append(len(obj_new))
		index_min=b.index(min(b))
		obj_min=read_tmp_pod(index_min)
	else:
		pass
	
	if num4_of_shuffle>0:
		b=[]
		for i1 in range(num4_of_shuffle):
			#obj1=shuffle_obj(obj)
			obj1=shuffle_obj(obj_min)
			obj2=do_merge_tetrahedra_in_obj(obj1,num4_of_cycle,4,verbose_level)
			if len(obj2)!=1:
				obj_new=obj2
			else:
				obj_new=obj1
			make_tmp_pod(obj_new,i1)
			b.append(len(obj_new))
		index_min=b.index(min(b))
		obj_min=read_tmp_pod(index_min)
	else:
		pass
	
	return obj_min

cpdef int on_out_surface(np.ndarray[np.int64_t, ndim=2] point,np.ndarray[np.int64_t, ndim=3] triangle):
	
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

cpdef double length_3d(np.ndarray[np.int64_t, ndim=2] vec):
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
#"""
# Numerical version
cpdef int point_on_segment(np.ndarray[np.int64_t, ndim=2] point, np.ndarray[np.int64_t, ndim=2] lineA, np.ndarray[np.int64_t, ndim=2] lineB):
	# judge whether a point is on a line segment, A-B, or not.
	# http://marupeke296.com/COL_2D_No2_PointToLine.html
	#
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
	#if lBA>EPS and abs(inner_product_numerical(vecPA,vecBA)-lPA*lBA)<EPS:
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
"""
# TAU-style version
cpdef int point_on_segment(np.ndarray[np.int64_t, ndim=2] point, np.ndarray[np.int64_t, ndim=2] lineA, np.ndarray[np.int64_t, ndim=2] lineB):
	# judge whether a point is on a line segment, A-B, or not.
	# http://marupeke296.com/COL_2D_No2_PointToLine.html
	#
	cdef long a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4
	cdef double val1,val2,lPA,lBA
	cdef np.ndarray[np.int64_t, ndim=1] m1,m2,m3
	cdef np.ndarray[np.int64_t, ndim=1] x0,y0,z0,x1,y1,z1,x2,y2,z2
	cdef np.ndarray[np.int64_t, ndim=2] vecPA,vecPB
	#
	m1,m2,m3,x0,y0,z0=projection(point[0],point[1],point[2],point[3],point[4],point[5])
	m1,m2,m3,x1,y1,z1=projection(lineA[0],lineA[1],lineA[2],lineA[3],lineA[4],lineA[5])
	m1,m2,m3,x2,y2,z2=projection(lineB[0],lineB[1],lineB[2],lineB[3],lineB[4],lineB[5])
	#
	a1,b1,c1=sub(x0[0],x0[1],x0[2],x1[0],x1[1],x1[2])
	a2,b2,c2=sub(y0[0],y0[1],y0[2],y1[0],y1[1],y1[2])
	a3,b3,c3=sub(z0[0],z0[1],z0[2],z1[0],z1[1],z1[2])
	vecPA=np.array([a1,b1,c1,a2,b2,c2,a3,b3,c3]).reshape(3,3)
	#
	a1,b1,c1=sub(x2[0],x2[1],x2[2],x1[0],x1[1],x1[2])
	a2,b2,c2=sub(y2[0],y2[1],y2[2],y1[0],y1[1],y1[2])
	a3,b3,c3=sub(z2[0],z2[1],z2[2],z1[0],z1[1],z1[2])
	vecBA=np.array([a1,b1,c1,a2,b2,c2,a3,b3,c3]).reshape(3,3)
	#
	lPA=length_3d(vecPA)
	lBA=length_3d(vecBA)
	#
	a4,b4,c4=inner_product(vecPA,vecBA)
	val1=(a4+b4*TAU)/float(c4)
	#
	if lBA>EPS and abs(val1-lPA*lBA)<EPS:
	#if lBA>0.0 and abs(val1-lPA*lBA)<EPS:
		val2=lPA/lBA
		if val2>=0.0 and val2<=1.0:
			return 0
		else:
			return 1
	else:
		return 2
"""
cpdef double triangle_area(np.ndarray[np.int64_t, ndim=2] v1,\
							np.ndarray[np.int64_t, ndim=2] v2,\
							np.ndarray[np.int64_t, ndim=2] v3):
	cdef double vx0,vx1,vx2,vy0,vy1,vy2,vz0,vz1,vz2,area
	#cdef int a1,a2,a3,b1,b2,b3,c1,c2,c3,d1,d2,d3
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
	#area=np.cross(vec2,vec1)/2.0  # cross product
	area=abs(area)
	return area
						
cpdef list tetrahedron_volume(np.ndarray[np.int64_t, ndim=2] v1,\
							np.ndarray[np.int64_t, ndim=2] v2,\
							np.ndarray[np.int64_t, ndim=2] v3,\
							np.ndarray[np.int64_t, ndim=2] v4):
	# This function returns volume of a tetrahedron
	# input: vertex coordinates of the tetrahedron (x0,y0,z0),(x1,y1,z1),(x2,y2,z2),(x3,y3,z3)
	#cdef int a1,a2,a3,b1,b2,b3,c1,c2,c3
	cdef long a1,a2,a3,b1,b2,b3,c1,c2,c3
	cdef np.ndarray[np.int64_t, ndim=1] x0,x1,x2,x3,y0,y1,y2,y3,z0,z1,z2,z3
	cdef np.ndarray[np.int64_t, ndim=2] a,b,c
	#
	#print 'tetrahedron_volume()'
	
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
	#print 'a =',a
	#
	[a1,a2,a3]=sub(x2[0],x2[1],x2[2],x0[0],x0[1],x0[2])
	[b1,b2,b3]=sub(y2[0],y2[1],y2[2],y0[0],y0[1],y0[2])
	[c1,c2,c3]=sub(z2[0],z2[1],z2[2],z0[0],z0[1],z0[2])
	b=np.array([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]])
	#print 'b =',b
	#
	[a1,a2,a3]=sub(x3[0],x3[1],x3[2],x0[0],x0[1],x0[2])
	[b1,b2,b3]=sub(y3[0],y3[1],y3[2],y0[0],y0[1],y0[2])
	[c1,c2,c3]=sub(z3[0],z3[1],z3[2],z0[0],z0[1],z0[2])
	c=np.array([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]])
	#print 'c=',c
	#
	[a1,a2,a3]=det_matrix(a,b,c) # determinant of 3x3 matrix
	#print 'a1,a2,a3 =',a1,a2,a3,(a1+TAU*a2)/float(a3)
	#[a1,a2,a3]=mul(a1,a2,a3,1,0,6)
	#return [a1,a2,a3]
	#
	# avoid a negative value
	#
	if a1+a2*TAU<0.0: # to avoid negative volume...
		[a1,a2,a3]=mul(a1,a2,a3,-1,0,6)
	else:
		[a1,a2,a3]=mul(a1,a2,a3,1,0,6)
	return [a1,a2,a3]
	"""
	[b1,b2,b3]=tau_to_sqrt5(a1,a2,a3)
	if b1!=0 and b2!=0:
		if b1*b1>5*b2*b2 and (b1*b2>0):
			return [-a1,-a2,a3]
		else:
			return [a1,a2,a3]
	elif b1!=0 and b2==0:
		if b1>0:
			return [a1,a2,a3]
		else:
			return [-a1,-a2,a3]
	elif b1==0 and b2!=0:
		if b2>0:
			return [a1,a2,a3]
		else:
			return [-a1,-a2,a3]
	else:
		return [a1,a2,a3]
	"""

cpdef list tau_to_sqrt5(int a,int b,int c):
	# TAU-tyle to sqrt5-style
	return [2*a+b,b,2*c]

cpdef list det_matrix(np.ndarray[np.int64_t, ndim=2] a, np.ndarray[np.int64_t, ndim=2] b, np.ndarray[np.int64_t, ndim=2] c):
	# determinant of 3x3 matrix in TAU style
	cdef long t1,t2,t3,t4,t5,t6,t7,t8,t9
	#cdef int a1,a2,a3,b1,b2,b3
	#
	#print 'det_matrix()'
	
	[t1,t2,t3]=mul(a[0][0],a[0][1],a[0][2],b[1][0],b[1][1],b[1][2])
	[t1,t2,t3]=mul(t1,t2,t3,c[2][0],c[2][1],c[2][2])
	#print ' (1) t1,t2,t3 =',t1,t2,t3,(t1+TAU*t2)/float(t3)
	#		
	[t4,t5,t6]=mul(a[2][0],a[2][1],a[2][2],b[0][0],b[0][1],b[0][2])
	[t4,t5,t6]=mul(t4,t5,t6,c[1][0],c[1][1],c[1][2])
	#print ' (2) t4,t5,t6 =',t4,t5,t6,(t4+TAU*t5)/float(t6)
	
	#
	[t7,t8,t9]=add(t1,t2,t3,t4,t5,t6)
	#print ' (3) t7,t8,t9 =',t7,t8,t9,(t7+TAU*t8)/float(t9)
	#
	[t1,t2,t3]=mul(a[1][0],a[1][1],a[1][2],b[2][0],b[2][1],b[2][2])
	[t1,t2,t3]=mul(t1,t2,t3,c[0][0],c[0][1],c[0][2])
	#print ' (4) t1,t2,t3 =',t1,t2,t3,(t1+TAU*t2)/float(t3)
	#
	[t7,t8,t9]=add(t7,t8,t9,t1,t2,t3)
	#print ' (5) t7,t8,t9 =',t7,t8,t9,(t7+TAU*t8)/float(t9)
	#
	[t1,t2,t3]=mul(a[2][0],a[2][1],a[2][2],b[1][0],b[1][1],b[1][2])
	[t1,t2,t3]=mul(t1,t2,t3,c[0][0],c[0][1],c[0][2])
	#print ' (6) t1,t2,t3 =',t1,t2,t3,(t1+TAU*t2)/float(t3)
	#
	[t7,t8,t9]=sub(t7,t8,t9,t1,t2,t3)
	#print ' (7) t7,t8,t9 =',t7,t8,t9,(t7+TAU*t8)/float(t9)
	#
	[t1,t2,t3]=mul(a[1][0],a[1][1],a[1][2],b[0][0],b[0][1],b[0][2])
	[t1,t2,t3]=mul(t1,t2,t3,c[2][0],c[2][1],c[2][2])
	#print ' (8) t1,t2,t3 =',t1,t2,t3,(t1+TAU*t2)/float(t3)
	#
	[t7,t8,t9]=sub(t7,t8,t9,t1,t2,t3)
	#print ' (9) t7,t8,t9 =',t7,t8,t9,(t7+TAU*t8)/float(t9)
	#
	[t1,t2,t3]=mul(a[0][0],a[0][1],a[0][2],b[2][0],b[2][1],b[2][2])
	[t1,t2,t3]=mul(t1,t2,t3,c[1][0],c[1][1],c[1][2])
	#print '(10) t1,t2,t3 =',t1,t2,t3,(t1+TAU*t2)/float(t3)
	#
	[t7,t8,t9]=sub(t7,t8,t9,t1,t2,t3)
	#print '(11) t7,t8,t9 =',t7,t8,t9,(t7+TAU*t8)/float(t9)
	#
	return [t7,t8,t9]

"""
cpdef int inside_outside_tetrahedron(np.ndarray[np.int64_t, ndim=2] point,\
									np.ndarray[np.int64_t, ndim=2] tetrahedron_v1,\
									np.ndarray[np.int64_t, ndim=2] tetrahedron_v2,\
									np.ndarray[np.int64_t, ndim=2] tetrahedron_v3,\
									np.ndarray[np.int64_t, ndim=2] tetrahedron_v4):
	# this function judges whether the point is inside a traiangle or not
	#
	# Attension
	# overflow when having very small tetrahedron volume.
	#
	cdef int verbose
	cdef long a1,a2,a3,b1,b2,b3,c1,c2,c3
	cdef np.ndarray[np.int64_t, ndim=1] m1,m2,m3,m4,m5,m6
	cdef np.ndarray[np.int64_t, ndim=2] v1,v2,v3,v4,p0,p1,p2,p3,p4
	#
	
	verbose=0
	
	if verbose>=2:
		print '                   inside_outside_tetrahedron()'
	else:
		pass
	
	m1,m2,m3,m4,m5,m6=projection(point[0],point[1],point[2],point[3],point[4],point[5])
	p0=np.array([m4,m5,m6])
	#print ' Xx %8.6f %8.6f %8.6f'%((m4[0]+TAU*m4[1])/float(m4[2]),(m5[0]+TAU*m5[1])/float(m5[2]),(m6[0]+TAU*m6[1])/float(m6[2]))
	#
	m1,m2,m3,m4,m5,m6=projection(tetrahedron_v1[0],tetrahedron_v1[1],tetrahedron_v1[2],tetrahedron_v1[3],tetrahedron_v1[4],tetrahedron_v1[5])
	p1=np.array([m4,m5,m6])
	#print ' Yy %8.6f %8.6f %8.6f'%((m4[0]+TAU*m4[1])/float(m4[2]),(m5[0]+TAU*m5[1])/float(m5[2]),(m6[0]+TAU*m6[1])/float(m6[2]))
	#
	m1,m2,m3,m4,m5,m6=projection(tetrahedron_v2[0],tetrahedron_v2[1],tetrahedron_v2[2],tetrahedron_v2[3],tetrahedron_v2[4],tetrahedron_v2[5])
	p2=np.array([m4,m5,m6])
	#print ' Yy %8.6f %8.6f %8.6f'%((m4[0]+TAU*m4[1])/float(m4[2]),(m5[0]+TAU*m5[1])/float(m5[2]),(m6[0]+TAU*m6[1])/float(m6[2]))
	#
	m1,m2,m3,m4,m5,m6=projection(tetrahedron_v3[0],tetrahedron_v3[1],tetrahedron_v3[2],tetrahedron_v3[3],tetrahedron_v3[4],tetrahedron_v3[5])
	p3=np.array([m4,m5,m6])
	#print ' Yy %8.6f %8.6f %8.6f'%((m4[0]+TAU*m4[1])/float(m4[2]),(m5[0]+TAU*m5[1])/float(m5[2]),(m6[0]+TAU*m6[1])/float(m6[2]))
	#
	m1,m2,m3,m4,m5,m6=projection(tetrahedron_v4[0],tetrahedron_v4[1],tetrahedron_v4[2],tetrahedron_v4[3],tetrahedron_v4[4],tetrahedron_v4[5])
	p4=np.array([m4,m5,m6])
	#print ' Yy %8.6f %8.6f %8.6f'%((m4[0]+TAU*m4[1])/float(m4[2]),(m5[0]+TAU*m5[1])/float(m5[2]),(m6[0]+TAU*m6[1])/float(m6[2]))
	#
	# volume 0
	[a1,a2,a3]=tetrahedron_volume(p1,p2,p3,p4) # volume 0
	#print 'volume 0 = %3d %3d %3d'%(a1,a2,a3)
	# volume 1
	[b1,b2,b3]=tetrahedron_volume(p0,p2,p3,p4) # volume 1
	#print 'volume 1 = %3d %3d %3d'%(b1,b2,b3)
	# volume 2
	[c1,c2,c3]=tetrahedron_volume(p1,p0,p3,p4)
	[b1,b2,b3]=add(b1,b2,b3,c1,c2,c3) # volume 1+volume 2
	#print 'volume 2 = %3d %3d %3d'%(c1,c2,c3)
	# volume 3
	[c1,c2,c3]=tetrahedron_volume(p1,p2,p0,p4)
	[b1,b2,b3]=add(b1,b2,b3,c1,c2,c3) # volume 1+volume 2+volume 3
	#print 'volume 3 = %3d %3d %3d'%(c1,c2,c3)
	# volume 4
	[c1,c2,c3]=tetrahedron_volume(p1,p2,p3,p0)
	[b1,b2,b3]=add(b1,b2,b3,c1,c2,c3) # vol_sum = volume 1+volume 2+volume 3+volume 4
	#print 'volume 4 = %3d %3d %3d'%(c1,c2,c3)
	#print 'volume_sum',(b1+b2*TAU)/float(b3)
	#print 'volume0',(a1+a2*TAU)/float(a3)
	#
	if a1==b1 and a2==b2 and a3==b3: # if vol_sum = volume 0
		return 0 # inside
	else:
		return 1 # outside

# test 
cpdef int inside_outside_tetrahedron(np.ndarray[np.int64_t, ndim=2] point,\
									np.ndarray[np.int64_t, ndim=2] tetrahedron_v1,\
									np.ndarray[np.int64_t, ndim=2] tetrahedron_v2,\
									np.ndarray[np.int64_t, ndim=2] tetrahedron_v3,\
									np.ndarray[np.int64_t, ndim=2] tetrahedron_v4):
	#
	# this function judges whether the point is inside a traiangle or not
	#
	# Attension
	# overflow when having very small tetrahedron volume.
	#
	cdef int verbose
	cdef long a1,a2,a3,b1,b2,b3,c1,c2,c3
	cdef long t1,t2,t3,t4,t5,t6,t7,t8,t9
	cdef np.ndarray[np.int64_t, ndim=1] m1,m2,m3,m4,m5,m6
	cdef np.ndarray[np.int64_t, ndim=2] v1,v2,v3,v4,p0,p1,p2,p3,p4	
	
	verbose=2
	
	if verbose>=2:
		print '                   inside_outside_tetrahedron()'
	else:
		pass
	
	m1,m2,m3,m4,m5,m6=projection(point[0],point[1],point[2],point[3],point[4],point[5])
	p0=np.array([m4,m5,m6])
	#
	m1,m2,m3,m4,m5,m6=projection(tetrahedron_v1[0],tetrahedron_v1[1],tetrahedron_v1[2],tetrahedron_v1[3],tetrahedron_v1[4],tetrahedron_v1[5])
	p1=np.array([m4,m5,m6])
	#
	m1,m2,m3,m4,m5,m6=projection(tetrahedron_v2[0],tetrahedron_v2[1],tetrahedron_v2[2],tetrahedron_v2[3],tetrahedron_v2[4],tetrahedron_v2[5])
	p2=np.array([m4,m5,m6])
	#
	m1,m2,m3,m4,m5,m6=projection(tetrahedron_v3[0],tetrahedron_v3[1],tetrahedron_v3[2],tetrahedron_v3[3],tetrahedron_v3[4],tetrahedron_v3[5])
	p3=np.array([m4,m5,m6])
	#
	m1,m2,m3,m4,m5,m6=projection(tetrahedron_v4[0],tetrahedron_v4[1],tetrahedron_v4[2],tetrahedron_v4[3],tetrahedron_v4[4],tetrahedron_v4[5])
	p4=np.array([m4,m5,m6])
	
	# volume 0
	[a1,a2,a3]=tetrahedron_volume(p1,p2,p3,p4) # volume 0
	if verbose>=2:
		print '                    volume 0: %d %d %d (%8.6f)'%(a1,a2,a3,(a1+TAU*a2)/float(a3))
	else:
		pass
	
	# volume 1
	[b1,b2,b3]=tetrahedron_volume(p0,p2,p3,p4) # volume 1
	if verbose>=2:
		print 'p0=np.array([[%d,%d,%d],[%d,%d,%d],[%d,%d,%d]])'%(p0[0][0],p0[0][1],p0[0][2],p0[1][0],p0[1][1],p0[1][2],p0[2][0],p0[2][1],p0[2][2])
		print 'p2=np.array([[%d,%d,%d],[%d,%d,%d],[%d,%d,%d]])'%(p2[0][0],p2[0][1],p2[0][2],p2[1][0],p2[1][1],p2[1][2],p2[2][0],p2[2][1],p2[2][2])
		print 'p3=np.array([[%d,%d,%d],[%d,%d,%d],[%d,%d,%d]])'%(p3[0][0],p3[0][1],p3[0][2],p3[1][0],p3[1][1],p3[1][2],p3[2][0],p3[2][1],p3[2][2])
		print 'p4=np.array([[%d,%d,%d],[%d,%d,%d],[%d,%d,%d]])'%(p4[0][0],p4[0][1],p4[0][2],p4[1][0],p4[1][1],p4[1][2],p4[2][0],p4[2][1],p4[2][2])
		print '                    volume 1: %d %d %d (%8.6f)'%(b1,b2,b3,(b1+TAU*b2)/float(b3))
	else:
		pass

	# volume 2
	[c1,c2,c3]=tetrahedron_volume(p1,p0,p3,p4)
	[b1,b2,b3]=add(b1,b2,b3,c1,c2,c3) # volume 1+volume 2
	if verbose>=2:
		print '                    volume 2: %d %d %d (%8.6f)'%(c1,c2,c3,(c1+TAU*c2)/float(c3))
	else:
		pass

	# volume 3
	[c1,c2,c3]=tetrahedron_volume(p1,p2,p0,p4)
	[b1,b2,b3]=add(b1,b2,b3,c1,c2,c3) # volume 1+volume 2+volume 3
	if verbose>=2:
		print '                    volume 3: %d %d %d (%8.6f)'%(c1,c2,c3,(c1+TAU*c2)/float(c3))
	else:
		pass

	# volume 4
	[c1,c2,c3]=tetrahedron_volume(p1,p2,p3,p0)
	[b1,b2,b3]=add(b1,b2,b3,c1,c2,c3) # vol_sum = volume 1+volume 2+volume 3+volume 4
	if verbose>=2:
		print '                    volume 4: %d %d %d (%8.6f)'%(c1,c2,c3,(c1+TAU*c2)/float(c3))
	else:
		pass
	
	if verbose>=2:
		print '                         sum: %d %d %d (%8.6f)'%(b1,b2,b3,(b1+TAU*b2)/float(b3))
	else:
		pass
		
	if a1==b1 and a2==b2 and a3==b3: # if vol_sum = volume 0
		return 0 # inside
	else:
		return 1 # outside

# new
# Attension
# overflow when having very small tetrahedron volume.
cpdef int inside_outside_tetrahedron_new(np.ndarray[np.int64_t, ndim=2] point,\
									np.ndarray[np.int64_t, ndim=2] tetrahedron_v1,\
									np.ndarray[np.int64_t, ndim=2] tetrahedron_v2,\
									np.ndarray[np.int64_t, ndim=2] tetrahedron_v3,\
									np.ndarray[np.int64_t, ndim=2] tetrahedron_v4):
	# this function judges whether the point is inside a traiangle or not
	#cdef int v01,v02,v03,v1,v2,v3,w1,w2,w3
	cdef long v01,v02,v03,v1,v2,v3,w1,w2,w3
	cdef np.ndarray[np.int64_t, ndim=1] tmp1
	cdef np.ndarray[np.int64_t, ndim=3] tmp3
	#
	
	# tetrahedron 0, volume0
	tmp1=np.append(tetrahedron_v1,tetrahedron_v2)
	tmp1=np.append(tmp1,tetrahedron_v3)
	tmp1=np.append(tmp1,tetrahedron_v4)
	tmp3=tmp1.reshape(4,6,3)
	[v01,v02,v03]=tetrahedron_volume_6d(tmp3)
	# tetrahedron 1, volume1
	tmp1=np.append(point,tetrahedron_v1)
	tmp1=np.append(tmp1,tetrahedron_v2)
	tmp1=np.append(tmp1,tetrahedron_v3)
	tmp3=tmp1.reshape(4,6,3)
	[v1,v2,v3]=tetrahedron_volume_6d(tmp3)
	# tetrahedron 2, volume2
	tmp1=np.append(point,tetrahedron_v1)
	tmp1=np.append(tmp1,tetrahedron_v3)
	tmp1=np.append(tmp1,tetrahedron_v4)
	tmp3=tmp1.reshape(4,6,3)
	[w1,w2,w3]=tetrahedron_volume_6d(tmp3)
	v1,v2,v3=add(v1,v2,v3,w1,w2,w3)
	# tetrahedron 3, volume3
	tmp1=np.append(point,tetrahedron_v1)
	tmp1=np.append(tmp1,tetrahedron_v2)
	tmp1=np.append(tmp1,tetrahedron_v4)
	tmp3=tmp1.reshape(4,6,3)
	[w1,w2,w3]=tetrahedron_volume_6d(tmp3)
	v1,v2,v3=add(v1,v2,v3,w1,w2,w3)
	# tetrahedron 4, volume4
	tmp1=np.append(point,tetrahedron_v2)
	tmp1=np.append(tmp1,tetrahedron_v3)
	tmp1=np.append(tmp1,tetrahedron_v4)
	tmp3=tmp1.reshape(4,6,3)
	[w1,w2,w3]=tetrahedron_volume_6d(tmp3)
	v1,v2,v3=add(v1,v2,v3,w1,w2,w3)
	
	# if volume0-(volume1+volume2+volume3+volume4)=0
	v1,v2,v3=sub(v01,v02,v03,v1,v2,v3)
	if v1==0 and v2==0:
		return 0 # inside
	else:
		return 1 # outside
"""

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
	#print 'detm=',detm
	vol = abs(detm)/6.0
	return vol

"""
# TAU-style version
# 四面体が小さい場合、正しく体積が求められない

#cpdef inside_outside_tetrahedron_numerical(double x0, double y0, double z0, double x1, double y1, double z1, double x2, double y2, double z2, double x3, double y3, double z3, double x4, double y4, double z4):
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
		
	m1,m2,m3,m4,m5,m6=projection(point[0],point[1],point[2],point[3],point[4],point[5])
	x0=(m4[0]+TAU*m4[1])/float(m4[2])
	y0=(m5[0]+TAU*m5[1])/float(m5[2])
	z0=(m6[0]+TAU*m6[1])/float(m6[2])
	m1,m2,m3,m4,m5,m6=projection(tetrahedron_v1[0],tetrahedron_v1[1],tetrahedron_v1[2],tetrahedron_v1[3],tetrahedron_v1[4],tetrahedron_v1[5])
	x1=(m4[0]+TAU*m4[1])/float(m4[2])
	y1=(m5[0]+TAU*m5[1])/float(m5[2])
	z1=(m6[0]+TAU*m6[1])/float(m6[2])
	m1,m2,m3,m4,m5,m6=projection(tetrahedron_v2[0],tetrahedron_v2[1],tetrahedron_v2[2],tetrahedron_v2[3],tetrahedron_v2[4],tetrahedron_v2[5])
	x2=(m4[0]+TAU*m4[1])/float(m4[2])
	y2=(m5[0]+TAU*m5[1])/float(m5[2])
	z2=(m6[0]+TAU*m6[1])/float(m6[2])
	m1,m2,m3,m4,m5,m6=projection(tetrahedron_v3[0],tetrahedron_v3[1],tetrahedron_v3[2],tetrahedron_v3[3],tetrahedron_v3[4],tetrahedron_v3[5])
	x3=(m4[0]+TAU*m4[1])/float(m4[2])
	y3=(m5[0]+TAU*m5[1])/float(m5[2])
	z3=(m6[0]+TAU*m6[1])/float(m6[2])
	m1,m2,m3,m4,m5,m6=projection(tetrahedron_v4[0],tetrahedron_v4[1],tetrahedron_v4[2],tetrahedron_v4[3],tetrahedron_v4[4],tetrahedron_v4[5])
	x4=(m4[0]+TAU*m4[1])/float(m4[2])
	y4=(m5[0]+TAU*m5[1])/float(m5[2])
	z4=(m6[0]+TAU*m6[1])/float(m6[2])	
	volume0 = tetrahedron_volume_numerical(x1,y1,z1, x2,y2,z2, x3,y3,z3, x4,y4,z4)
	volume1 = tetrahedron_volume_numerical(x0,y0,z0, x2,y2,z2, x3,y3,z3, x4,y4,z4)
	volume2 = tetrahedron_volume_numerical(x1,y1,z1, x0,y0,z0, x3,y3,z3, x4,y4,z4)
	volume3 = tetrahedron_volume_numerical(x1,y1,z1, x2,y2,z2, x0,y0,z0, x4,y4,z4)
	volume4 = tetrahedron_volume_numerical(x1,y1,z1, x2,y2,z2, x3,y3,z3, x0,y0,z0)
	#if abs(volume0-volume1-volume2-volume3-volume4)<EPS:	
	#if abs(volume0-volume1-volume2-volume3-volume4)<1e-8:
	if abs(volume0-volume1-volume2-volume3-volume4)<EPS*volume0:
	#if abs(volume0-volume1-volume2-volume3-volume4)<1e-6*volume0:
	#if abs(volume0-volume1-volume2-volume3-volume4)<1e-5*volume0:
		return 0 # inside
	else:
		print 'diff1',abs(volume0-volume1-volume2-volume3-volume4)
		print 'diff2',EPS*volume0
		return 1 # outside
"""

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
	
	tetrahedron0=np.append(tetrahedron_v1,tetrahedron_v2)
	tetrahedron0=np.append(tetrahedron0,tetrahedron_v3)
	tetrahedron0=np.append(tetrahedron0,tetrahedron_v4)
	tetrahedron0=tetrahedron0.reshape(4,6,3)
	volume0=tetrahedron_volume_6d_numerical(tetrahedron0)
	
	tetrahedron1=np.append(point,tetrahedron_v2)
	tetrahedron1=np.append(tetrahedron1,tetrahedron_v3)
	tetrahedron1=np.append(tetrahedron1,tetrahedron_v4)
	tetrahedron1=tetrahedron1.reshape(4,6,3)
	volume1=tetrahedron_volume_6d_numerical(tetrahedron1)
		
	tetrahedron2=np.append(tetrahedron_v1,point)
	tetrahedron2=np.append(tetrahedron2,tetrahedron_v3)
	tetrahedron2=np.append(tetrahedron2,tetrahedron_v4)
	tetrahedron2=tetrahedron2.reshape(4,6,3)
	volume2=tetrahedron_volume_6d_numerical(tetrahedron2)
	
	tetrahedron3=np.append(tetrahedron_v1,tetrahedron_v2)
	tetrahedron3=np.append(tetrahedron3,point)
	tetrahedron3=np.append(tetrahedron3,tetrahedron_v4)
	tetrahedron3=tetrahedron3.reshape(4,6,3)
	volume3=tetrahedron_volume_6d_numerical(tetrahedron3)
	
	tetrahedron4=np.append(tetrahedron_v1,tetrahedron_v2)
	tetrahedron4=np.append(tetrahedron4,tetrahedron_v3)
	tetrahedron4=np.append(tetrahedron4,point)
	tetrahedron4=tetrahedron4.reshape(4,6,3)
	volume4=tetrahedron_volume_6d_numerical(tetrahedron4)
		
	"""
	m1,m2,m3,m4,m5,m6=projection(point[0],point[1],point[2],point[3],point[4],point[5])
	x0=(m4[0]+TAU*m4[1])/float(m4[2])
	y0=(m5[0]+TAU*m5[1])/float(m5[2])
	z0=(m6[0]+TAU*m6[1])/float(m6[2])
	m1,m2,m3,m4,m5,m6=projection(tetrahedron_v1[0],tetrahedron_v1[1],tetrahedron_v1[2],tetrahedron_v1[3],tetrahedron_v1[4],tetrahedron_v1[5])
	x1=(m4[0]+TAU*m4[1])/float(m4[2])
	y1=(m5[0]+TAU*m5[1])/float(m5[2])
	z1=(m6[0]+TAU*m6[1])/float(m6[2])
	m1,m2,m3,m4,m5,m6=projection(tetrahedron_v2[0],tetrahedron_v2[1],tetrahedron_v2[2],tetrahedron_v2[3],tetrahedron_v2[4],tetrahedron_v2[5])
	x2=(m4[0]+TAU*m4[1])/float(m4[2])
	y2=(m5[0]+TAU*m5[1])/float(m5[2])
	z2=(m6[0]+TAU*m6[1])/float(m6[2])
	m1,m2,m3,m4,m5,m6=projection(tetrahedron_v3[0],tetrahedron_v3[1],tetrahedron_v3[2],tetrahedron_v3[3],tetrahedron_v3[4],tetrahedron_v3[5])
	x3=(m4[0]+TAU*m4[1])/float(m4[2])
	y3=(m5[0]+TAU*m5[1])/float(m5[2])
	z3=(m6[0]+TAU*m6[1])/float(m6[2])
	m1,m2,m3,m4,m5,m6=projection(tetrahedron_v4[0],tetrahedron_v4[1],tetrahedron_v4[2],tetrahedron_v4[3],tetrahedron_v4[4],tetrahedron_v4[5])
	x4=(m4[0]+TAU*m4[1])/float(m4[2])
	y4=(m5[0]+TAU*m5[1])/float(m5[2])
	z4=(m6[0]+TAU*m6[1])/float(m6[2])	
	volume0 = tetrahedron_volume_numerical(x1,y1,z1, x2,y2,z2, x3,y3,z3, x4,y4,z4)
	volume1 = tetrahedron_volume_numerical(x0,y0,z0, x2,y2,z2, x3,y3,z3, x4,y4,z4)
	volume2 = tetrahedron_volume_numerical(x1,y1,z1, x0,y0,z0, x3,y3,z3, x4,y4,z4)
	volume3 = tetrahedron_volume_numerical(x1,y1,z1, x2,y2,z2, x0,y0,z0, x4,y4,z4)
	volume4 = tetrahedron_volume_numerical(x1,y1,z1, x2,y2,z2, x3,y3,z3, x0,y0,z0)
	"""
	#if abs(volume0-volume1-volume2-volume3-volume4)<EPS:	
	#if abs(volume0-volume1-volume2-volume3-volume4)<1e-8:
	if abs(volume0-volume1-volume2-volume3-volume4)<EPS*volume0:
	#if abs(volume0-volume1-volume2-volume3-volume4)<1e-6*volume0:
	#if abs(volume0-volume1-volume2-volume3-volume4)<1e-5*volume0:
		return 0 # inside
	else:
		#print 'diff1',abs(volume0-volume1-volume2-volume3-volume4)
		#print 'diff2',EPS*volume0
		return 1 # outside

cdef np.ndarray matrix_dot(np.ndarray array_1, np.ndarray array_2):
	cdef Py_ssize_t mx1,my1,mx2,my2
	cdef Py_ssize_t x,y,z	
	mx1 = array_1.shape[0]
	my1 = array_1.shape[1]
	mx2 = array_2.shape[0]
	my2 = array_2.shape[1]
	array_3 = np.zeros((mx1,my2), dtype=np.int)
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
	array_2 = np.identity(mx, dtype=np.int)
	if mx == my:
		if n == 0:
			return np.identity(mx, dtype=np.int)
		elif n<0:
			return np.zeros((mx,mx), dtype=np.int)
		else:
			for i in range(n):
				tmp = np.zeros((6, 6), dtype=np.int)
				for x in range(array_2.shape[0]):
					for y in range(array_1.shape[1]):
						for z in range(array_2.shape[1]):
							tmp[x][y] += array_2[x][z] * array_1[z][y]
				array_2 = tmp
			return array_2
	else:
		print 'ERROR: matrix has not regular shape'
		return 
"""
cpdef np.ndarray icosagmat():
#def icosagmat():
	# 6x6 matices of icosahedral symmetry operations
	# 1 x,z,t,u,v,y;
	# 2 -x,-y,-v,-u,-t,-z;
	# 3 -v,-y,t,z,-u,-x;
	# 4 y,z,x,v,-t,-u;
	# x,y,z,t,u,v -> identity operation
	cdef np.ndarray[np.int64_t, ndim=2] m1,m2,m3,m4,m5
	# c5
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
	return m1,m2,m3,m4,m5
"""

#cdef list icosasymop():
cpdef icosasymop():
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
"""
cdef class MyOperator:
	# A = (p1+p2*TAU)/p3
	# B = (q1+q2*TAU)/q3
	# 4つの演算(A+B, A-B, A*B, A/B)の解を(c1+c2*TAU)/c3の形で出力
	cdef int _p1, _p2, _p3, _q1, _q2, _q3
	def __init__(self,int p1,int p2,int p3,int q1,int q2,int q3):
		self._p1=p1
		self._p2=p2
		self._p3=p3
		self._q1=q1
		self._q2=q2
		self._q3=q3
	def add(self):#A+B
		cdef int c1,c2,c3,gcd
		cdef np.ndarray[np.int64_t,ndim=1] x
		c1=self._p1*self._q3 + self._q1*self._p3
		c2=self._p2*self._q3 + self._q2*self._p3
		c3=self._p3*self._q3
		x=np.array([c1,c2,c3], dtype=np.int)
		gcd=np.gcd.reduce(x)#c1,c2,c3の最大公約数
		if c3/gcd<0:
			return -c1/gcd,-c2/gcd,-c3/gcd
		else:
			return c1/gcd,c2/gcd,c3/gcd
	def sub(self):#A-B
		cdef int c1,c2,c3,gcd
		cdef np.ndarray[np.int64_t,ndim=1] x
		c1=self._p1*self._q3 - self._q1*self._p3
		c2=self._p2*self._q3 - self._q2*self._p3
		c3=self._p3*self._q3
		x=np.array([c1,c2,c3], dtype=np.int)
		gcd=np.gcd.reduce(x)#c1,c2,c3の最大公約数
		if c3/gcd<0:
			return -c1/gcd,-c2/gcd,-c3/gcd
		else:
			return c1/gcd,c2/gcd,c3/gcd
	def mul(self):#A*B
		cdef int c1,c2,c3,gcd
		cdef np.ndarray[np.int64_t,ndim=1] x
		c1=self._p1*self._q1 + self._p2*self._q2
		c2=self._p1*self._q2 + self._p2*self._q1 + self._p2*self._q2
		c3=self._p3*self._q3
		x=np.array([c1,c2,c3], dtype=np.int)
		gcd=np.gcd.reduce(x)#c1,c2,c3の最大公約数
		if c3/gcd<0:
			return -c1/gcd,-c2/gcd,-c3/gcd
		else:
			return c1/gcd,c2/gcd,c3/gcd
	def div(self):#A/B
		cdef int c1,c2,c3,gcd
		cdef np.ndarray[np.int64_t,ndim=1] x
		if self._q1==0 and self._q2==0:
			return 'ERROR: division error' 
		else:
			if self._q2!=0:
				if self._q1!=0:
					c1=(self._p1*self._q1 + self._p1*self._q2 - self._p2*self._q2)*self._q3
					c2=(self._p2*self._q1 - self._p1*self._q2)*self._q3
					c3=(self._q1*self._q1 - self._q2*self._q2 + self._q1*self._q2)*self._p3
				else:
					c1=(-self._p1+self._p2)*self._q3
					c2=self._p1*self._q3
					c3=self._q2*self._p3
			elif self._q1!=0:
				c1=self._p1*self._q3
				c2=self._p2*self._q3
				c3=self._q1*self._p3
			x=np.array([c1,c2,c3], dtype=np.int)
			gcd=np.gcd.reduce(x)#c1,c2,c3の最大公約数
			if c3/gcd<0:
				return -c1/gcd,-c2/gcd,-c3/gcd
			else:
				return c1/gcd,c2/gcd,c3/gcd
"""
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
	#v2e=mtrixcal(m3,m0,m0,m4,m0,m3,h1,h2,h3,h4,h5,h6) # tau,0,0,-tau,0,tau
	v2e=mtrixcal(m3,m0,m0,m1,m3,m1,h1,h2,h3,h4,h5,h6) # tau,0,0,1,TAU,1
	v3e=mtrixcal(m0,m1,m2,m4,m0,m3,h1,h2,h3,h4,h5,h6) # 0,1,-1,-tau,0,tau
	v1i=mtrixcal(m3,m2,m2,m0,m4,m0,h1,h2,h3,h4,h5,h6) # tau,-1,-1,0,-tau,0
	v2i=mtrixcal(m2,m0,m0,m3,m2,m3,h1,h2,h3,h4,h5,h6) # -1,0,0,tau,-1,tau
	v3i=mtrixcal(m0,m3,m4,m1,m0,m2,h1,h2,h3,h4,h5,h6) # 0,tau,-tau,1,0,-1
	return [v1e,v2e,v3e,v1i,v2i,v3i]

cdef np.ndarray mtrixcal(np.ndarray[np.int64_t, ndim=1] v1,\
						np.ndarray[np.int64_t, ndim=1] v2,\
						np.ndarray[np.int64_t, ndim=1] v3,\
						np.ndarray[np.int64_t, ndim=1] v4,\
						np.ndarray[np.int64_t, ndim=1] v5,\
						np.ndarray[np.int64_t, ndim=1] v6,\
						np.ndarray[np.int64_t, ndim=1] n1,\
						np.ndarray[np.int64_t, ndim=1] n2,\
						np.ndarray[np.int64_t, ndim=1] n3,\
						np.ndarray[np.int64_t, ndim=1] n4,\
						np.ndarray[np.int64_t, ndim=1] n5,\
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

#cdef projection_perp(n1,n2,n3,n4,n5,n6):
cpdef list projection_perp(np.ndarray[np.int64_t, ndim=1] n1,\
						np.ndarray[np.int64_t, ndim=1] n2,\
						np.ndarray[np.int64_t, ndim=1] n3,\
						np.ndarray[np.int64_t, ndim=1] n4,\
						np.ndarray[np.int64_t, ndim=1] n5,\
						np.ndarray[np.int64_t, ndim=1] n6):
	# This returns 6D indeces of a projection of 6D vector (n1,n2,n3,n4,n5,n6) onto Eperp
	# Direct lattice vector
	cdef np.ndarray[np.int64_t, ndim=1] m1,m2,m3
	cdef np.ndarray[np.int64_t, ndim=1] h1,h2,h3,h4,h5,h6
	#m1=np.array([ 2, 1, 2]) #  (TAU+2)/2 in 'TAU-style'
	#m2=np.array([ 0, 1, 2]) #  TAU/2
	#m3=np.array([ 0,-1, 2]) # -TAU/2
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

cpdef list add(long p1,long p2,long p3,long q1,long q2,long q3): # A+B
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

cpdef list sub(long p1,long p2,long p3,long q1,long q2,long q3): # A-B
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

cpdef list mul(long p1,long p2,long p3,long q1,long q2,long q3): # A*B
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

cpdef list div(long p1,long p2,long p3,long q1,long q2,long q3): # A/B
	cdef long gcd,c1,c2,c3
	cdef np.ndarray[np.int64_t,ndim=1] x
	if q1==0 and q2==0:
		print 'ERROR_1:division error'
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
				print 'ERROR_2:division error',c1,c2,c3,p1,p2,p3,q1,q2,q3
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

#cpdef list full_sym_obj(np.ndarray[np.int64_t, ndim=3] obj,np.ndarray[np.int64_t, ndim=2] centre):
#cpdef np.ndarray full_sym_obj(np.ndarray[np.int64_t, ndim=3] obj,np.ndarray[np.int64_t, ndim=2] centre):
cpdef np.ndarray generator_obj_symmetric_surface(np.ndarray[np.int64_t, ndim=3] obj,np.ndarray[np.int64_t, ndim=2] centre):
	cdef int i
	cdef list od,mop
	cdef np.ndarray[np.int64_t,ndim=4] tmp4
	od=[]
	mop=icosasymop()
	for i in range(len(mop)):
		od.extend(symop_obj(mop[i],obj,centre))
	tmp4=np.array(od).reshape(len(od)/54,3,6,3) # 54=3*6*3
	print ' Number of triangles on POD surface: %d'%(len(tmp4))
	return tmp4	

cpdef np.ndarray generator_obj_symmetric_tetrahedron(np.ndarray[np.int64_t, ndim=3] obj,np.ndarray[np.int64_t, ndim=2] centre):
	cdef int i
	cdef list od,mop
	cdef np.ndarray[np.int64_t,ndim=4] tmp4
	print 'Generating symmetric POD'
	od=[]
	mop=icosasymop()
	for i in range(len(mop)):
		od.extend(symop_obj(mop[i],obj,centre))
	tmp4=np.array(od).reshape(len(od)/72,4,6,3) # 72=4*6*3
	print ' Number of tetrahedron: %d'%(len(tmp4))
	return tmp4	

cpdef np.ndarray generator_obj_symmetric_tetrahedron_specific_symop(np.ndarray[np.int64_t, ndim=3] obj,np.ndarray[np.int64_t, ndim=2] centre,list symmetry_operation):
	# using specific symmetry operations
	cdef int i
	cdef list od
	cdef np.ndarray[np.int64_t,ndim=4] tmp4
	print 'Generating symmetric POD'
	od=[]
	for i in range(len(symmetry_operation)):
		od.extend(symop_obj(symmetry_operation[i],obj,centre))
	tmp4=np.array(od).reshape(len(od)/72,4,6,3) # 72=4*6*3
	print ' Number of tetrahedron: %d'%(len(tmp4))
	return tmp4	

# B-Reps
cpdef generator_breps_obj_symmetric(np.ndarray[np.int64_t, ndim=4] points,np.ndarray[np.int64_t, ndim=3] faces,np.ndarray[np.int64_t, ndim=2] centre,int verbose):
	
	# 非対称領域にある複数の四面体からなるobjに、m-3-5の対称操作を施して、対称的なobjをつくる
	cdef int i1,i2,i3,counter1
	cdef flag
	cdef list od,mop,list1
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a
	
	if verbose>0:
		print ' generator_obj_symmetric_tetrahedron_breps()'
	else:
		pass
	
	od=[]
	mop=icosasymop()
	
	for i1 in range(len(mop)):
		for i2 in range(len(points)): # i2-th tetrahedron in points
			for i3 in range(4): # i3-th triange
				if i1<60: # without inversion
					for i4 in [0,1,2]:
						#points[i2][faces[i2][i3][i4]] # i4-th point of i3-th triange
						od.extend(symop_vec(mop[i1],points[i2][faces[i2][i3][i4]],centre))
				else: # with inversion
					for i4 in [0,2,1]:
						od.extend(symop_vec(mop[i1],points[i2][faces[i2][i3][i4]],centre))
	if verbose>1:
		print ' Number of tetrahedron: %d'%(len(od)/72)
	else:
		pass
	
	#return np.array(od).reshape(len(od)/72,4,6,3) # 72=4*6*3
	
	tmp4a=np.array(od).reshape(len(od)/72,4,6,3) # 72=4*6*3
	
	# B-Reps形式のobjに直す．
	# tmp2aには、表面をなす三角形について、時計回りの順に頂点が入っている（重複あり）
	# まず、重複のない頂点の集合を得てこれをpointsとし、その配列インデックスを得て、これをfacesとする．
	list1=[] # faces
	tmp3a=remove_doubling_dim4_in_perp_space(tmp4a) # points
	for i1 in range(len(tmp4a)):
		for i2 in range(3):
			counter1=0
			for i3 in range(len(tmp3a)):
				flag=check_two_vertices(tmp4a[i1][i2],tmp3a[i3],verbose-2)
				if flag==1: # equivalent
					counter1+=1
					break
				else:
					pass
			if counter1!=0:
				list1.append(i3)
			else:
				pass
	return tmp3a,np.array(list1).reshape(len(list1)/3,3)

# New
# B-Reps
cpdef list generator_breps_obj_symmetric_special(np.ndarray[np.int64_t, ndim=3] points,np.ndarray[np.int64_t, ndim=2] faces,np.ndarray[np.int64_t, ndim=2] centre,int verbose):
	
	# 非対称領域にある複数の四面体からなるobjに、m-3-5の対称操作を施して、対称的なobjをつくる．
	# 対称心はcentreで指定しているが、原点を想定している．
	# centreを含む面を除くことで、object表面のみにある三角形のみを作る．
	
	cdef int i1,i2,i3,i4,counter1
	cdef flag
	cdef list od,mop,list1
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a
	
	if verbose>0:
		print ' generator_obj_symmetric_tetrahedron_breps()'
	else:
		pass
	
	od=[]
	mop=icosasymop()
	
	for i1 in range(len(mop)):
		for i2 in range(len(faces)):
			counter1=0
			for i3 in [0,1,2]:
				flag=check_two_vertices(points[faces[i2][i3]],centre,verbose-2) # centreを頂点にもつ三角形はここで除く
				if flag==1: # equivalent
					counter1+=1
					break
				else:
					pass
			if counter1!=0:
				if i1<60: # without inversion
					for i4 in [0,1,2]:
						od.extend(symop_vec(mop[i1],points[faces[i2][i4]],centre))
				else: # with inversion
					for i4 in [0,2,1]:
						od.extend(symop_vec(mop[i1],points[faces[i2][i4]],centre))
			else:
				pass
	tmp4a=np.array(od).reshape(len(od)/54,3,6,3)
	if verbose>1:
		print '  number of triangle: %d'%(len(tmp4a))
	else:
		pass
	
	# B-Reps形式のobjに直す．
	# tmp2aには、表面をなす三角形について、時計回りの順に頂点が入っている（重複あり）
	# まず、重複のない頂点の集合を得てこれをpointsとし、その配列インデックスを得て、これをfacesとする．
	tmp3a=remove_doubling_dim4_in_perp_space(tmp4a) # points
	if verbose>1:
		print '  number of points: %d'%(len(tmp3a))
	else:
		pass
	list1=[] # faces
	for i1 in range(len(tmp4a)):
		for i2 in range(3):
			counter1=0
			for i3 in range(len(tmp3a)):
				flag=check_two_vertices(tmp4a[i1][i2],tmp3a[i3],verbose-2)
				if flag==1: # equivalent
					counter1+=1
					break
				else:
					pass
			if counter1!=0:
				list1.append(i3)
			else:
				pass
	return [tmp3a,np.array(list1).reshape(len(list1)/3,3)]

cpdef np.ndarray generator_obj_symmetric_tetrahedron_0(np.ndarray[np.int64_t, ndim=3] obj,np.ndarray[np.int64_t, ndim=2] centre, int numop):
	cdef int i
	cdef list od,mop
	cdef np.ndarray[np.int64_t,ndim=4] tmp4
	od=[]
	print 'Generating asymmetric POD'
	mop=icosasymop()
	od.extend(symop_obj(mop[numop],obj,centre))
	tmp4=np.array(od).reshape(len(od)/72,4,6,3) # 72=4*6*3
	print ' Number of tetrahedron: %d'%(len(tmp4))
	return tmp4	

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
	#print '    remove_doubling_dim3_in_perp_space()'
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
	print >>f,'%d'%(len(tmp3))
	print >>f,'%s'%(filename)
	for i1 in range(len(tmp3)):
		a1,a2,a3,a4,a5,a6=projection(tmp3[i1][0],tmp3[i1][1],tmp3[i1][2],tmp3[i1][3],tmp3[i1][4],tmp3[i1][5])
		print >>f,'Xx %8.6f %8.6f %8.6f # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d'%\
		((a4[0]+a4[1]*TAU)/float(a4[2]),(a5[0]+a5[1]*TAU)/float(a5[2]),(a6[0]+a6[1]*TAU)/float(a6[2]),\
		tmp3[i1][0][0],tmp3[i1][0][1],tmp3[i1][0][2],\
		tmp3[i1][1][0],tmp3[i1][1][1],tmp3[i1][1][2],\
		tmp3[i1][2][0],tmp3[i1][2][1],tmp3[i1][2][2],\
		tmp3[i1][3][0],tmp3[i1][3][1],tmp3[i1][3][2],\
		tmp3[i1][4][0],tmp3[i1][4][1],tmp3[i1][4][2],\
		tmp3[i1][5][0],tmp3[i1][5][1],tmp3[i1][5][2])
	return 0

cpdef int generator_xyz_dim4_triangle(np.ndarray[np.int64_t, ndim=4] obj,filename):
	cdef int i1,i2
	cdef np.ndarray[np.int64_t,ndim=1] a1,a2,a3,a4,a5,a6
	f=open('%s'%(filename),'w')
	#tmp3=remove_doubling_dim4(obj)
	print >>f,'%d'%(len(obj)*3)
	print >>f,'%s'%(filename)
	for i1 in range(len(obj)):
		for i2 in range(3):
			a1,a2,a3,a4,a5,a6=projection(obj[i1][i2][0],obj[i1][i2][1],obj[i1][i2][2],obj[i1][i2][3],obj[i1][i2][4],obj[i1][i2][5])
			print >>f,'Xx %8.6f %8.6f %8.6f # %3d-the tetrahedron %d-th vertex # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d'%\
			((a4[0]+a4[1]*TAU)/float(a4[2]),\
			(a5[0]+a5[1]*TAU)/float(a5[2]),\
			(a6[0]+a6[1]*TAU)/float(a6[2]),\
			i1,i2,\
			obj[i1][i2][0][0],obj[i1][i2][0][1],obj[i1][i2][0][2],\
			obj[i1][i2][1][0],obj[i1][i2][1][1],obj[i1][i2][1][2],\
			obj[i1][i2][2][0],obj[i1][i2][2][1],obj[i1][i2][2][2],\
			obj[i1][i2][3][0],obj[i1][i2][3][1],obj[i1][i2][3][2],\
			obj[i1][i2][4][0],obj[i1][i2][4][1],obj[i1][i2][4][2],\
			obj[i1][i2][5][0],obj[i1][i2][5][1],obj[i1][i2][5][2])
	return 0

cpdef int generator_xyz_dim4_tetrahedron(np.ndarray[np.int64_t, ndim=4] obj,filename,int option):
	cdef int i1,i2
	cdef long w1,w2,w3
	cdef np.ndarray[np.int64_t,ndim=1] a1,a2,a3,a4,a5,a6
	
	# option=0 # generate single .xyz file
	# option=1 # generate single .xyz file
	
	if option==0:
		f=open('%s.xyz'%(filename),'w')
		#tmp3=remove_doubling_dim4(obj)
		print >>f,'%d'%(len(obj)*4)
		print >>f,'%s'%(filename)
		for i1 in range(len(obj)):
			for i2 in range(4):
				a1,a2,a3,a4,a5,a6=projection(obj[i1][i2][0],obj[i1][i2][1],obj[i1][i2][2],obj[i1][i2][3],obj[i1][i2][4],obj[i1][i2][5])
				print >>f,'Xx %8.6f %8.6f %8.6f # %3d-the tetrahedron %d-th vertex # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d'%\
				((a4[0]+a4[1]*TAU)/float(a4[2]),\
				(a5[0]+a5[1]*TAU)/float(a5[2]),\
				(a6[0]+a6[1]*TAU)/float(a6[2]),\
				i1,i2,\
				obj[i1][i2][0][0],obj[i1][i2][0][1],obj[i1][i2][0][2],\
				obj[i1][i2][1][0],obj[i1][i2][1][1],obj[i1][i2][1][2],\
				obj[i1][i2][2][0],obj[i1][i2][2][1],obj[i1][i2][2][2],\
				obj[i1][i2][3][0],obj[i1][i2][3][1],obj[i1][i2][3][2],\
				obj[i1][i2][4][0],obj[i1][i2][4][1],obj[i1][i2][4][2],\
				obj[i1][i2][5][0],obj[i1][i2][5][1],obj[i1][i2][5][2])
		w1,w2,w3=obj_volume_6d(obj)
		print >>f,'volume = %d %d %d (%8.6f)'%(w1,w2,w3,(w1+TAU*w2)/float(w3))
		for i1 in range(len(obj)):
			[v1,v2,v3]=tetrahedron_volume_6d(obj[i1])
			print >>f,'%3d-the tetrahedron, %d %d %d (%8.6f)'%(i1,v1,v2,v3,(v1+TAU*v2)/float(v3))	
	elif option==1:
		for i1 in range(len(obj)):
			f=open('%s_%d.xyz'%(filename,i1),'w')
			print >>f,'%d'%(4)
			print >>f,'%s_%d'%(filename,i1)
			for i2 in range(4):
				a1,a2,a3,a4,a5,a6=projection(obj[i1][i2][0],obj[i1][i2][1],obj[i1][i2][2],obj[i1][i2][3],obj[i1][i2][4],obj[i1][i2][5])
				print >>f,'Xx %8.6f %8.6f %8.6f # %3d-the tetrahedron %d-th vertex # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d'%\
				((a4[0]+a4[1]*TAU)/float(a4[2]),\
				(a5[0]+a5[1]*TAU)/float(a5[2]),\
				(a6[0]+a6[1]*TAU)/float(a6[2]),\
				i1,i2,\
				obj[i1][i2][0][0],obj[i1][i2][0][1],obj[i1][i2][0][2],\
				obj[i1][i2][1][0],obj[i1][i2][1][1],obj[i1][i2][1][2],\
				obj[i1][i2][2][0],obj[i1][i2][2][1],obj[i1][i2][2][2],\
				obj[i1][i2][3][0],obj[i1][i2][3][1],obj[i1][i2][3][2],\
				obj[i1][i2][4][0],obj[i1][i2][4][1],obj[i1][i2][4][2],\
				obj[i1][i2][5][0],obj[i1][i2][5][1],obj[i1][i2][5][2])
				[v1,v2,v3]=tetrahedron_volume_6d(obj[i1])
			print >>f,'%3d-the tetrahedron, %d %d %d (%8.6f)'%(i1,v1,v2,v3,(v1+TAU*v2)/float(v3))	
	return 0

cpdef int generator_xyz_dim4_tetrahedron_select(np.ndarray[np.int64_t, ndim=4] obj,filename,int num):
	cdef int i2
	cdef long w1,w2,w3
	cdef np.ndarray[np.int64_t,ndim=1] a1,a2,a3,a4,a5,a6
	
	# option=0 # generate single .xyz file
	# option=1 # generate single .xyz file
	
	f=open('%s_%d.xyz'%(filename,num),'w')
	print >>f,'%d'%(4)
	print >>f,'%s_%d'%(filename,num)
	for i2 in range(4):
		a1,a2,a3,a4,a5,a6=projection(obj[num][i2][0],obj[num][i2][1],obj[num][i2][2],obj[num][i2][3],obj[num][i2][4],obj[num][i2][5])
		print >>f,'Xx %8.6f %8.6f %8.6f # %3d-the tetrahedron %d-th vertex # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d'%\
		((a4[0]+a4[1]*TAU)/float(a4[2]),\
		(a5[0]+a5[1]*TAU)/float(a5[2]),\
		(a6[0]+a6[1]*TAU)/float(a6[2]),\
		num,i2,\
		obj[num][i2][0][0],obj[num][i2][0][1],obj[num][i2][0][2],\
		obj[num][i2][1][0],obj[num][i2][1][1],obj[num][i2][1][2],\
		obj[num][i2][2][0],obj[num][i2][2][1],obj[num][i2][2][2],\
		obj[num][i2][3][0],obj[num][i2][3][1],obj[num][i2][3][2],\
		obj[num][i2][4][0],obj[num][i2][4][1],obj[num][i2][4][2],\
		obj[num][i2][5][0],obj[num][i2][5][1],obj[num][i2][5][2])
		[v1,v2,v3]=tetrahedron_volume_6d(obj[num])
	print >>f,'%3d-the tetrahedron, %d %d %d (%8.6f)'%(num,v1,v2,v3,(v1+TAU*v2)/float(v3))	
	return 0

cpdef int generator_xyz_dim3(np.ndarray[np.int64_t, ndim=3] obj,filename):
	cdef int i1
	cdef np.ndarray[np.int64_t,ndim=1] a1,a2,a3,a4,a5,a6
	cdef np.ndarray[np.int64_t,ndim=2] tmp2
	cdef np.ndarray[np.int64_t,ndim=3] tmp3
	f=open('%s'%(filename),'w')
	tmp3=remove_doubling_dim3(obj)
	print >>f,'%d'%(len(tmp3))
	print >>f,'%s'%(filename)
	for i1 in range(len(tmp3)):
		a1,a2,a3,a4,a5,a6=projection(tmp3[i1][0],tmp3[i1][1],tmp3[i1][2],tmp3[i1][3],tmp3[i1][4],tmp3[i1][5])
		print >>f,'Xx %8.6f %8.6f %8.6f # %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d'%\
		((a4[0]+a4[1]*TAU)/float(a4[2]),(a5[0]+a5[1]*TAU)/float(a5[2]),(a6[0]+a6[1]*TAU)/float(a6[2]),\
		tmp3[i1][0][0],tmp3[i1][0][1],tmp3[i1][0][2],\
		tmp3[i1][1][0],tmp3[i1][1][1],tmp3[i1][1][2],\
		tmp3[i1][2][0],tmp3[i1][2][1],tmp3[i1][2][2],\
		tmp3[i1][3][0],tmp3[i1][3][1],tmp3[i1][3][2],\
		tmp3[i1][4][0],tmp3[i1][4][1],tmp3[i1][4][2],\
		tmp3[i1][5][0],tmp3[i1][5][1],tmp3[i1][5][2])
	return 0

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
		print '      Generating edges'
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
		tmp1d=np.array([],dtype=np.int)
		counter_ab=0
		counter_ac=0
		counter_bc=0
		for j in range(0,num2):
			tmp1a=np.array([],dtype=np.int)
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
			tmp1a=np.array([],dtype=np.int)
		for j in range(0,num2):
			tmp1b=np.array([],dtype=np.int)
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
			tmp1b=np.array([],dtype=np.int)
		for j in range(0,num2):
			tmp1c=np.array([],dtype=np.int)	
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
			tmp1c=np.array([],dtype=np.int)
		tmp1d=np.append(tmp1d,tmp1a)
		tmp1d=np.append(tmp1d,tmp1b)
		tmp1d=np.append(tmp1d,tmp1c)
		tmp1d=np.append(tmp4a,tmp1d)
		num2=len(tmp1d)/36 # 36=2*6*3
		tmp4a=tmp1d.reshape(num2,2,6,3)
	if verbose>=1:
		print '       Number of edges: %d'%(len(tmp4a))
	else:
		pass
	return tmp4a

"""
cdef int equivalent_triangle(np.ndarray[np.int64_t, ndim=1] triangle1,\
							np.ndarray[np.int64_t, ndim=1] triangle2):
	cdef int i,i1,i2,i3,i4,i5,i6,counter
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
		if np.all(tmp3a[i1]==tmp3b[i4]) and np.all(tmp3a[i2]==tmp3b[i5]) and np.all(tmp3a[i3]==tmp3b[i6]):
			counter+=1
			break
		else:
			pass
	if counter==0:
		return 1 # not equivalent traiangles
	else:
		return 0 # equivalent traiangle
"""

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

# B-Reps
cdef int equivalent_triangle_breps(np.ndarray[np.int64_t, ndim=1] triangle1,\
							np.ndarray[np.int64_t, ndim=1] triangle2):
	cdef int i,i1,i2,i3,i4,i5,i6,counter1,counter2
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
	
	# ２つの三角形の頂点が、同じ方向の場合
	comb=[[0,1,2,0,1,2],\
		[0,1,2,1,2,0],\
		[0,1,2,2,0,1]]
	counter1=0
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
			counter1+=1
			break
		else:
			pass
	
	# ２つの三角形の頂点の方向が逆の場合
	comb=[[0,1,2,0,2,1],\
		[0,1,2,2,1,0],\
		[0,1,2,1,0,2]]
	counter2=0
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
			counter2+=1
			break
		else:
			pass
	
	# ２つの三角形は等価でない
	if counter1==0 and counter2==0:
		return 0
	
	# ２つの三角形は等価、２つの三角形の頂点の方向が同じ場合
	elif counter1!=0 and counter2==0:
		return 1
	
	# ２つの三角形は等価、２つの三角形の頂点の方向が逆の場合
	elif counter1==0 and counter2!=0:
		return -1
	
	# その他
	else:
		return 2 # 

#
cpdef intersection_breps_obj(np.ndarray[np.int64_t, ndim=4] obj1_surface,\
							np.ndarray[np.int64_t, ndim=4] obj2_surface,\
							np.ndarray[np.int64_t, ndim=4] obj1_edge,\
							np.ndarray[np.int64_t, ndim=4] obj2_edge,\
							np.ndarray[np.int64_t, ndim=4] obj1_tetrahedron,\
							np.ndarray[np.int64_t, ndim=4] obj2_tetrahedron,\
							int verbose):
	
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
	
	if verbose>0:
		print ' intersection_breps_obj()'
	else:
		pass
	
	counter=0
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
		print ' No intersection'
		return np.array([0],dtype=np.int).reshape(1,1,1,1)
	else:
		point=p.reshape(len(p)/18,6,3) # 18=6*3
		point1=remove_doubling_dim3(point)
		print ' dividing into three PODs:'
		print '    Common   :     OD1 and     ODB'
		print '  UnCommon 1 :     OD1 and Not OD2'
		print '  UnCommon 2 : Not OD1 and     OD2'
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


# B-Reps
cpdef list merge_obj_breps(np.ndarray[np.int64_t, ndim=4] points, np.ndarray[np.int64_t, ndim=3] faces, int verbose):
	#
	# 四面体からなるB-Reps OBJをまとめ（表面の三角形を得ることと等しい）、points(dim3)とfaces(dim2)を返す
	#	
	cdef int i1,i2,i3
	cdef int counter1,flag
	cdef list list1
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1c
	cdef np.ndarray[np.int64_t,ndim=2] tmp2a
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a
	
	# pointsとfaceをもつbreps-objにあるユニークなfaceを得る
	if verbose>0:
		print ' merge_obj_breps()'
	else:
		pass
	
	# initialize tmp2a by four triangles of 1-th tetrahedron
	for i2 in range(4):
		for i3 in range(3):
			if i2==0 and i3==0:
				tmp1a=points[0][faces[0][i2][0]].reshape(18)
			else:
				tmp1a=np.append(tmp1a,points[0][faces[0][i2][i3]])
	tmp2a=tmp1a.reshape(len(tmp1a)/54,54) # 54=3*6*3
	
	# 重複した三角形をなくし、ユニークな三角形を得る
	list1=[]
	for i1 in range(1,len(points)):
		tmp1c=np.array([0])
		for i2 in range(4): # i2-th triangle of i1-th tetrahedron
			counter1=0
			tmp1a=points[i1][faces[i1][i2][0]].reshape(18) # 1st vertex
			for i3 in range(1,3):
				tmp1a=np.append(tmp1a,points[i1][faces[i1][i2][i3]]) # add, 2nd and 3rd vertices
			for i3 in range(len(tmp2a)):
				flag=equivalent_triangle_breps(tmp1a,tmp2a[i3])
				if flag==-1: # 2つの三角形が等しく、頂点の順番が逆の場合
					counter1+=1
					list1.append(i3) # 重複した三角形の配列index,i3,をキープする
					break
				else:
					pass
			if counter1==0:
				if len(tmp1c)==1:
					tmp1c=tmp1a
				else:
					tmp1c=np.append(tmp1c,tmp1a)
			else:
				pass
		if len(tmp1c)==1:
			pass
		else:
			tmp1c=np.append(tmp2a,tmp1c)
			tmp2a=tmp1c.reshape(len(tmp1c)/54,54)
	
	if verbose>1:
		print '  number of unique triangles: %d'%(len(tmp2a))
	else:
		pass
	
	#print list1
	
	# breps-objの表面にある三角形を得る．list1内は重複した三角形の配列index
	tmp1a=np.array([0])
	for i1 in range(len(tmp2a)):
		counter1=0
		for i2 in list1:
			if i1==i2:
				counter1+=1
				break
			else:
				pass
		if counter1==0:
			if len(tmp1a)==1:
				tmp1a=tmp2a[i1]
			else:
				tmp1a=np.append(tmp1a,tmp2a[i1])
		else:
			pass
	tmp4a=tmp1a.reshape(len(tmp1a)/54,3,6,3)
	if verbose>1:
		print '  number of surface triangles: %d'%(len(tmp4a))
	else:
		pass
	
	# B-Reps形式のobjに直す．
	# tmp2aには、表面をなす三角形について、時計回りの順に頂点が入っている（重複あり）
	# まず、重複のない頂点の集合を得てこれをpointsとし、その配列インデックスを得て、これをfacesとする．
	list1=[] # faces
	tmp3a=remove_doubling_dim4_in_perp_space(tmp4a) # points
	for i1 in range(len(tmp4a)):
		for i2 in range(3):
			counter1=0
			for i3 in range(len(tmp3a)):
				flag=check_two_vertices(tmp4a[i1][i2],tmp3a[i3],verbose-2)
				if flag==1: # equivalent
					counter1+=1
					break
				else:
					pass
			if counter1!=0:
				list1.append(i3)
			else:
				pass
	
	return [tmp3a,np.array(list1).reshape(len(list1)/3,3)]
	
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
		print '      generator_surface_1()'
	else:
		pass
	#
	# (1) preparing a list of triangle surfaces without doubling (tmp2)
	#
	if verbose>1:
		print '       Number of triangles in POD: %d'%(len(obj)*4)
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
		print '       Number of unique triangles: %d'%(len(tmp2a))
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
				print '      ERROR_001 %d: check your model.'%(counter1)
			else:
				pass
				
	if verbose>=1:
		print '       Number of triangles on POD surface:%d'%(len(tmp1)/54)
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
		print '      Generating surfaces()'
	else:
		pass
	#
	# (1) preparing a list of triangle surfaces without doubling (tmp2)
	#
	if verbose>=1:
		print '       Number of triangles in POD: %d'%(len(obj)*4)
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
		print '       Number of unique triangles: %d'%(len(tmp2))
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
			print '      ERROR_001 %d: check your model.'%(counter3)
		else:
			pass
		return np.array([0]).reshape(1,1,1,1)
	else:
		tmp4a=tmp1a.reshape(len(tmp1a)/54,3,6,3) # 54=3*6*3
		if verbose>=1:
			print '       Number of triangles on POD surface:%d'%(counter2)
		else:
			pass
		return tmp4a

#cpdef np.ndarray intersection_two_obj(np.ndarray[np.int64_t, ndim=3] obj1,np.ndarray[np.int64_t, ndim=3] obj2,np.ndarray[np.int64_t, ndim=2] od1_centre,np.ndarray[np.int64_t, ndim=2] od2_centre,int flag1,int flag2, path):
cpdef intersection_two_obj(np.ndarray[np.int64_t, ndim=3] obj1,\
							np.ndarray[np.int64_t, ndim=3] obj2,\
							np.ndarray[np.int64_t, ndim=2] od1_centre,\
							np.ndarray[np.int64_t, ndim=2] od2_centre,\
							int flag1,\
							int flag2,\
							path):
	cdef int i1,i2,i3,counter,counter1,counter2,num1,num2,flag
	cdef np.ndarray[np.int64_t,ndim=1] p,tmp,tmp1
	cdef np.ndarray[np.int64_t,ndim=2] point_tmp,point1_tmp
	cdef np.ndarray[np.int64_t,ndim=3] tr1,tr2,ed1,ed2,point,point1,point_common,point_a,point_a1,point_a2,point_b,point_b1,point_b2,tmp3,tet
	cdef np.ndarray[np.int64_t,ndim=4] obj1_edge,obj2_edge,obj1_surface,obj2_surface,obj1_tetrahedron,obj2_tetrahedron
	cdef np.ndarray[np.int64_t,ndim=4] obj_common,obj_1,obj_2
	cdef list comb
	#
	print 'Itersection by intersection_two_obj()'
	print ' Generating 1st object:'
	if flag1==0:
		obj1_tetrahedron=generator_obj_symmetric_tetrahedron(obj1,od1_centre)
		obj1_surface=generator_surface(obj1_tetrahedron)
		obj1_edge=generator_edge(obj1_surface)
	elif flag1==1:
		obj1_tetrahedron=obj1
		obj1_surface=generator_surface(obj1_tetrahedron)
	elif flag1==2:
		obj1_tetrahedron=generator_obj_symmetric_tetrahedron_0(obj1,od1_centre,1)
		obj1_surface=generator_surface(obj1_tetrahedron)
		obj1_edge=generator_edge(obj1_surface)
	else:
		pass
	#
	print ' Generating 2nd object:'
	if flag2==0:
		obj2_tetrahedron=generator_obj_symmetric_tetrahedron(obj2,od2_centre)
		#obj2_surface=generator_surface(obj2_tetrahedron)
		#obj2_edge=generator_edge(obj2_surface)
	elif flag2==1:
		obj2_tetrahedron=obj2
		#obj2_surface=generator_surface(obj2_tetrahedron)
	elif flag2==2:
		obj2_tetrahedron=generator_obj_symmetric_tetrahedron_0(obj2,od2_centre,1)
		obj2_surface=generator_surface(obj2_tetrahedron)
		obj2_edge=generator_edge(obj2_surface)
	else:
		pass
	
	#
	# Generating two obj in input
	print 'INPUTed two POD'
	file_tmp='%s/OBJ1.xyz'%(path)
	generator_xyz_dim4(obj1_surface,file_tmp)
	print ' 1st obj saved as %s/OBJ1.xyz'%(path)
	#file_tmp='%s/OBJ2.xyz'%(path)
	#generator_xyz_dim4(obj2_surface,file_tmp)
	#print '	  2nd obj saved as %s/OBJ2.xyz'%(path)





	#
	### Subdivition ###
	#
	#	 OD1 and ODB	 : common part, obj_common
	#	 OD1 and Not OD2 : obj_1
	# Not OD1 and OD2	 : obj_2
	#
	if flag1==0 and flag2==0:
		point_common,point_a,point_b=intersection_using_surface(obj1_surface,obj2_surface,obj1_edge,obj2_edge,obj1_tetrahedron,obj2_tetrahedron,path)
		#
		#----------------------------------------------
		# Algorithm 1
		#----------------------------------------------
		# This is very simple but work correctly only when each subdivided 
		# three ODs (i.e part part, ODA and ODB) are be able to define as 
		# a series of tetrahedra.
		#
		# If two objs are both symmetric (flag1=0 and flag2=0), Algorithm 1 should be fine.
		#
		if np.all(point_common!=np.array([0],dtype=np.int).reshape(1,1,1,1)):
			print '   Generating .XYZ files in %s'%(path)
			print '	  %s/point_a.xyz'%(path)
			print '	  %s/point_b.xyz'%(path)
			print '	  %s/point_common.xyz'%(path)
			file_tmp='%s/point_a.xyz'%(path)
			generator_xyz_dim3(point_a,file_tmp)
			file_tmp='%s/point_b.xyz'%(path)
			generator_xyz_dim3(point_b,file_tmp)
			file_tmp='%s/point_common.xyz'%(path)
			generator_xyz_dim3(point_common,file_tmp)
		else:
			print '   No intersection'
			print '   Generating .XYZ files in %s'%(path)
			print '	  %s/point_a.xyz'%(path)
			print '	  %s/point_b.xyz'%(path)
			file_tmp='%s/point_a.xyz'%(path)
			generator_xyz_dim3(point_a,file_tmp)
			file_tmp='%s/point_b.xyz'%(path)
			generator_xyz_dim3(point_b,file_tmp)
		return point_common,point_a,point_b
	else:
	#elif flag1==1 and flag2==1:
		#
		#flag=2
		#flag=3
		flag=2
		# ----------------------------------------------
		# Algorithm 2
		# ----------------------------------------------
		# intersection_using_tetrahedron()
		#
		# parameters:
		# tetrahedron_1(dim3)
		# tetrahedron_2(dim3)
		#
		# return: (dim4)
		# common part: tetrahedron_1 and tetrahedron_2
		#
		if flag==2:
			obj_common=intersection_using_tetrahedron(obj1_tetrahedron,obj2_tetrahedron,path)
			print '   Generating .XYZ files in %s'%(path)
			if obj_common.tolist()!=[[[[0]]]]:
				print '   OBJ1 and OBJ2 : %d x tetrahedron'%(len(obj_common))
				file_tmp='%s/obj_common.xyz'%(path)
				#generator_xyz_dim4(obj_common,file_tmp)
				generator_xyz_dim4_tetrahedron(obj_common,file_tmp,0)
				print '	  saved as %s/obj_common.xyz'%(path)
			else:
				print '   OBJ1 and OBJ2 : empty'
			#return obj_common
			
			# Generating obj_common
			#obj_common_surface=generator_surface(obj_common)
			#print '   obj_common'
			#file_tmp='%s/obj_common_surface.xyz'%(path)
			#generator_xyz_dim4(obj_common_surface,file_tmp)
			#print '	  1st obj saved as %s/obj_common_surface.xyz'%(path)
			
			#
			#obj_a=object_subtraction(obj1_tetrahedron,obj_common)
			obj_a=object_subtraction_new(obj1_tetrahedron,obj_common,obj2_tetrahedron,path)
			""" # (dim3)
			if obj_a.tolist()!=[[[0]]]:
				print '   (obj_a) OBJ1 and NOT OBJ2 : %d x tetrahedron'%(len(obj_a))
				file_tmp='%s/obj_a.xyz'%(path)
				generator_xyz_dim3(obj_a,file_tmp)
				print '	  saved as %s/obj_a.xyz'%(path)
			else:
				print '   (obj_a) OBJ1 and NOTOBJ2 : empty'
			"""
			#""" # (dim3)
			if obj_a.tolist()!=[[[[0]]]]:
				print '   (obj_a) OBJ1 and NOT OBJ2 : %d x tetrahedron'%(len(obj_a))
				file_tmp='%s/obj_a.xyz'%(path)
				#generator_xyz_dim4(obj_a,file_tmp)
				generator_xyz_dim4_tetrahedron(obj_a,file_tmp,0)
				print '	  saved as %s/obj_a.xyz'%(path)
			else:
				print '   (obj_a) OBJ1 and NOTOBJ2 : empty'
			#"""
				
				
			#
			"""
			#obj_b=object_subtraction(obj2_tetrahedron,obj_common)
			obj_b=object_subtraction_new(obj2_tetrahedron,obj_common,obj1_tetrahedron)
			if obj_b.tolist()!=[[[[0]]]]:
				print '   (obj_b) NOT OBJ1 and OBJ2 : %d x tetrahedron'%(len(obj_b))
				file_tmp='%s/obj_b.xyz'%(path)
				generator_xyz_dim4(obj_b,file_tmp)
				print '	  saved as %s/obj_b.xyz'%(path)
			else:
				print '   (obj_b) NOT OBJ1 and OBJ2 : empty'
			return obj_common,obj_a,obj_b
			"""
		#
		# ----------------------------------------------
		# Algorithm 3 (upgrade version of Algorithm 2)
		# 
		#	ATTENTION: this does NOT work correctly!
		# ----------------------------------------------
		# intersection_using_tetrahedron_new()
		#
		# parameters:
		# tetrahedron_1(dim3)
		# tetrahedron_2(dim3)
		#
		# return: (dim4)
		# common object	  :	 tetrahedron_1 and	 tetrahedron_2
		# non-common object 1:	 tetrahedron_1 and NOT tetrahedron_2
		# non-common object 2: NOT tetrahedron_1 and	 tetrahedron_2
		#
		if flag==3:
			obj_common,obj_a,obj_b=intersection_using_tetrahedron_new(obj1_tetrahedron,obj2_tetrahedron)
			print '   Generating .XYZ files in %s'%(path)
			if obj_common.tolist()!=[[[[0]]]]:
				print '   (obj_common) OBJ1 and OBJ2 : %d x tetrahedron'%(len(obj_common))
				file_tmp='%s/obj_common.xyz'%(path)
				generator_xyz_dim4(obj_common,file_tmp)
				print '	  saved as %s/obj_common.xyz'%(path)
			else:
				print '   (obj_common) OBJ1 and OBJ2 : empty'
			#
			if obj_a.tolist()!=[[[[0]]]]:
				print '   (obj_a) OBJ1 and NOT OBJ2 : %d x tetrahedron'%(len(obj_a))
				file_tmp='%s/obj_a.xyz'%(path)
				generator_xyz_dim4(obj_a,file_tmp)
				print '	  saved as %s/obj_a.xyz'%(path)
			else:
				print '   (obj_a) OBJ1 and NOTOBJ2 : empty'
				#
			if obj_b.tolist()!=[[[[0]]]]:
				print '   (obj_b) NOT OBJ1 and OBJ2 : %d x tetrahedron'%(len(obj_b))
				file_tmp='%s/obj_b.xyz'%(path)
				generator_xyz_dim4(obj_b,file_tmp)
				print '	  saved as %s/obj_b.xyz'%(path)
			else:
				print '   (obj_b) NOT OBJ1 and OBJ2 : empty'
			return obj_common,obj_a,obj_b

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
	print 'Intersecting of OD1 and ODB'
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
		print ' No intersection'
		return np.array([0],dtype=np.int).reshape(1,1,1,1)
	else:
		point=p.reshape(len(p)/18,6,3) # 18=6*3
		point1=remove_doubling_dim3(point)
		print ' dividing into three PODs:'
		print '    Common   :     OD1 and     ODB'
		print '  UnCommon 1 :     OD1 and Not OD2'
		print '  UnCommon 2 : Not OD1 and     OD2'
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

cpdef np.ndarray object_subtraction(np.ndarray[np.int64_t, ndim=4] obj1,np.ndarray[np.int64_t, ndim=4] obj2):
	# This get an object = obj1 and not obj2 = obj1 and not (obj1 and obj2)
	cdef int i1,i2,i3,num
	cdef int counter1,counter2,counter3,counter4
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b,tmp1c,tmp1d
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a,tmp4b,tmp4c,tmp4d
	cdef list indx1
	counter2=0
	counter3=0
	counter4=0
	print 'Object_subtraction()'
	for i1 in range(len(obj1)):
		counter1=0
		indx1=[]
		tmp3a=obj1[i1]
		#####
		for i2 in range(len(obj2)):
			tmp3b=obj2[i2]
			#print 'hallo1'
			tmp4a,tmp4b=intersection_two_tetrahedron_new(tmp3a,tmp3b) # two objects after tetrahedralization
			if tmp4a.tolist()!=[[[[0]]]]:
				if counter3==0:
					tmp1a=tmp3a.reshape(72) # 4*6*3=72
				else:
					tmp1a=np.append(tmp1a,tmp3a)
				counter1+=1
				indx1.append(i2)
			else:
				counter1+=0
		######
		# tmp1c: set of tetrahedra in obj1 that are outseide of obj2
		#print 'hallo2'
		if counter1==0: # i1-th tetrahedron of obj1 is outside of obj2
			if counter2==0:
				tmp1c=tmp3a.reshape(72) # 72=4*6*3
			else:
				tmp1c=np.append(tmp1c,tmp3a)
			counter2+=1
		else:
			counter4+=1
		#####
		# if i1-th tetrahedron in obj1 (tmp3a) intersects with obj2
		#print 'hallo3'
		if counter1!=0:
			#####
			# generate set of tetrahedra in obj2 that intersect with tmp4b
			# -->> tmp4d:
			#print 'hallo4'
			for i3 in range(len(indx1)):
				if i3==0:
					num=indx1[0]
					tmp1d=obj2[num]
				else:
					num=indx1[i3]
					tmp1d=np.append(tmp1d,obj2[num])
			tmp4d=tmp1d.reshape(len(tmp1d)/72,4,6,3)
			#
			#
			# intersection_tetrahedron_object(tetrahedron,object)
			# intersection between a tetrahedron and an object (set of tetrahedra)
			#
			# tetrahedron = tmp3a
			# object = tmp4d
			#
			#print 'hallo5'
			tmp4c=intersection_tetrahedron_object(tmp3a,tmp4d)
			if counter3==0:
				tmp1b=tmp4c.reshape(len(tmp4c)/72,4,6,3)
			else:
				tmp1b=np.append(tmp1b,tmp4c)
			counter3+=1
			# merge two sets (tmp1c,tmp1b) of tetrahedra that are outseide of obj2
			tmp1c=np.append(tmp1c,tmp1b)
		else: #if i1 tetrahedron in obj1 (tmp3a) does NOT intersect with obj2
			pass
	#print 'hallo6'	
	if counter4+counter3!=0:
		return tmp1c.reshape(len(tmp1c)/72,4,6,3)
	else: # if (obj1 and not obj2) is empty
		return np.array([0],dtype=np.int).reshape(1,1,1,1)

# 	subtraction_tetrahedron_object() uses this.
cpdef intersection_two_tetrahedron_mod2(np.ndarray[np.int64_t, ndim=3] tetrahedron_1,\
											np.ndarray[np.int64_t, ndim=3] tetrahedron_2):
	# this returns intersecting points (dim3) of two tetrahedra
	cdef int i,i1,i2,j,count,counter1,counter2
	cdef np.ndarray[np.int64_t,ndim=1] p,tmp,tmp1,v1,v2,v3,v4,v5,v6,w1,w2,w3,w4,w5,w6
	cdef np.ndarray[np.int64_t,ndim=2] comb,tmp2a,tmp2b
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b,tri
	cdef np.ndarray[np.int64_t,ndim=4] obj1,obj2,obj_a,obj_b,obj_common
	#print ' intersection_two_tetrahedron_new()'
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
	counter1=0
	#print '    intersection_two_tetrahedron_mod2()'
	#for i in range(len(comb)): # len(comb) = 24
	for i in range(24):
		# case 1: intersection betweem
		# 6 edges of tetrahedron_1
		# 4 surfaces of tetrahedron_2
		segment_1=tetrahedron_1[comb[i][0]] 
		segment_2=tetrahedron_1[comb[i][1]]
		surface_1=tetrahedron_2[comb[i][2]]
		surface_2=tetrahedron_2[comb[i][3]]
		surface_3=tetrahedron_2[comb[i][4]]
		tmp=intersection_segment_surface(segment_1,segment_2,surface_1,surface_2,surface_3)
		if len(tmp)!=1:
			if counter1==0 :
				p=tmp # intersection
			else:
				p=np.append(p,tmp) # intersecting points
			counter1+=1
		else:
			pass
		# case 2: intersection betweem
		# 6 edges of tetrahedron_2
		# 4 surfaces of tetrahedron_1
		segment_1=tetrahedron_2[comb[i][0]]
		segment_2=tetrahedron_2[comb[i][1]]
		surface_1=tetrahedron_1[comb[i][2]]
		surface_2=tetrahedron_1[comb[i][3]]
		surface_3=tetrahedron_1[comb[i][4]]
		tmp=intersection_segment_surface(segment_1,segment_2,surface_1,surface_2,surface_3)
		if len(tmp)!=1:
			if counter1==0:
				p=tmp # intersection
			else:
				p=np.append(p,tmp) # intersecting points
			counter1+=1
		else:
			pass
	if counter1==0:
		#print '    no intersection(1)'
		return np.array([0],dtype=np.int).reshape(1,1,1)
		#return tetrahedron_1.reshape(1,4,6,3),tetrahedron_2.reshape(1,4,6,3) # return themselves
	else:
		# working in perp space. 
		# remove doubling in 'p' (intersecting points)
		# remove any points in 'p' which are also vertces of tetrahedron_1
		#print '    intersection:%d'%(counter1)
		#tmp3b=tmp.reshape(len(tmp)/18,6,3)
		tmp3b=tetrahedron_1
		tmp3a=p.reshape(len(p)/18,6,3)
		counter2=0
		for i1 in range(len(tmp3a)):
			tmp2a=tmp3a[i1]
			counter1=0
			for i2 in range(len(tmp3b)):
				tmp2b=tmp3b[i2]
				v1,v2,v3,v4,v5,v6=projection(tmp2a[0],tmp2a[1],tmp2a[2],tmp2a[3],tmp2a[4],tmp2a[5])
				w1,w2,w3,w4,w5,w6=projection(tmp2b[0],tmp2b[1],tmp2b[2],tmp2b[3],tmp2b[4],tmp2b[5])
				if np.all(v4==w4) and np.all(v5==w5) and np.all(v6==w6):
					counter1+=1
					break
				else:
					counter1+=0
			if counter1==0:
				if counter2==0:
					tmp1=np.array([tmp2a[0],tmp2a[1],tmp2a[2],tmp2a[3],tmp2a[4],tmp2a[5]]).reshape(18)
				else:
					tmp1=np.append(tmp1,[tmp2a[0],tmp2a[1],tmp2a[2],tmp2a[3],tmp2a[4],tmp2a[5]])
				counter2+=1
			else:
				pass
		if counter2!=0: # if there is intersecting points.
			#print '    intersection:%d -> %d'%(counter1,counter2)
			tmp3a=tmp1.reshape(len(tmp1)/18,6,3)
			return tmp3a
		else:
			#print '    no intersection(2)'
			return np.array([0],dtype=np.int).reshape(1,1,1)

# Developing
cpdef np.ndarray subtraction_tetrahedron_object(np.ndarray[np.int64_t, ndim=3] tetrahedron,\
												np.ndarray[np.int64_t, ndim=4] obj2,\
												np.ndarray[np.int64_t, ndim=3] point_b,\
												np.ndarray[np.int64_t, ndim=4] obj2_surface,\
												np.ndarray[np.int64_t, ndim=4] obj3_surface, \
												path):
	# this subtractes a tetrahedron from an object (set of tetrahedra)
	# return (dim4)
	#
	cdef int flag
	cdef int i1,i2
	cdef int counter0,counter1,counter2,counter3,counter4
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b,tmp1c,v1,v2,v3,v4,v5,v6
	cdef np.ndarray[np.int64_t,ndim=2] tmp2a,tmp2b
	#cdef np.ndarray[np.int64_t,ndim=2] v1,v2,v3,v4,v5,v6
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b,tmp3c,obj_surface_vertex
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a,obj_common

	#print 'Subtraction_tetrahedron_object()'
	#print 'Intersection_two_tetrahedron_mod2()'
		
	#####################################
	#                                   #
	#                                   #
	#####################################
	# (1) get vertices of tetrahedron which are outside of obj2 (points A)
	counter1=0
	for i1 in range(4):
		tmp2a=tetrahedron[i1]
		counter0=0
		for i2 in range(len(obj2)):
			tmp3a=obj2[i2]
			flag=inside_outside_tetrahedron(tmp2a,tmp3a[0],tmp3a[1],tmp3a[2],tmp3a[3])
			if flag==0: # inside
				counter0+=1
				break
			else:
				pass
		if counter0==0:
			if counter1==0:
				tmp1a=tmp2a.reshape(18)
			else:
				tmp1a=np.append(tmp1a,tmp2a)
			counter1+=1
		else:
			pass
	if counter1==0: # tetrahedron is inside obj3
		#print 'inconsistency: common part is empty?'
		return np.array([0],dtype=np.int).reshape(1,1,1)
	else:
		#print '    points A:'
		tmp3a=tmp1a.reshape(len(tmp1a)/18,6,3)
		tmp3a=remove_doubling_dim3_in_perp_space(tmp3a)
		#for i1 in range(len(tmp3a)):
		#	v1,v2,v3,v4,v5,v6=projection(tmp3a[i1][0],tmp3a[i1][1],tmp3a[i1][2],tmp3a[i1][3],tmp3a[i1][4],tmp3a[i1][5])
		#	print 'Xx %8.6f %8.6f %8.6f'%((v4[0]+TAU*v4[1])/float(v4[2]),(v5[0]+TAU*v5[1])/float(v5[2]),(v6[0]+TAU*v6[1])/float(v6[2]))
		#pass
		#if counter0!=0:
		#	tmp1a=np.append(tmp1a,tmp1b)
		#else:
		#	pass
	#####################################
	#                                   #
	#                                   #
	#####################################	
	"""  point_b
	# (2) get vertices of obj2_surface which are on obj3_surface (points B)
	tmp3b=obj2_surface.reshape(len(obj2_surface)/18,6,3)
	for i1 in range(len(tmp3b)):
		tmp2b=tmp3b[i1]
		for i2 in range(len(obj3_surface)):
			tmp3c=obj3_surface[i2]
			flag=on_out_surface(tmp2b,tmp3c)
			if flag==0: # on surface
				if counter0==0:
					tmp1b=tmp2b.reshape(18)
				else:
					tmp1b=np.append(tmp1b,tmp2b)
				counter0+=1
			else: # out
				pass
	print '    number of points B = %d'%(counter0)
	"""
	#####################################
	#                                   #
	#                                   #
	#####################################
	# (3) get points in (points B) which are inside tetrahedron -- > (points C)
	#tmp1b=np.append(tmp1b,obj3_surface)
	tmp1b=np.append(point_b,obj3_surface)
	tmp3b=tmp1b.reshape(len(tmp1b)/18,6,3)
	tmp3b=remove_doubling_dim3_in_perp_space(tmp3b)
	counter2=0
	for i1 in range(len(tmp3b)):
		tmp2b=tmp3b[i1]
		flag=inside_outside_tetrahedron(tmp2b,tetrahedron[0],tetrahedron[1],tetrahedron[2],tetrahedron[3])
		if flag==0: # inside
			if counter2==0:
				tmp1b=tmp2b.reshape(18)
			else:
				tmp1b=np.append(tmp1b,tmp2b)
			counter2+=1
		else:
			pass
	tmp3b=tmp1b.reshape(len(tmp1b)/18,6,3)
	#tmp3b=remove_doubling_dim3_in_perp_space(tmp3b)
	#print '    points C:'
	#for i1 in range(len(tmp3b)):
	#	v1,v2,v3,v4,v5,v6=projection(tmp3b[i1][0],tmp3b[i1][1],tmp3b[i1][2],tmp3b[i1][3],tmp3b[i1][4],tmp3b[i1][5])
	#	print 'Zz %8.6f %8.6f %8.6f'%((v4[0]+TAU*v4[1])/float(v4[2]),(v5[0]+TAU*v5[1])/float(v5[2]),(v6[0]+TAU*v6[1])/float(v6[2]))
	#####################################
	#                                   #
	#                                   #
	#####################################
	# (4) merge points A and points C
	if counter1==0 and counter2==0:
		return np.array([0],dtype=np.int).reshape(1,1,1)
	else:
		if counter1!=0 and counter2!=0:
			tmp1c=np.append(tmp1a,tmp1b)
			return tmp1c.reshape(len(tmp1c)/18,6,3)
		elif counter1!=0 and counter2==0:
			tmp1c=tmp1a
			return tmp1c.reshape(len(tmp1c)/18,6,3)
		elif counter1==0 and counter2!=0:
			tmp1c=tmp1b
			return tmp1c.reshape(len(tmp1c)/18,6,3)
		else:
			return np.array([0],dtype=np.int).reshape(1,1,1)

# Developing
cpdef np.ndarray subtraction_tetrahedron_object_dev(np.ndarray[np.int64_t, ndim=3] tetrahedron,\
													np.ndarray[np.int64_t, ndim=4] obj2,\
													np.ndarray[np.int64_t, ndim=3] point_b,\
													np.ndarray[np.int64_t, ndim=4] obj2_surface,\
													np.ndarray[np.int64_t, ndim=4] obj3_surface, \
													path):
	# this subtractes a tetrahedron from an object (set of tetrahedra)
	# return (dim4)
	#
	cdef int flag
	cdef int i1,i2
	cdef int counter0,counter1,counter2,counter3,counter4
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b,tmp1c,v1,v2,v3,v4,v5,v6
	cdef np.ndarray[np.int64_t,ndim=2] tmp2a,tmp2b
	#cdef np.ndarray[np.int64_t,ndim=2] v1,v2,v3,v4,v5,v6
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b,tmp3c,obj_surface_vertex
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a,obj_common

	print '  subtraction_tetrahedron_object()'
	print '    intersection_two_tetrahedron_mod2()'
	
	# common part between tetrahedron and obj3.
	#obj_common=intersection_using_tetrahedron(tetrahedron.reshape(1,4,6,3),obj3,path)
	#tmp1a=np.append(tmp1a,obj_common)
	#tmp3a=tmp1a.reshape(len(tmp1a)/18,6,3)
	#print '    number of common tetrahedra = %d'%(len(obj_common))
	
	"""
	# get vertices of common tetrahedra, which are on the obj3_surface.
	counter0=0
	tmp3b=remove_doubling_dim4(obj2)
	for i1 in range(len(tmp3b)) :
		tmp2b=tmp3b[i1]
		for i2 in range(len(obj2_surface)):
			tmp3c=obj2_surface[i2]
			flag=on_out_surface(tmp2b,tmp3c)
			if flag==0: # on surface
				if counter0==0:
					tmp1b=tmp2b.reshape(18)
				else:
					tmp1b=np.append(tmp1b,tmp2b)
				counter0+=1
			else: # out
				pass
	print '    number of vertices of common tetrahedra located on the surface = %d'%(counter0)
	if counter0!=0:
		tmp3b=tmp1b.reshape(len(tmp1b)/18,6,3)
		for i1 in range(len(tmp3b)):
			v1,v2,v3,v4,v5,v6=projection(tmp3b[i1][0],tmp3b[i1][1],tmp3b[i1][2],tmp3b[i1][3],tmp3b[i1][4],tmp3b[i1][5])
			print 'Zz %8.6f %8.6f %8.6f'%((v4[0]+TAU*v4[1])/float(v4[2]),(v5[0]+TAU*v5[1])/float(v5[2]),(v6[0]+TAU*v6[1])/float(v6[2]))
	else:
		pass
	"""
	#####################################
	#                                   #
	#                                   #
	#####################################
	# (1) get vertices of tetrahedron which are outside of obj2 (points A)
	counter1=0
	for i1 in range(4):
		tmp2a=tetrahedron[i1]
		for i2 in range(len(obj2)):
			tmp3a=obj2[i2]
			flag=inside_outside_tetrahedron(tmp2a,tmp3a[0],tmp3a[1],tmp3a[2],tmp3a[3])
			if flag!=0: # outside
				if counter1==0:
					tmp1a=tmp2a.reshape(18)
				else:
					tmp1a=np.append(tmp1a,tmp2a)
				counter1+=1
			else:
				pass
	if counter1==0: # tetrahedron is inside obj3
		#print 'inconsistency: common part is empty?'
		return np.array([0],dtype=np.int).reshape(1,1,1)
	else:
		pass
		#if counter0!=0:
		#	tmp1a=np.append(tmp1a,tmp1b)
		#else:
		#	pass
	#####################################
	#                                   #
	#                                   #
	#####################################	
	"""  point_b
	# (2) get vertices of obj2_surface which are on obj3_surface (points B)
	tmp3b=obj2_surface.reshape(len(obj2_surface)/18,6,3)
	for i1 in range(len(tmp3b)):
		tmp2b=tmp3b[i1]
		for i2 in range(len(obj3_surface)):
			tmp3c=obj3_surface[i2]
			flag=on_out_surface(tmp2b,tmp3c)
			if flag==0: # on surface
				if counter0==0:
					tmp1b=tmp2b.reshape(18)
				else:
					tmp1b=np.append(tmp1b,tmp2b)
				counter0+=1
			else: # out
				pass
	print '    number of points B = %d'%(counter0)
	"""
	#####################################
	#                                   #
	#                                   #
	#####################################
	# (3) get points in (points B) which are inside tetrahedron (points C)
	#tmp1b=np.append(tmp1b,obj3_surface)
	tmp1b=np.append(point_b,obj3_surface)
	tmp3b=tmp1b.reshape(len(tmp1b)/18,6,3)
	#tmp3b=remove_doubling_dim3_in_perp_space(tmp3b)
	counter2=0
	for i1 in range(len(tmp3b)):
		tmp2b=tmp3b[i1]
		flag=inside_outside_tetrahedron(tmp2b,tetrahedron[0],tetrahedron[1],tetrahedron[2],tetrahedron[3])
		if flag==0: # inside
			if counter2==0:
				tmp1b=tmp2a.reshape(18)
			else:
				tmp1b=np.append(tmp1b,tmp2b)
			counter2+=1
		else:
			pass
	tmp3b=tmp1b.reshape(len(tmp1b)/18,6,3)
	tmp3b=remove_doubling_dim3_in_perp_space(tmp3b)
	print '    points C:'
	#for i1 in range(len(tmp3b)):
	#	v1,v2,v3,v4,v5,v6=projection(tmp3b[i1][0],tmp3b[i1][1],tmp3b[i1][2],tmp3b[i1][3],tmp3b[i1][4],tmp3b[i1][5])
	#	print 'Zz %8.6f %8.6f %8.6f'%((v4[0]+TAU*v4[1])/float(v4[2]),(v5[0]+TAU*v5[1])/float(v5[2]),(v6[0]+TAU*v6[1])/float(v6[2]))
	#####################################
	#                                   #
	#                                   #
	#####################################
	# (4) merge points A and points C
	if counter1==0 and counter2==0:
		return np.array([0],dtype=np.int).reshape(1,1,1)
	else:
		if counter1!=0 and counter2!=0:
			tmp1c=np.append(tmp1a,tmp1b)
		elif counter1!=0 and counter2==0:
			tmp1c=tmp1a
		elif counter1==0 and counter2!=0:
			tmp1c=tmp1b
		else:
			pass
		print '    merge points A and points B -> tetrahedralization'
		tmp3c=tmp1c.reshape(len(tmp1c)/18,6,3)
		tmp3c=remove_doubling_dim3(tmp3c)
		if len(tmp3c)>=4:
			tmp4c=tetrahedralization_points(tmp3c)
			print '    number of tetrahedron = %d'%(len(tmp4c))
			#####################################
			#                                   #
			#                                   #
			#####################################
			# (5) get tetrahedra in tmp4c which are outside of obj2
			counter3=0
			for i1 in range(len(tmp4c)):
				counter4=0
				for i2 in range(4):
					tmp2c=tmp4c[i1][i2]
					for i3 in range(len(obj2)):
						tmp3a=obj2[i3]
						flag=inside_outside_tetrahedron(tmp2c,tmp3a[0],tmp3a[1],tmp3a[2],tmp3a[3])
						if flag==0: # inside
							counter4+=1
							break
						else:
							pass
				if counter4<4:
					if counter3==0:
						tmp1c=tmp4c[i1].reshape(72)
					else:
						tmp1c=np.append(tmp1c,tmp4c[i1])
					counter3+=1
				else:
					pass
			print '    number of tetrahedron outside of obj2 = %d'%(counter3)
			if counter3!=0:
				return tmp1c.reshape(len(tmp1c)/72,4,6,3)
			else:
				return np.array([0],dtype=np.int).reshape(1,1,1)
		else:
			return np.array([0],dtype=np.int).reshape(1,1,1)
			
		

			
	"""
	# get points of obj2_surface which are inside tetrahedon (points-A)
	obj_surface_vertex=remove_doubling_dim4(obj3_surface)
	counter1=0
	for i1 in range(len(obj_surface_vertex)):
		tmp2a=obj_surface_vertex[i1]
		flag=inside_outside_tetrahedron(tmp2a,tetrahedron[0],tetrahedron[1],tetrahedron[2],tetrahedron[3])
		if flag==0: #inside
			if counter1==0:
				tmp1a=tmp2a.reshape(18) # 18=6*3
			else:
				tmp1a=np.append(tmp1a,tmp2a)
			counter1+=1
		else:
			pass		
	print '    number of vertices of obj3_surface inside tetrahedron = %d'%(counter1)
	"""
					
					
	"""
	# add vertices of tetrahedron and tmp1b to points-A
	if counter0!=0:
		tmp1a=np.append(tmp1a,tmp1b)
	else:
		pass
	if counter1==0:
		print 'inconsistency: common part is empty?'
		return 1
	else:
		pass
	tmp1a=np.append(tetrahedron,tmp1a)
	tmp3a=tmp1a.reshape(len(tmp1a)/18,6,3)
	# Tetrahedralization
	tmp4a=tetrahedralization_points(tmp3a)
	print '    tetrahedralization(): %d * tetrahedron'%(len(tmp4a))
	#
	for i1 in range(len(tmp4a)):
		for i2 in range(4):
			v1,v2,v3,v4,v5,v6=projection(tmp4a[i1][i2][0],\
										tmp4a[i1][i2][1],\
										tmp4a[i1][i2][2],\
										tmp4a[i1][i2][3],\
										tmp4a[i1][i2][4],\
										tmp4a[i1][i2][5])
			print 'Xx %8.6f %8.6f %8.6f'%((v4[0]+TAU*v4[1])/float(v4[2]),\
										(v5[0]+TAU*v5[1])/float(v5[2]),\
										(v6[0]+TAU*v6[1])/float(v6[2]))
	# get tetrahedron which is outside of obj_common
	counter1=0
	for i1 in range(len(tmp4a)):
		counter3=0
		for i2 in range(4):
			tmp2a=tmp4a[i1][i2]
			counter2=0
			for i3 in range(len(obj_common)):
				tmp3b=obj_common[i3]
				flag=inside_outside_tetrahedron(tmp2a,tmp3b[0],tmp3b[1],tmp3b[2],tmp3b[3])
				if flag==0: # inside
					counter2+=1
					break
				else:
					pass
			if counter2==0:
				counter3+=1
			else:
				pass
		if counter3!=0: # i1-the tetrahedron in tmp4a is outside of obj_common
			if counter1==0:
				tmp1a=tmp4a[i1].reshape(72) # 4*6*3
			else:
				tmp1a=np.append(tmp1a,tmp4a[i1])
			counter1+=1
		else:
			pass
	if counter1!=0:
		return tmp1a.reshape(len(tmp1a)/72,4,6,3)
	else:
		return np.array([0],dtype=np.int).reshape(1,1,1,1)
	"""
"""
# NEW  but old...
cpdef np.ndarray subtraction_tetrahedron_object(np.ndarray[np.int64_t, ndim=3] tetrahedron,np.ndarray[np.int64_t, ndim=4] obj):
	# this subtractes a tetrahedron from an object (set of tetrahedra)
	# return (dim4)
	cdef int i1,i2,i3,i4,counter0,counter1,counter2,counter3,counter4,flag
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b,v1,v2,v3,v4,v5,v6
	cdef np.ndarray[np.int64_t,ndim=2] tmp2a
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b,tmp3c,tmp3d,tmp3e
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a,tmp4b,tmp4c

	print '  subtraction_tetrahedron_object()'
	print '    intersection_two_tetrahedron_mod2()'
	
	# get intersecting posints between 'tetrahedron' and 'obj'
	counter1=0
	for i1 in range(len(obj)):
		tmp3b=obj[i1]
		tmp3c=intersection_two_tetrahedron_mod2(tetrahedron,tmp3b)
		if tmp3c.tolist()!=[[[0]]]:
			if counter1==0:
				tmp1a=tmp3c.reshape(len(tmp3c)*18) # 18=6*3
			else:
				tmp1a=np.append(tmp1a,tmp3c)
			counter1+=1
		else:
			pass
	print '    number of intersecting points = %d'%(counter1)
		
	# get vertces of 'obj' which are inside 'tetrahedron', and intersecting points (tmp1a).
	if counter1!=0:
		#tmp3b=obj.reshape(len(obj)/72,4,6,3)
		tmp1a=np.append(obj,tmp1a)
		tmp3a=tmp1a.reshape(len(tmp1a)/18,6,3)
	else:
		tmp3a=obj.reshape(len(obj)*4,6,3)
	#tmp3a=remove_doubling_dim4_in_perp_space(tmp3a)
	tmp3a=remove_doubling_dim3_in_perp_space(tmp3a)
	#
	counter0=0
	for i1 in range(len(tmp3a)):
		flag=inside_outside_tetrahedron(tmp3a[i1],tetrahedron[0],tetrahedron[1],tetrahedron[2],tetrahedron[3])
		if flag==0: #inside
			if counter0==0:
				tmp1b=tmp3a[i1].reshape(18) # 18=6*3
			else:
				tmp1b=np.append(tmp1b,tmp3a[i1])
			counter0+=1
		else:
			pass		
	print '    number of vertices of obj which are inside tetrahedron = %d'%(counter0)

	if counter0!=0:
		#
		# get intersecting points and vertices of obj which are inside tetrahedron (not include vertex of tetrahedron)
		tmp3c=tmp1b.reshape(len(tmp1b)/18,6,3)
		counter1=0
		for i1 in range(len(tmp3c)):
			for i2 in range(4):
				if np.all(tmp3c[i1]==tetrahedron[i2][0]) or np.all(tmp3c[i1]==tetrahedron[i2][1]) or np.all(tmp3c[i1]==tetrahedron[i2][2]) or np.all(tmp3c[i1]==tetrahedron[i2][3]):
					break
				else:
					if counter1==0:
						tmp1c=tmp3c[i1].reshape(18)
					else:
						tmp1c=np.append(tmp1c,tmp3c[i1])
					counter1+=1
		if counter1!=0:
			tmp3c=tmp1c.reshape(len(tmp1c)/18,6,3)
		else:
			tmp3c=np.array([0]).reshape(1,1,1)
		
		#
		# get vertices of tetrahedron which is inside obj
		counter3=0
		for i1 in range(4):
			for i2 in range(len(obj)):
				tmp3a=obj[i2]
				flag=inside_outside_tetrahedron(tetrahedron[i1],tmp3a[0],tmp3a[1],tmp3a[2],tmp3a[3])
				if flag==0: # inside
					if counter3==0:
						tmp1a=tmp3a.reshape(72)
					else:
						tmp1a=np.append(tmp1a,tmp3a)
					counter3+=1
				else:
					pass
		if counter3!=0:
			tmp3a=tmp1a.reshape(len(tmp1a)/18,6,3)
		else:
			tmp3a=np.array([0]).reshape(1,1,1)
		
		if counter1!=0 and counter3!=0:
			tmp1a=np.append(tmp3b,tmp3a)
			tmp3a=tmp1a.reshape(len(tmp1a)/18,6,3)
			# Tetrahedralization
			tmp4a=tetrahedralization_points(tmp3a)
			print '    (1) tetrahedralization(): %d * tetrahedron'%(len(tmp4a))
			for i4 in range(len(tmp4a)):
				for i5 in range(4):
					v1,v2,v3,v4,v5,v6=projection(tmp4a[i4][i5][0],tmp4a[i4][i5][1],tmp4a[i4][i5][2],tmp4a[i4][i5][3],tmp4a[i4][i5][4],tmp4a[i4][i5][5])
					print 'Xx %8.6f %8.6f %8.6f'%((v4[0]+TAU*v4[1])/float(v4[2]),\
												(v5[0]+TAU*v5[1])/float(v5[2]),\
												(v6[0]+TAU*v6[1])/float(v6[2]))
		else:
			pass

		#tmp1b=np.append(tmp4c,tmp1b) # (intersection points and obj) inside tetrahedron
		tmp1d=np.append(tmp1b,tetrahedron) # (intersection points and obj) inside tetrahedron + tetrahedron
		tmp3d=tmp1d.reshape(len(tmp1d)/18,6,3)
		# Tetrahedralization
		tmp4d=tetrahedralization_points(tmp3d)
		print '    (2) tetrahedralization(): %d * tetrahedron'%(len(tmp4d))
		for i4 in range(len(tmp4d)):
			for i5 in range(4):
				v1,v2,v3,v4,v5,v6=projection(tmp4d[i4][i5][0],tmp4d[i4][i5][1],tmp4d[i4][i5][2],tmp4d[i4][i5][3],tmp4d[i4][i5][4],tmp4d[i4][i5][5])
				print 'Yy %8.6f %8.6f %8.6f'%((v4[0]+TAU*v4[1])/float(v4[2]),\
											(v5[0]+TAU*v5[1])/float(v5[2]),\
											(v6[0]+TAU*v6[1])/float(v6[2]))

		counter2=0
		for i1 in range(len(tmp4d)):
			tmp3d=tmp4d[i1]
			counter4=0
			for i2 in range(4):
				counter3=0
				for i3 in range(len(tmp4a)):
					tmp3a=tmp4a[i3]
					flag=inside_outside_tetrahedron(tmp3d[i2],tmp3a[0],tmp3a[1],tmp3a[2],tmp3a[3])
					#print flag
					if flag==0: # inside
						counter3+=1
						break
					else: # outdede
						pass
				if counter3!=0: # inside
					counter4+=1
				else:
					pass
			#print '       %2d-th tetrahedron: number of vertex inside obj: %d'%(i1,counter4)
			if counter4<4: # some vertces of i1-th tetrahedron in tmp4a (tmp3a) is outside of 'obj'
				for i4 in range(len(tmp3d)):
					v1,v2,v3,v4,v5,v6=projection(tmp3d[i4][0],tmp3d[i4][1],tmp3d[i4][2],tmp3d[i4][3],tmp3d[i4][4],tmp3d[i4][5])
					print '%8.6f %8.6f %8.6f'%((v4[0]+TAU*v4[1])/float(v4[2]),\
												(v5[0]+TAU*v5[1])/float(v5[2]),\
												(v6[0]+TAU*v6[1])/float(v6[2]))
				if counter2==0:
					tmp1d=tmp3d.reshape(72) # 4*6*3
				else:
					tmp1d=np.append(tmp1d,tmp3d)
				counter2+=1
			else:
				pass
		print '    number of tetrahedron outside of obj (except tetrahedra separated from obj) = %d'%(counter2)
		if counter2!=0:
			return tmp1d.reshape(len(tmp1d)/72,4,6,3)
		else:
			return np.array([0],dtype=np.int).reshape(1,1,1,1)
	else:
		return np.array([0],dtype=np.int).reshape(1,1,1,1)
"""

# NEW
cpdef np.ndarray object_subtraction_new(np.ndarray[np.int64_t, ndim=4] obj1,\
										np.ndarray[np.int64_t, ndim=4] obj2,\
										np.ndarray[np.int64_t, ndim=4] obj3,\
										path):
	# get A and not B = A and not (A and B)
	# obj1: A
	# obj2: A and B
	# obj3: B
	cdef int i1,i2,counter0,counter1,counter2,counter5,flag,flag1,flag2,num,verbose
	cdef long w1,w2,w3,v01,v02,v03
	cdef float vol
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b,v1,v2,v3,v4,v5,v6
	cdef np.ndarray[np.int64_t,ndim=2] tmp2a,tmp2b
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b,tmp3c,tetrahedron
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a,obj2_surface,obj3_surface
	
	verbose=0
	
	counter0=0
	counter1=0
	print ' object_subtraction_new()'
	#if verbose==1:
	#	print ' Subtraction by object_subtraction_new(obj1, obj2, obj3)'
	#	print '  get A not B = A not (A and B)'
	#	print '  obj1: A'
	#	print '  obj2: A and B'
	#	print '  obj3: B'
	#else:
	#	pass
	#
	if verbose==1:
		print '   (1) A and B'
	else:
		pass
	obj2_surface=generator_surface(obj2)
	if verbose==1:
		print '   (2) B'
	else:
		pass
	obj3_surface=generator_surface(obj3)
	"""
	#tmp3b=remove_doubling_dim4(obj2_surface)
	tmp3b=remove_doubling_dim4(obj3_surface)
	for i1 in range(len(tmp3b)):
		v1,v2,v3,v4,v5,v6=projection(tmp3b[i1][0],tmp3b[i1][1],tmp3b[i1][2],tmp3b[i1][3],tmp3b[i1][4],tmp3b[i1][5])
		print 'Zz %8.6f %8.6f %8.6f'%((v4[0]+TAU*v4[1])/float(v4[2]),(v5[0]+TAU*v5[1])/float(v5[2]),(v6[0]+TAU*v6[1])/float(v6[2]))
	"""
	if verbose==1:
		print '   (3) A not B'
	else:
		pass
	if obj2.tolist()!=[[[[0]]]]:
		##########
		# get vertices of obj2_surface which are on obj3_surface (points B)
		#    see subtraction_tetrahedron_object()
		tmp3b=obj2_surface.reshape(len(obj2_surface)*3,6,3)
		#tmp3b=remove_doubling_dim3_in_perp_space(tmp3b)
		counter0=0
		for i1 in range(len(tmp3b)):
			tmp2b=tmp3b[i1]
			for i2 in range(len(obj3_surface)):
				tmp3c=obj3_surface[i2]
				flag=on_out_surface(tmp2b,tmp3c)
				if flag==0: # on surface
					if counter0==0:
						tmp1b=tmp2b.reshape(18)
					else:
						tmp1b=np.append(tmp1b,tmp2b)
					counter0+=1
				else: # out
					pass
		if counter0!=0:
			tmp3b=tmp1b.reshape(len(tmp1b)/18,6,3)
		else:
			pass
		########### 
		#   TEMP  #
		###########
		tmp3b=remove_doubling_dim3_in_perp_space(tmp3b)
		#print '       number of points B = %d'%(len(tmp3b))
		#for i1 in range(len(tmp3b)):
		#	v1,v2,v3,v4,v5,v6=projection(tmp3b[i1][0],tmp3b[i1][1],tmp3b[i1][2],tmp3b[i1][3],tmp3b[i1][4],tmp3b[i1][5])
		#	print 'Yy %8.6f %8.6f %8.6f'%((v4[0]+TAU*v4[1])/float(v4[2]),(v5[0]+TAU*v5[1])/float(v5[2]),(v6[0]+TAU*v6[1])/float(v6[2]))
		###########
		#         #
		###########
		counter1=0
		for i1 in range(len(obj1)):
			#print '       %d-th tetrahedron in A'%(i1)
			tetrahedron=obj1[i1]
			############
			#  subtraction_tetrahedron_object()
			#  This returns vertecies which correspond to "A and not B = A and not (A and B)".
			############
			tmp3a=subtraction_tetrahedron_object(tetrahedron,obj2,tmp3b,obj2_surface,obj3_surface,path)
			if tmp3a.tolist()!=[[[0]]]:
				if counter1==0:
					tmp1a=tmp3a.reshape(len(tmp3a)*18) # 18=6*3
				else:
					tmp1a=np.append(tmp1a,tmp3a)
				counter1+=1
			else:
				pass
			############
			"""
			############
			#  subtraction_tetrahedron_object_dev()
			#  under construction
			#  This returns set of tetrahedra for "A and not B = A and not (A and B)".
			############
			tmp4a=subtraction_tetrahedron_object_dev(tetrahedron,obj2,tmp3b,obj2_surface,obj3_surface,path)
			if tmp4a.tolist()!=[[[[0]]]]:
				if counter1==0:
					tmp1a=tmp4a.reshape(len(tmp4a)*72) # 72=4*6*3
				else:
					tmp1a=np.append(tmp1a,tmp4a)
				counter1+=1
			else:
				pass
			############
			"""
		if counter1!=0:
			#return tmp1a.reshape(len(tmp1a)/72,4,6,3)
			tmp3a=tmp1a.reshape(len(tmp1a)/18,6,3)
			tmp3a=remove_doubling_dim3_in_perp_space(tmp3a)
			#for i1 in range(len(tmp3a)):
			#	v1,v2,v3,v4,v5,v6=projection(tmp3a[i1][0],tmp3a[i1][1],tmp3a[i1][2],tmp3a[i1][3],tmp3a[i1][4],tmp3a[i1][5])
			#	print 'Yy %8.6f %8.6f %8.6f'%((v4[0]+TAU*v4[1])/float(v4[2]),(v5[0]+TAU*v5[1])/float(v5[2]),(v6[0]+TAU*v6[1])/float(v6[2]))
			# return tmp4a
			######################
			#                    #
			# Tetrahedralization #
			#                    #
			######################
			counter3=0
			tmp4a=tetrahedralization_points(tmp3a)
			if tmp4a.tolist()!=[[[[0]]]]:
				for i1 in range(len(tmp4a)):
					tmp3a=tmp4a[i1]
					counter2=0
					#print 'tmp3a = ',tmp3a
					for i2 in range(4):
						tmp2a=tmp3a[i2]
						counter1=0
						for i3 in range(len(obj2)):
							tmp3b=obj2[i3]
							flag=inside_outside_tetrahedron(tmp2a,tmp3b[0],tmp3b[1],tmp3b[2],tmp3b[3])
							if flag==0: # inside
								counter1+=1
								break
							else: # outside 
								pass
						if counter1==0:
							counter2+=1
						else:
							pass
					if counter2<4: # outside
						if counter3==0:
							tmp1a=tmp3a.reshape(72) # 72=4*6*3
						else:
							tmp1a=np.append(tmp1a,tmp3a)
						counter3+=1
					else:
						pass
				tmp4a=tmp1a.reshape(len(tmp1a)/72,4,6,3)
				w1,w2,w3=obj_volume_6d(tmp4a)
				vol=(w1+w2*TAU)/float(w3)
				if verbose==1:
					print '   A not B obj, volume = %d %d %d (%10.6f)'%(w1,w2,w3,vol)
				else:
					pass
				"""
				#
				# check each tetrahedron in "tmp4a" is really inside "obj1" and out of "obj3"
				print '   checking common obj'
				counter5=0
				num=len(tmp4a)
				for i1 in range(len(tmp4a)):
					flag1=tetrahedron_inside_obj(tmp4a[i1],obj1)
					flag2=tetrahedron_inside_obj(tmp4a[i1],obj3)
					if flag1==0 and flag2==1:
						if counter5==0:
							tmp1a=tmp4a[i1].reshape(72)
						else:
							tmp1a=np.append(tmp1a,tmp4a[i1])
						counter5+=1
					else:
						pass
				tmp4a=tmp1a.reshape(len(tmp1a)/72,4,6,3)
				print '   numbre of tetrahedron: %d -> %d'%(num,counter5)
				v01,v02,v03=obj_volume_6d(tmp4a)
				vol=(v01+v02*TAU)/float(v03)
				print '   A not B obj, volume = %d %d %d (%10.6f)'%(v01,v02,v03,vol)
				"""
				return tmp4a
			else:
				return np.array([0],dtype=np.int).reshape(1,1,1,1)
		else:
			#return np.array([0],dtype=np.int).reshape(1,1,1,1)
			return np.array([0],dtype=np.int).reshape(1,1,1,1)
	else:
		return obj1	

# working
cpdef np.ndarray object_subtraction_dev1(np.ndarray[np.int64_t, ndim=4] obj1,\
										np.ndarray[np.int64_t, ndim=4] obj2,\
										np.ndarray[np.int64_t, ndim=4] obj3,\
										verbose):
	# get A and not B = A and not (A and B)
	# obj1: A
	# obj2: A and B
	cdef int flag,flag1,counter1,counter2,counter3
	cdef int i1,i2,i3
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b
	cdef np.ndarray[np.int64_t,ndim=2] tmp2a
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b
	cdef np.ndarray[np.int64_t,ndim=3] vertx_obj2
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a,tmp4b,tmp4c
	cdef np.ndarray[np.int64_t,ndim=4] surface_obj3
	
	print ' object_subtraction_dev1()'
	
	#if verbose==1:
	#	print '  get A not B = A not (A and B)'
	#	print '  obj1: A'
	#	print '  obj2: A and B'
	#	print '  obj3: B'
	#else:
	#	pass

	tmp1b=np.array([0])
	tmp4b=generator_surface(obj2)
	vertx_obj2=remove_doubling_dim3_in_perp_space(tmp4b.reshape(len(tmp4b)*3,6,3)) # vertices of obj2
	surface_obj3=generator_surface(obj3) # surface of obj3
	
	# get vertices of obj2 which on the surface of obj3
	tmp1a=np.array([0])
	counter2=0
	for i2 in range(len(vertx_obj2)):
		counter1=0
		for i3 in range(len(surface_obj3)):
			flag=on_out_surface(vertx_obj2[i2],surface_obj3[i3])
			if flag==0: # on
				counter1+=1
				break
			else:
				pass
		if counter1!=0:
			if counter2==0:
				tmp1a=vertx_obj2[i2].reshape(18)
			else:
				tmp1a=np.append(tmp1a,vertx_obj2[i2])
			counter2+=1
		else:
			pass

	vertx_obj2=remove_doubling_dim3_in_perp_space(tmp1a.reshape(len(tmp1a)/18,6,3)) # vertices of obj2
	#print 'len(vertx_obj2)=',len(vertx_obj2)
	
	counter3=0
	for i1 in range(len(obj1)):

		counter2=0

		# get vertices of i1-th tetrahedron of obj1 which are NOT inside obj3
		for i2 in range(4):
			counter1=0
			for i3 in range(len(obj3)):
				flag=inside_outside_tetrahedron(obj1[i1][i2],obj3[i3][0],obj3[i3][1],obj3[i3][2],obj3[i3][3])
				if flag==0: # insede
					counter1+=1
					break
				else:
					pass
			if counter1==0:
				if counter2==0:
					tmp1a=obj1[i1][i2].reshape(18)
				else:
					tmp1a=np.append(tmp1a,obj1[i1][i2])
				counter2+=1
			else:
				pass
		
		# get vertx_obj2 (vertices of obj2 on the surface of obj3) which are inside i1-th tetrahedron in obj1
		for i2 in range(len(vertx_obj2)):
			flag=inside_outside_tetrahedron(vertx_obj2[i2],obj1[i1][0],obj1[i1][1],obj1[i1][2],obj1[i1][3])
			if flag==0: # insede
				if counter2==0:
					tmp1a=vertx_obj2[i2].reshape(18)
				else:
					tmp1a=np.append(tmp1a,vertx_obj2[i2])
				counter2+=1
			else:
				pass
		
		if counter2!=0:
			tmp3b=remove_doubling_dim3_in_perp_space(tmp1a.reshape(len(tmp1a)/18,6,3))
			tmp1a=np.array([0])
			if len(tmp3b)>=4:
				if coplanar_check(tmp3b)==0:
					tmp4a=tetrahedralization_points(tmp3b)
					counter2=0
					for i2 in range(len(tmp4a)):
						# geometric center, centroid of the tetrahedron, tmp2c
						tmp2a=centroid(tmp4a[i2])
						counter1=0
						for i3 in range(len(obj2)):
							# check tmp2c is out of obj2 or not
							flag=inside_outside_tetrahedron(tmp2a,obj2[i3][0],obj2[i3][1],obj2[i3][2],obj2[i3][3])
							if flag==0: # inside
								counter1+=1
								break
							else:
								pass
						if counter1==0:
							if counter2==0:
								tmp1a=tmp4a[i2].reshape(72)
							else:
								tmp1a=np.append(tmp1a,tmp4a[i2])
							counter2+=1
						else:
							pass
					if counter2!=0:
						if counter3==0:
							tmp1b=tmp1a
						else:
							tmp1b=np.append(tmp1b,tmp1a)
						counter3+=1
					else:
						pass
				else:
					pass
			else:
				pass
		else:
			pass
	if counter3!=0:
		return tmp1b.reshape(len(tmp1b)/72,4,6,3)
	else:
		return np.array([0]).reshape(1,1,1,1)

# working
cpdef np.ndarray object_subtraction_dev(np.ndarray[np.int64_t, ndim=4] obj1,\
										np.ndarray[np.int64_t, ndim=4] obj2,\
										np.ndarray[np.int64_t, ndim=4] obj3,\
										verbose):
	# get A and not B = A and not (A and B)
	# obj1: A
	# obj2: A and B
	cdef int flag,flag1,counter1,counter2,counter3
	cdef int i1,i2,i3
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b
	cdef np.ndarray[np.int64_t,ndim=2] tmp2a
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a,tmp4b
	
	print ' object_subtraction_dev()'
	
	#if verbose==1:
	#	print '  get A not B = A not (A and B)'
	#	print '  obj1: A'
	#	print '  obj2: A and B'
	#else:
	#	pass
	tmp1a=np.array([0])
	tmp1b=np.array([0])
	tmp4b=generator_surface(obj2)
	tmp3a=remove_doubling_dim3_in_perp_space(tmp4b.reshape(len(tmp4b)*3,6,3)) # vertices of obj2
	for i1 in range(len(obj1)):
		# get vertices of obj2 which are inside i1-th tetrahedron of obj1
		for i2 in range(len(tmp3a)):
			flag=inside_outside_tetrahedron(tmp3a[i2],obj1[i1][0],obj1[i1][1],obj1[i1][2],obj1[i1][3])
			if flag==0: # insede
				if len(tmp1a)==1:
					tmp1a=tmp3a[i2].reshape(18)
				else:
					tmp1a=np.append(tmp1a,tmp3a[i2])
			else:
				pass
		# get vertices of i1-th tetrahedron of obj1 which are NOT inside obj2
		for i2 in range(4):
			counter1=0
			for i3 in range(len(obj2)):
				flag=inside_outside_tetrahedron(obj1[i1][i2],obj2[i2][0],obj2[i2][1],obj2[i2][2],obj2[i2][3])
				if flag==0: # insede
					counter1+=1
					break
				else:
					pass
			if counter1==0:
				if len(tmp1a)==1:
						tmp1a=obj1[i1][i2].reshape(18)
				else:
					tmp1a=np.append(tmp1a,obj1[i1][i2])
			else:
				pass
		if len(tmp1a)!=1:
			tmp3b=remove_doubling_dim3_in_perp_space(tmp1a.reshape(len(tmp1a)/18,6,3))
			tmp1a=np.array([0])
			if len(tmp3b)>=4:
				tmp4a=tetrahedralization_points(tmp3b)
				for i2 in range(len(tmp4a)):
					# geometric center, centroid of the tetrahedron, tmp2c
					tmp2a=centroid(tmp4a[i2])
					counter1=0
					for i3 in range(len(obj2)):
						# check tmp2c is out of obj2 or not
						flag=inside_outside_tetrahedron(tmp2a,obj2[i3][0],obj2[i3][1],obj2[i3][2],obj2[i3][3])
						if flag==0: # inside
							counter1+=1
							break
						else:
							pass
					if counter1==0:
						if len(tmp1a)==1:
							tmp1a=tmp4a[i2].reshape(72)
						else:
							tmp1a=np.append(tmp1a,tmp4a[i2])
					else:
						pass
				if len(tmp1b)==1:
					tmp1b=tmp1a
				else:
					tmp1b=np.append(tmp1b,tmp1a)
			else:
				pass
		else:
			pass
	if len(tmp1b)!=1:
		tmp4a=tmp1b.reshape(len(tmp1b)/72,4,6,3)
		# もう一度、obj2との交点を求め、四面体分割。そして、obj2に含まれない部分を取り出す。
		# ただし、複雑に凹凸部分と重なる部分では、obj2に含まれる部分を取り除くことができないので注意
		counter1=0
		for i1 in range(len(obj1)):
			for i2 in range(len(tmp4a)):
				tmp3a=intersection_two_tetrahedron_0(obj1[i1],tmp4a[i2])
				if tmp3a.tolist()!=[[[0]]]:
					if counter1==0:
						tmp1c=tmp3a.reshape(len(tmp3a)*18)
					else:
						tmp1c=np.append(tmp1c,tmp3a)
					counter1+=1
				else:
					pass
		tmp1c=np.append(tmp4a,tmp1c)
		tmp3a=remove_doubling_dim3_in_perp_space(tmp1c.reshape(len(tmp1c)/18,6,3))
		tmp4a=tetrahedralization_points(tmp3a)
		tmp1a=np.array([0])
		counter1=0
		for i2 in range(len(tmp4a)):
			counter2=0
			for i3 in range(4) :
				for i4 in range(len(obj2)):
					flag=inside_outside_tetrahedron(tmp4a[i2][i3],obj2[i4][0],obj2[i4][1],obj2[i4][2],obj2[i4][3])
					if flag==0: # inside
						counter2+=1
						break
					else:
						pass
			if counter2==4:
				# geometric center, centroid of the tetrahedron, tmp2c
				tmp2a=centroid(tmp4a[i2])
				for i5 in range(len(obj2)):
					flag1=inside_outside_tetrahedron(tmp2a,obj2[i5][0],obj2[i5][1],obj2[i5][2],obj2[i5][3])
					if flag1==0: # inside
						counter3+=1
						break
					else:
						pass
				if counter3==0:
					if counter1==0:
						tmp1a=tmp4a[i2].reshape(72)
					else:
						tmp1a=np.append(tmp1a,tmp4a[i2])
					counter1+=1
				else:
					pass
			else:
				if counter1==0:
					tmp1a=tmp4a[i2].reshape(72)
				else:
					tmp1a=np.append(tmp1a,tmp4a[i2])
				counter1+=1
		# obj2の凹凸部分との共通部分を除くために、分割した四面体ごとに、再度、obj3との交点を求める。
		# 交点があるならば、その点を含めて四面体分割する。
		# そして、各四面体の重心を用いて、obj1 not obj 2 かを判断
		if counter1!=0:
			tmp1b=np.array([0])
			tmp4a=tmp1a.reshape(len(tmp1a)/72,4,6,3)
			for i3 in range(len(tmp4a)):
				tmp1c=np.array([0])
				for i4 in range(len(obj3)):
					tmp3a=intersection_two_tetrahedron_0(tmp4a[i3],obj3[i4])
					if tmp3a.tolist()!=[[[0]]]:
						if len(tmp1c)==1:
							tmp1c=tmp3a.reshape(len(tmp3a)*18)
						else:
							tmp1c=np.append(tmp1c,tmp3a)
					else:
						pass
				if len(tmp1c)!=1:
					tmp3a=remove_doubling_dim3_in_perp_space(tmp1c.reshape(len(tmp1c)/18,6,3))
					tmp4b=tetrahedralization(tmp4a[i3],tmp3a)
					for i4 in range(len(tmp4b)):
						tmp2a=centroid(tmp4b[i4])
						counter3=0
						for i5 in range(len(obj2)):
							flag1=inside_outside_tetrahedron(tmp2a,obj2[i5][0],obj2[i5][1],obj2[i5][2],obj2[i5][3])
							if flag1==0: # inside
								counter3+=1
								break
							else:
								pass
						if counter3==0:
							if len(tmp1b)==1:
								tmp1b=tmp4b[i4].reshape(72)
							else:
								tmp1b=np.append(tmp1b,tmp4b[i4])
						else:
							pass
				else:
					pass
			if len(tmp1b)!=0:
				return tmp1b.reshape(len(tmp1b)/72,4,6,3)
			else:
				return np.array([0]).reshape(1,1,1,1)
		else:
			return np.array([0]).reshape(1,1,1,1)
	else:
		return np.array([0]).reshape(1,1,1,1)

# working
cpdef np.ndarray object_subtraction_dev2(np.ndarray[np.int64_t, ndim=4] obj1,\
										np.ndarray[np.int64_t, ndim=4] obj2,\
										np.ndarray[np.int64_t, ndim=4] obj3,\
										verbose):
	# get A and not B = A and not (A and B)
	# obj1: A
	# obj2: A and B
	cdef int flag,flag1,counter1,counter2,counter3
	cdef int i1,i2,i3
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b
	cdef np.ndarray[np.int64_t,ndim=2] tmp2a
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b
	cdef np.ndarray[np.int64_t,ndim=3] vertx_obj2
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a,tmp4b,tmp4c
	cdef np.ndarray[np.int64_t,ndim=4] surface_obj3
	
	print ' object_subtraction_dev2()'
	
	#if verbose==1:
	#	print '  get A not B = A not (A and B)'
	#	print '  obj1: A'
	#	print '  obj2: A and B'
	#	print '  obj3: B'
	#else:
	#	pass

	tmp1b=np.array([0])
	tmp4b=generator_surface(obj2)
	vertx_obj2=remove_doubling_dim3_in_perp_space(tmp4b.reshape(len(tmp4b)*3,6,3)) # vertices of obj2
	surface_obj3=generator_surface(obj3) # surface of obj3
	
	# get vertices of obj2 which on the surface of obj3
	tmp1a=np.array([0])
	counter2=0
	for i2 in range(len(vertx_obj2)):
		counter1=0
		for i3 in range(len(surface_obj3)):
			flag=on_out_surface(vertx_obj2[i2],surface_obj3[i3])
			if flag==0: # on
				counter1+=1
				break
			else:
				pass
		if counter1!=0:
			if counter2==0:
				tmp1a=vertx_obj2[i2].reshape(18)
			else:
				tmp1a=np.append(tmp1a,vertx_obj2[i2])
			counter2+=1
		else:
			pass

	vertx_obj2=remove_doubling_dim3_in_perp_space(tmp1a.reshape(len(tmp1a)/18,6,3)) # vertices of obj2
	#print 'len(vertx_obj2)=',len(vertx_obj2)
	
	# get vertices of obj1 which are NOT inside obj3
	counter2=0
	for i1 in range(len(obj1)):
		for i2 in range(4):
			counter1=0
			for i3 in range(len(obj3)):
				flag=inside_outside_tetrahedron(obj1[i1][i2],obj3[i3][0],obj3[i3][1],obj3[i3][2],obj3[i3][3])
				if flag==0: # insede
					counter1+=1
					break
				else:
					pass
			if counter1==0:
				if counter2==0:
					tmp1a=obj1[i1][i2].reshape(18)
				else:
					tmp1a=np.append(tmp1a,obj1[i1][i2])
				counter2+=1
			else:
				pass
	
	# get vertices of obj2 on the surface of obj3 which are inside obj1
	for i1 in range(len(obj1)):
		for i2 in range(len(vertx_obj2)):
			flag=inside_outside_tetrahedron(vertx_obj2[i2],obj1[i1][0],obj1[i1][1],obj1[i1][2],obj1[i1][3])
			if flag==0: # insede
				if counter2==0:
					tmp1a=vertx_obj2[i2].reshape(18)
				else:
					tmp1a=np.append(tmp1a,vertx_obj2[i2])
				counter2+=1
			else:
				pass
	
	if len(tmp1a)!=1:
		counter3=0
		tmp3b=remove_doubling_dim3_in_perp_space(tmp1a.reshape(len(tmp1a)/18,6,3))
		tmp1a=np.array([0])
		if len(tmp3b)>=4:
			if coplanar_check(tmp3b)==0:
				tmp4a=tetrahedralization_points(tmp3b)
				counter2=0
				for i2 in range(len(tmp4a)):
					# geometric center, centroid of the tetrahedron, tmp2c
					tmp2a=centroid(tmp4a[i2])
					counter1=0
					for i3 in range(len(obj2)):
						# check tmp2c is out of obj2 or not
						flag=inside_outside_tetrahedron(tmp2a,obj2[i3][0],obj2[i3][1],obj2[i3][2],obj2[i3][3])
						if flag==0: # inside
							counter1+=1
							break
						else:
							pass
					if counter1==0:
						if counter2==0:
							tmp1a=tmp4a[i2].reshape(72)
						else:
							tmp1a=np.append(tmp1a,tmp4a[i2])
						counter2+=1
					else:
						pass
				if counter2!=0:
					if counter3==0:
						tmp1b=tmp1a
					else:
						tmp1b=np.append(tmp1b,tmp1a)
					counter3+=1
				else:
					pass
			else:
				pass
		if counter3!=0:
			return tmp1b.reshape(len(tmp1b)/72,4,6,3)
		else:
			return np.array([0]).reshape(1,1,1,1)
	else:
		return np.array([0]).reshape(1,1,1,1)

cpdef np.ndarray object_not_object(np.ndarray[np.int64_t, ndim=4] obj1,\
									np.ndarray[np.int64_t, ndim=4] obj2):
	# get A not B
	# obj1: A
	# obj2: B
	cdef int i1
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a	
	tmp1a=np.array([0])
	for i1 in range(len(obj1)):
		tmp4a=tetrahedron_not_obj(obj1[i1],obj2)
		if tmp4a.tolist()!=[[[[0]]]]:
			if len(tmp1a)==1:
				tmp1a=tmp4a.reshape(len(tmp4a)*72)
			else:
				tmp1a=np.append(tmp1a,tmp4a)
		else:
			pass
	if len(tmp1a)!=1:
		return tmp1a.reshape(len(tmp1a)/72,4,6,3)
	else:
		return np.array([0]).reshape(1,1,1,1)

cpdef np.ndarray tetrahedron_not_obj(np.ndarray[np.int64_t, ndim=3] tetrahedron,\
									np.ndarray[np.int64_t, ndim=4] obj):
	#
	# tetrahedron_not_obj()
	#
	# Parameters:
	# (1) tetrahedron
	# (2) obj
	#
	# Teturn:
	# tetrahedron NOT obj (a set of tetrahedron)
	#
	cdef int i1,counter1
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b,tmp1c
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a
	
	tmp1a=np.array([0])
	tmp1b=np.array([0])
	for i1 in range(len(obj)):
		tmp3a=intersection_two_tetrahedron_0(tetrahedron,obj[i1])
		if tmp3a.tolist()!=[[[0]]]:
			if len(tmp1b)==1:
				tmp1b=tmp3a.reshape(len(tmp3a)*18)
			else:
				tmp1b=np.append(tmp1b,tmp3a)
		else:
			pass
	if len(tmp1b)!=1:
		tmp3a=remove_doubling_dim3_in_perp_space(tmp1b.reshape(len(tmp1b)/18,6,3))
		tmp4a=tetrahedralization(tetrahedron,tmp3a)
		for i1 in range(len(tmp4a)):
			tmp2a=centroid(tmp4a[i1])
			counter1=0
			for i2 in range(len(obj)):
				flag1=inside_outside_tetrahedron(tmp2a,obj[i2][0],obj[i2][1],obj[i2][2],obj[i2][3])
				if flag1==0: # inside
					counter1+=1
					break
				else:
					pass
			if counter1==0:
				if len(tmp1a)==1:
					tmp1a=tmp4a[i1].reshape(72)
				else:
					tmp1a=np.append(tmp1a,tmp4a[i1])
			else:
				pass
		if len(tmp1a)!=1:
			return tmp1a.reshape(len(tmp1a)/72,4,6,3)
		else:
			return np.array([0]).reshape(1,1,1,1)
	else:
		return np.array([0]).reshape(1,1,1,1)

cpdef np.ndarray intersection_two_tetrahedron_0(np.ndarray[np.int64_t, ndim=3] tetrahedron_1,\
												np.ndarray[np.int64_t, ndim=3] tetrahedron_2):
	cdef int i1,i2,counter1,counter2
	cdef long a1,b1,c1
	cdef float vol1
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b,tmp1c,v1,v2,v3,v4,v5,v6
	cdef np.ndarray[np.int64_t,ndim=2] comb
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b
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

	#print '     intersection_two_tetrahedron_3()'
	tmp1a=np.array([0])
	tmp1b=np.array([0])
	tmp1c=np.array([0])

	counter1=0
	for i1 in range(len(comb)): # len(combination_index) = 24
		# case 1: intersection betweem
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
		# case 2: intersection betweem
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
	if counter1!=0:
		return tmp1a.reshape(len(tmp1a)/18,6,3)
	else:
		return np.array([0]).reshape(1,1,1)
			
			
"""
cpdef object_subtraction_1(np.ndarray[np.int64_t, ndim=4] obj1,\
							np.ndarray[np.int64_t, ndim=4] obj2):
	# get A and not B = A and not (A and B)
	# obj1: A
	# obj2: A and B
	# obj3: B
	cdef int i1,i2,i3,i4,i5,counter0,counter1,counter2,counter3,counter4,counter5,counter6,flag1
	cdef long v01,v02,v03
	cdef float vol
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b,tmp1c,tmp1d
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a
	cdef np.ndarray[np.int64_t,ndim=4] obj2_surface

	print ' object_subtraction_1()'
	print '  get A not B = A not (A and B)'
	print '  obj1: A'
	print '  obj2: A and B'
	#
	# get surfaces of obj2
	obj2_surface=generator_surface(obj2)
	tmp3a=obj2_surface.reshape(len(obj2_surface)*3,6,3)
	tmp3a=remove_doubling_dim3_in_perp_space(tmp3a)
	#
	tmp1a=np.array([0])
	if obj2.tolist()!=[[[[0]]]]:
		counter0=0
		counter3=0
		for i1 in range(len(obj1)):
			counter2=0
			for i2 in range(4):
				counter1=0
				for i3 in range(len(obj2)):
					flag1=inside_outside_tetrahedron(obj1[i1][i2],obj2[i3][0],obj2[i3][1],obj2[i3][2],obj2[i3][3])
					if flag1==0: # inside
						counter1+=1
						break
					else:
						pass
				if counter1!=0:
					if counter2==0:
						tmp1b=obj1[i1][i2].reshape(18)
					else:
						tmp1b=np.append(tmp1b,obj1[i1][i2])
					counter2+=1
				else:
					pass
			if counter2==4: # i1-th tetrahedron is inside obj2
				pass
			elif counter2==0: # i1-th tetrahedron is out of obj2
				if counter0==0:
					tmp1a=obj1[i1].reshape(72)
				else:
					tmp1a=np.append(tmp1a,obj1[i1])
				counter0+=1
			else: # i1-th tetrahedron is partially out of obj2
				# tmp1b: vertices of i1-th tetrahedron which is out of obj2
				# get vertices of obj2 which are inside i1-th tetrahedron, and merge them into tmp1b, then do tetrahedralization
				#
				counter4=0
				for i3 in range(len(tmp3a)):
					flag1=inside_outside_tetrahedron(tmp3a[i3],obj1[i1][0],obj1[i1][1],obj1[i1][2],obj1[i1][3])
					if flag1==0: # inside
						if counter4==0:
							tmp1d=tmp3a[i3].reshape(18)
						else:
							tmp1d=np.append(tmp1d,tmp3a[i3])
						counter4+=1
					else:
						pass
				if counter4!=0:
					tmp1b=np.append(tmp1b,tmp1d)
					tmp3b=tmp1b.reshape(len(tmp1b)/18,6,3)
					# Tetrahedralization
					tmp4a=tetrahedralization_points(tmp3b)
					# check each tetrahedron in tmp4a is not inside obj2
					for i4 in range(len(tmp4a)):
						counter5=0
						for i5 in range(4):
							counter6=0
							for i6 in range(len([obj2])):
								flag1=inside_outside_tetrahedron(tmp4a[i4][i5],obj2[i6][0],obj2[i6][1],obj2[i6][2],obj2[i6][3])
								if flag1==0: # inside
									counter6+=1
									break
								else:
									pass
							if counter6==0:
								counter5+=1
							else:
								pass
						if counter5!=4:
							if counter0==0:
								tmp1a=tmp4a[i4].reshape(72)
							else:
								tmp1a=np.append(tmp1a,tmp4a[i4])
							counter0+=1
				else:
					pass
		if tmp1a.tolist()!=[0]:
			tmp4a=tmp1a.reshape(len(tmp1a)/72,4,6,3)
			v01,v02,v03=obj_volume_6d(tmp4a)
			vol=(v01+v02*TAU)/float(v03)
			print '   A not B obj, volume = %d %d %d (%10.6f)'%(v01,v02,v03,vol)
			return tmp4a
		else:
			return tmp1a.reshape(1,1,1,1)
	else:
		return obj1	
"""

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
	
cpdef np.ndarray object_subtraction_2(np.ndarray[np.int64_t, ndim=4] obj1,\
									np.ndarray[np.int64_t, ndim=4] obj2,\
									int verbose):
	# get A and not B = A and not (A and B)
	# obj1: A
	# obj2: A and B
	# obj3: B
	cdef int i1,i2,i3,i4,i5,flag1,counter1,counter2
	cdef long v01,v02,v03
	cdef float vol
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b,tmp1c
	cdef np.ndarray[np.int64_t,ndim=2] tmp2c
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a
	cdef np.ndarray[np.int64_t,ndim=4] obj2_surface
	
	print ' object_subtraction_2()'
	#if verbose==1:
	#	print '  get A not B = A not (A and B)'
	#	print '  obj1: A'
	#	print '  obj2: A and B'
	#else:
	#	pass

	#
	tmp1a=np.array([0])
	if obj2.tolist()!=[[[[0]]]]:
	
		# get surfaces of onj2
		#obj2_surface=generator_surface(obj2)
		obj2_surface=generator_surface_1(obj2)
		#
		tmp3a=obj2_surface.reshape(len(obj2_surface)*3,6,3)
		tmp3a=remove_doubling_dim3_in_perp_space(tmp3a)
	
		for i1 in range(len(obj1)):
			tmp1b=np.array([0])
			for i2 in range(len(tmp3a)):
				flag1=inside_outside_tetrahedron(tmp3a[i2],obj1[i1][0],obj1[i1][1],obj1[i1][2],obj1[i1][3])
				if flag1==0: # inside
					if len(tmp1b)==1:
						tmp1b=tmp3a[i2].reshape(18)
					else:
						tmp1b=np.append(tmp1b,tmp3a[i2])
				else:
					pass
			if len(tmp1b)!=1:
				# Tetrahedralization
				tmp4a=tetrahedralization(obj1[i1],tmp1b.reshape(len(tmp1b)/18,6,3))
				for i3 in range(len(tmp4a)):
					#"""
					### Algorithm 1 ###
					# geometric center, centroid of the tetrahedron, tmp2c
					tmp2c=centroid(tmp4a[i3])
					counter2=0
					for i5 in range(len(obj2)):
						# check tmp2c is out of obj2 or not
						flag1=inside_outside_tetrahedron(tmp2c,obj2[i5][0],obj2[i5][1],obj2[i5][2],obj2[i5][3])
						if flag1==0: # inside
							counter2+=1
							break
						else:
							pass
					if counter2==0:
						if len(tmp1a)==1:
							tmp1a=tmp4a[i3].reshape(72)
						else:
							tmp1a=np.append(tmp1a,tmp4a[i3])
					else:
						pass
					#"""
					"""
					### Algorithm 2 ###
					counter1=0
					for i4 in range(4):
						for i5 in range(len(obj2)):
							flag1=inside_outside_tetrahedron(tmp4a[i3][i4],obj2[i5][0],obj2[i5][1],obj2[i5][2],obj2[i5][3])
							if flag1==0: # inside
								counter1+=1
								break
							else:
								pass
					if counter1!=4:
						if len(tmp1a)==1:
							tmp1a=tmp4a[i3].reshape(72)
						else:
							tmp1a=np.append(tmp1a,tmp4a[i3])
					else:
						# geometric center, centroid of the tetrahedron, tmp2c
						tmp2c=centroid(tmp4a[i3])
						counter2=0
						for i5 in range(len(obj2)):
							# check tmp2c is out of obj2 or not
							flag1=inside_outside_tetrahedron(tmp2c,obj2[i5][0],obj2[i5][1],obj2[i5][2],obj2[i5][3])
							if flag1==0: # inside
								counter2+=1
								break
							else:
								pass
						if counter2==0:
							if len(tmp1a)==1:
								tmp1a=tmp4a[i3].reshape(72)
							else:
								tmp1a=np.append(tmp1a,tmp4a[i3])
						else:
							pass
					"""	
			else:
				if len(tmp1a)==1:
					tmp1a=obj1[i1].reshape(72)
				else:
					tmp1a=np.append(tmp1a,obj1[i1])
		if len(tmp1a)!=1:
			tmp4a=tmp1a.reshape(len(tmp1a)/72,4,6,3)
			if verbose==1:
				v01,v02,v03=obj_volume_6d(tmp4a)
				vol=(v01+v02*TAU)/float(v03)
				print '   A not B obj, volume = %d %d %d (%10.6f)'%(v01,v02,v03,vol)
			else:
				pass
			return tmp4a
		else:
			return tmp1a.reshape(1,1,1,1)
	else:
		return obj1	

cpdef np.ndarray object_subtraction_3(np.ndarray[np.int64_t, ndim=4] obj1,\
							np.ndarray[np.int64_t, ndim=4] obj2):
	# get A and not B = A and not (A and B)
	# obj1: A
	# obj2: A and B
	# obj3: B
	cdef int i1,i2,i3,i4,i5,counter0,counter1,counter2,counter3,counter4,counter5,counter6,counter7,flag1
	cdef long v01,v02,v03
	cdef float vol
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b,tmp1c,tmp1d
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a
	cdef np.ndarray[np.int64_t,ndim=4] obj2_surface
	
	print ' object_subtraction_3()'
	#print '  get A not B = A not (A and B)'
	#print '  obj1: A'
	#print '  obj2: A and B'

	tmp1a=np.array([0])
	if obj2.tolist()!=[[[[0]]]]:
	
		# get surfaces of obj2
		obj2_surface=generator_surface(obj2)
		tmp3a=obj2_surface.reshape(len(obj2_surface)*3,6,3)
		tmp3a=remove_doubling_dim3_in_perp_space(tmp3a)
		
		counter0=0
		counter3=0
		for i1 in range(len(obj1)):
			counter2=0
			for i2 in range(4):
				counter1=0
				for i3 in range(len(obj2)):
					flag1=inside_outside_tetrahedron(obj1[i1][i2],obj2[i3][0],obj2[i3][1],obj2[i3][2],obj2[i3][3])
					if flag1==0: # inside
						counter1+=1
						break
					else:
						pass
				if counter1!=0:
					if counter2==0:
						tmp1b=obj1[i1][i2].reshape(18)
					else:
						tmp1b=np.append(tmp1b,obj1[i1][i2])
					counter2+=1
				else:
					pass
			if counter2==4: # i1-th tetrahedron is inside obj2
				tmp2c=centroid(obj1[i1])
				counter7=0
				for i3 in range(len(obj2)):
					flag1=inside_outside_tetrahedron(tmp2c,obj2[i3][0],obj2[i3][1],obj2[i3][2],obj2[i3][3])
					if flag1==0:
						counter7+=1
					else:
						pass
				if counter7==0:
					if counter0==0:
						tmp1a=obj1[i1].reshape(72)
					else:
						tmp1a=np.append(tmp1a,obj1[i1])
					counter0+=1
			elif counter2==0: # i1-th tetrahedron is out of obj2
				if counter0==0:
					tmp1a=obj1[i1].reshape(72)
				else:
					tmp1a=np.append(tmp1a,obj1[i1])
				counter0+=1
			else: # i1-th tetrahedron is partially out of obj2
				# tmp1b: vertices of i1-th tetrahedron which is out of obj2
				# get vertices of obj2 which are inside i1-th tetrahedron, and merge them into tmp1b, then do tetrahedralization
				#
				counter4=0
				for i3 in range(len(tmp3a)):
					flag1=inside_outside_tetrahedron(tmp3a[i3],obj1[i1][0],obj1[i1][1],obj1[i1][2],obj1[i1][3])
					if flag1==0: # inside
						if counter4==0:
							tmp1d=tmp3a[i3].reshape(18)
						else:
							tmp1d=np.append(tmp1d,tmp3a[i3])
						counter4+=1
					else:
						pass
				if counter4!=0:
					tmp1b=np.append(tmp1b,tmp1d)
					tmp3b=tmp1b.reshape(len(tmp1b)/18,6,3)
					# Tetrahedralization
					tmp4a=tetrahedralization_points(tmp3b)
					if tmp4a.tolist()!=[[[[0]]]]:
						# check each tetrahedron in tmp4a is not inside obj2
						for i4 in range(len(tmp4a)):
							tmp2c=centroid(tmp4a[i4])
							counter7=0
							for i5 in range(len(obj2)):
								flag1=inside_outside_tetrahedron(tmp2c,obj2[i5][0],obj2[i5][1],obj2[i5][2],obj2[i5][3])
								if flag1==0: # inside
									pass
								else:
									if counter0==0:
										tmp1a=tmp4a[i4].reshape(72)
									else:
										tmp1a=np.append(tmp1a,tmp4a[i4])
									counter0+1
							"""
							counter5=0
							for i5 in range(4):
								counter6=0
								for i6 in range(len([obj2])):
									flag1=inside_outside_tetrahedron(tmp4a[i4][i5],obj2[i6][0],obj2[i6][1],obj2[i6][2],obj2[i6][3])
									if flag1==0: # inside
										counter6+=1
										break
									else:
										pass
								if counter6==0:
									counter5+=1
								else:
									pass
							if counter5!=4:
								if counter0==0:
									tmp1a=tmp4a[i4].reshape(72)
								else:
									tmp1a=np.append(tmp1a,tmp4a[i4])
								counter0+=1
							else:
								pass
							"""
					else:
						pass
				else:
					pass
		if tmp1a.tolist()!=[0]:
			tmp4a=tmp1a.reshape(len(tmp1a)/72,4,6,3)
			v01,v02,v03=obj_volume_6d(tmp4a)
			vol=(v01+v02*TAU)/float(v03)
			print '   A not B obj, volume = %d %d %d (%10.6f)'%(v01,v02,v03,vol)
			return tmp4a
		else:
			return tmp1a.reshape(1,1,1,1)
	else:
		return obj1	
	
"""
# NEW
cpdef object_subtraction_new(np.ndarray[np.int64_t, ndim=4] obj1,np.ndarray[np.int64_t, ndim=4] obj2,np.ndarray[np.int64_t, ndim=4] obj3):
	# get A and not B = A and not (A and B)
	# obj1: A
	# obj2: A and B
	# obj3: B
	cdef int i1,counter1
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tetrahedron
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a
	counter1=0
	print 'object_subtraction_new()'
	obj3_surface=generator_surface(obj3)
	#tmp3a=remove_doubling_dim4(obj3_surface)
	for i1 in range(len(obj1)):
		print '%d-th tetrahedron'%(i1)
		tetrahedron=obj1[i1]
		tmp4a=subtraction_tetrahedron_object(tetrahedron,obj2,obj3_surface)
		if tmp4a.tolist()!=[[[[0]]]]:
			if counter1==0:
				tmp1a=tmp4a.reshape(len(tmp4a)*72) # 72=4*6*3
			else:
				tmp1a=np.append(tmp1a,tmp4a)
			counter1+=1
		else:
			pass
	if counter1!=0:
		return tmp1a.reshape(len(tmp1a)/72,4,6,3)
	else:
		return np.array([0],dtype=np.int).reshape(1,1,1,1)
"""
cpdef np.ndarray intersection_tetrahedron_object(np.ndarray[np.int64_t, ndim=3] tetrahedron,np.ndarray[np.int64_t, ndim=4] obj):
	# intersection between a tetrahedron and an object (set of tetrahedra)
	cdef int i1,i2,counter1,counter2,flag
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a
	cdef np.ndarray[np.int64_t,ndim=1] tmp2a
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a,tmp4b,tmp4c
	counter1=0
	counter2=0
	print 'intersection_tetrahedron_object()'
	for i1 in range(len(obj)):
		tmp3b=obj[i1]
		tmp4a,tmp4b=intersection_two_tetrahedron_new(tetrahedron,tmp3b) # two objects after tetrahedralization
		if counter1==0:
			tmp1a=tmp4a.reshape(72) # 72=4*6*3
		else:
			tmp1a=np.append(tmp1a,tmp4a)
		counter1+=1
	#print 'hallo7'
	tmp4a=tmp1a.reshape(len(tmp1a)/72,4,6,3)
	#tmp3a=remove_doubling_dim4(tmp4a)
	tmp3a=remove_doubling_dim4_in_perp_space(tmp4a)
	for i1 in range(len(tmp3a)):
		tmp2a=tmp3a[i1]
		counter1=0
		for i2 in range(len(tetrahedron)):
			tmp2b=tetrahedron[i2]
			if np.all(tmp2a==tmp2b):
				counter1+=1
				break
			else:
				pass
		if counter2==0:
			tmp1a=tmp2a.reshape(18) # 18=6*3
		else:
			tmp1a=np.append(tmp1a,tmp2a)
		counter2+=1
	tmp3a=tmp1a.reshape(len(tmp1a)/18,6,3)
	tmp4a=tetrahedralization(tetrahedron,tmp3a)
	counter2=0
	#print 'hallo8'
	for i1 in range(len(tmp4a)):
		tmp3a=tmp4a[i1]
		counter1=0
		for i2 in range(4):
			flag=inside_outside_tetrahedron(tmp3a[i2],tetrahedron[0],tetrahedron[1],tetrahedron[2],tetrahedron[3])
			if flag==0: # indede
				counter1+=1
				break
			else: # outdede
				pass
		if counter1<4: # i1-th tetrahedron in tmp4a is outside of 'tetrahedron'
			if counter2==0:
				tmp1a=tmp3a.reshape(72) # 4*6*3
			else:
				tmp1a=np.append(tmp1a,tmp3a)
	return tmp1a.reshape(len(tmp1a)/72,4,6,3)

	##### NEW VERSION of intersection_using_tetrahedron() ######
cpdef intersection_using_tetrahedron_new(np.ndarray[np.int64_t, ndim=4] obj1,np.ndarray[np.int64_t, ndim=4] obj2):
	# Common part between two sets of tetrahedra (obj1 and obj2)
	cdef int i1,i2,i3,counter1,counter2,counter3,counter4,counter5,counter6,counter7,counter8,counter9
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b,tmp1c,tmp1d,tmp1e,tmp1f,tmp1_common
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a,tmp4b,tmp4c,tmp4d,tmp4e,tmp4f,tmp4_common,tmp4a_common,tmp4b_common
	cdef list lst
	counter1=0
	counter2=0
	counter4=0
	counter6=0
	counter7=0
	counter8=0
	print ' intersection_two_tetrahedron_new()'
	for i1 in range(len(obj1)):
		counter3=0
		lst=[]
		for i2 in range(len(obj2)):
			tmp4a,tmp4b=intersection_two_tetrahedron_new(obj1[i1],obj2[i2]) # two objects after tetrahedralization
			if tmp4a.tolist()!=[[[[0]]]]:
				if counter1==0:
					tmp1a=tmp4a.reshape(72*len(tmp4a)) # 72=4*6*3
				else:
					tmp1a=np.append(tmp1a,tmp4a)
				counter1+=1
				counter3+=1
				lst.append(i2) # save index i2
			else: # if no intersection
				counter3+=0
		################
		if len(lst)!=0:
			#print 'hallo1'
			counter9=0
			tmp4c=obj1[i1].reshape(1,4,6,3)
			for i2 in lst:
				for i3 in range(len(tmp4c)):
					#print 'tmp4c[i3]= '
					#print tmp4c[i3]
					#print 'obj2[i2]='
					#print obj2[i2]
					tmp4a,tmp4b=intersection_two_tetrahedron_new(tmp4c[i3],obj2[i2])
					if tmp4a.tolist()!=[[[[0]]]]:
						if counter9==0:
							tmp1e=tmp4a.reshape(len(tmp4a)*72)
						else:
							tmp1e=np.append(tmp1e,tmp4a)
						counter9+=1
					else:
						pass
				tmp4c=tmp1e.reshape(len(tmp1e)/72,4,6,3)
			#print 'hallo2'
			if counter7==0:
				tmp1e=tmp4c.reshape(len(tmp4c)*72)
			else:
				tmp1e=np.append(tmp1e,tmp4c)
			counter7+=1
		else:
			pass
		################
		if counter3==0: # i1-th tetrahedron of obj1 is outside of obj2
			if counter4==0:
				tmp1c=obj1[i1].reshape(72) # 72=4*6*3
			else:
				tmp1c=np.append(tmp1c,obj1[i1])
			counter4+=1
		else:
			pass
	for i2 in range(len(obj2)):
		counter5=0
		lst=[]
		for i1 in range(len(obj1)):
			tmp4a,tmp4b=intersection_two_tetrahedron_new(obj1[i1],obj2[i2]) # two objects after tetrahedralization
			if tmp4b.tolist()!=[[[[0]]]]:
				if counter2==0:
					tmp1b=tmp4b.reshape(72*len(tmp4b)) # 72=4*6*3
				else:
					tmp1b=np.append(tmp1b,tmp4b)
				counter2+=1
				counter5+=1
				lst.append(i1) # save index i1
			else: # if no intersection
				counter5+=0
		################
		if len(lst)!=0:
			#print 'hallo3'
			counter9=0
			tmp4d=obj2[i2].reshape(1,4,6,3)
			for i1 in lst:
				for i3 in range(len(tmp4d)):
					tmp4a,tmp4b=intersection_two_tetrahedron_new(tmp4d[i3],obj1[i1])
					if tmp4a.tolist()!=[[[[0]]]]:
						if counter9==0:
							tmp1f=tmp4a.reshape(len(tmp4a)*72)
						else:
							tmp1f=np.append(tmp1f,tmp4a)
					else:
						pass
				tmp4d=tmp1f.reshape(len(tmp1f)/72,4,6,3)
			#print 'hallo4'
			if counter8==0:
				tmp1f=tmp4d.reshape(len(tmp4d)*72)
			else:
				tmp1f=np.append(tmp1f,tmp4d)
			counter8+=1
		else:
			pass
		################
		if counter5==0: # i2-th tetrahedron of obj2 is outside of obj1
			if counter6==0:
				tmp1d=obj2[i2].reshape(72) # 72=4*6*3
			else:
				tmp1d=np.append(tmp1d,obj2[i2])
			counter6+=1
		else:
			pass
	#if counter1==0:
		#tmp4a=np.array([0],dtype=np.int).reshape(1,1,1,1)
	#print 'hallo5'
	if counter7==0:
		tmp4e=np.array([0],dtype=np.int).reshape(1,1,1,1)
	else:
		#tmp4a=tmp1a.reshape(len(tmp1a)/72,4,6,3)
		#tmp4a_common,tmp4a_non_common=decompose_into_common_non_common_obj(tmp4a,obj2)
		tmp4e=tmp1e.reshape(len(tmp1e)/72,4,6,3)
		tmp4a_common,tmp4a_non_common=decompose_into_common_non_common_obj(tmp4e,obj2)
		if counter4!=0:
			tmp1c=np.append(tmp4a_non_common,tmp1c)
			tmp4a_non_common=tmp1c.reshape(len(tmp1c)/72,4,6,3)
		else:
			pass
	#print 'hallo6'
	#if counter2==0:
		#tmp4b=np.array([0],dtype=np.int).reshape(1,1,1,1)
	if counter8==0:
		tmp4f=np.array([0],dtype=np.int).reshape(1,1,1,1)
	else:
		#tmp4b=tmp1b.reshape(len(tmp1b)/72,4,6,3)
		#tmp4b_common,tmp4b_non_common=decompose_into_common_non_common_obj(tmp4b,obj1)
		tmp4f=tmp1f.reshape(len(tmp1f)/72,4,6,3)
		tmp4b_common,tmp4b_non_common=decompose_into_common_non_common_obj(tmp4f,obj1)
		if counter6!=0:
			tmp1d=np.append(tmp4b_non_common,tmp1d)
			tmp4b_non_common=tmp1d.reshape(len(tmp1d)/72,4,6,3)
		else:
			pass
	#print 'counter1,7=',counter1,counter7
	#print 'counter2,8=',counter2,counter8
	#print 'hallo7'
	#print 'len(tmp4a_common),len(tmp4n_common) =',len(tmp4a_common),len(tmp4b_common)
	# merge common and non-common parts
	#if counter1!=0 and counter2!=0:
	if counter7!=0 and counter8!=0:
		tmp1_common=np.append(tmp4a_common,tmp4b_common)
		tmp4_common=tmp1_common.reshape(len(tmp1_common)/72,4,6,3)
	#elif counter1!=0 and counter2==0:
	elif counter7!=0 and counter8==0:
		tmp4_common=tmp4a_common
	#elif counter1==0 and counter2!=0:
	elif counter7==0 and counter8!=0:
		tmp4_common=tmp4b_common
	else: # if common part is empty 
		tmp4_common=np.array([0],dtype=np.int).reshape(1,1,1,1)
	return tmp4_common,tmp4a_non_common,tmp4b_non_common

cpdef decompose_into_common_non_common_obj(np.ndarray[np.int64_t, ndim=4] obj1,\
											np.ndarray[np.int64_t, ndim=4] obj2):
	# this compare obj1 and obj2 and decompose into two objects:
	# obj_common	: obj1 and obj2
	# obj_non_common: obj1 and NOT obj2
	cdef int i1,i2,i3,flag,counter1,counter2,counter3,counter4
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,obj_common,obj_non_common
	cdef np.ndarray[np.int64_t,ndim=2] tmp2a
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a,tmp4b
	print ' decompose_into_common_non_common_obj()'
	counter1=0
	counter3=0
	for i1 in range(len(obj1)):
		counter4=0
		tmp3a=obj1[i1]
		for i2 in range(4):
			counter2=0
			tmp2a=tmp3a[i2] # check whether each vertces of i1-th tetrahedron in obj1 is inside or outsede of obj2
			for i3 in range(len(obj2)):
				tmp3b=obj2[i3]
				flag=inside_outside_tetrahedron(tmp2a,tmp3b[0],tmp3b[1],tmp3b[2],tmp3b[3])
				#print 'flag=%d'%(flag)
				if flag==0:
					counter2+=1
					break
				else:
					pass
			if counter2==1: # if i2-th vertex of i1-th tetrahedron is inside obj2. 
				counter4+=1
			else:
				pass
		#print 'hello 1'
		#print '  counter4 = ',counter4
		#tmp1a=np.append(tmp3a[0],tmp3a[1])
		#tmp1a=np.append(tmp1a,tmp3a[2])
		#tmp1a=np.append(tmp1a,tmp3a[3])
		tmp1a=tmp3a.reshape(72)# 4*6*3 = 72
		#print ' hello 2'
		if counter4==4: # all four vertces of i1-th tetrahedron in obj1 is inside obj2.
			if counter1==0:
				obj_common=tmp1a
			else:
				obj_common=np.append(obj_common,tmp1a)
			counter1+=1
		else: # some vertces of i1-th tetrahedron in obj1 is outside obj2.
			if counter3==0:
				#print ' hello 4'
				obj_non_common=tmp1a
				#print ' hello 5'
			else:
				#print ' hello 6'
				obj_non_common=np.append(obj_non_common,tmp1a)
				#print ' hello 7'
			counter3+=1
		#print ' hello 3'
	if counter1==0:
		tmp4a=np.array([0],dtype=np.int).reshape(1,1,1,1)
	else:
		tmp4a=obj_common.reshape(len(obj_common)/72,4,6,3)
	if counter3==0:
		tmp4b=np.array([0],dtype=np.int).reshape(1,1,1,1)
	else:
		tmp4b=obj_non_common.reshape(len(obj_non_common)/72,4,6,3)
	return tmp4a,tmp4b

	##### NEW VERSION of intersection_two_tetrahedron() ######
cpdef intersection_two_tetrahedron_new(np.ndarray[np.int64_t, ndim=3] tetrahedron_1,\
											np.ndarray[np.int64_t, ndim=3] tetrahedron_2):
#cpdef intersection_two_tetrahedron(np.ndarray[np.int64_t, ndim=3] tetrahedron_1,np.ndarray[np.int64_t, ndim=3] tetrahedron_2):
	cdef int i,i1,i2,j,count,counter1,counter2
	cdef np.ndarray[np.int64_t,ndim=1] p,tmp,tmp1,v1,v2,v3,v4,v5,v6,w1,w2,w3,w4,w5,w6
	cdef np.ndarray[np.int64_t,ndim=2] comb,tmp2a,tmp2b
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b,tri
	cdef np.ndarray[np.int64_t,ndim=4] obj1,obj2,obj_a,obj_b,obj_common
	#print ' intersection_two_tetrahedron_new()'
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
	counter1=0
	#for i in range(len(comb)): # len(comb) = 24
	for i in range(24):
		# case 1: intersection betweem
		# 6 edges of tetrahedron_1
		# 4 surfaces of tetrahedron_2
		segment_1=tetrahedron_1[comb[i][0]] 
		segment_2=tetrahedron_1[comb[i][1]]
		surface_1=tetrahedron_2[comb[i][2]]
		surface_2=tetrahedron_2[comb[i][3]]
		surface_3=tetrahedron_2[comb[i][4]]
		tmp=intersection_segment_surface(segment_1,segment_2,surface_1,surface_2,surface_3)
		if len(tmp)!=1:
			if counter1==0 :
				p=tmp # intersection
			else:
				p=np.append(p,tmp) # intersecting points
			counter1+=1
		else:
			pass
		# case 2: intersection betweem
		# 6 edges of tetrahedron_2
		# 4 surfaces of tetrahedron_1
		segment_1=tetrahedron_2[comb[i][0]]
		segment_2=tetrahedron_2[comb[i][1]]
		surface_1=tetrahedron_1[comb[i][2]]
		surface_2=tetrahedron_1[comb[i][3]]
		surface_3=tetrahedron_1[comb[i][4]]
		tmp=intersection_segment_surface(segment_1,segment_2,surface_1,surface_2,surface_3)
		if len(tmp)!=1:
			if counter1==0:
				p=tmp # intersection
			else:
				p=np.append(p,tmp) # intersecting points
			counter1+=1
		else:
			pass
	if counter1==0:
		return np.array([0],dtype=np.int).reshape(1,1,1,1),np.array([0],dtype=np.int).reshape(1,1,1,1)
		#return tetrahedron_1.reshape(1,4,6,3),tetrahedron_2.reshape(1,4,6,3) # return themselves
	else:
		# working in perp space. 
		# remove doubling in 'p' (intersecting points)
		# remove any points in 'p' which are also vertces of tetrahedron_1 and/or tetrahedron_2
		tmp=np.append(tetrahedron_1,tetrahedron_2)
		tmp3b=tmp.reshape(len(tmp)/18,6,3)
		tmp3a=p.reshape(len(p)/18,6,3)
		counter2=0
		for i1 in range(len(tmp3a)):
			tmp2a=tmp3a[i1]
			counter1=0
			for i2 in range(len(tmp3b)):
				tmp2b=tmp3b[i2]
				v1,v2,v3,v4,v5,v6=projection(tmp2a[0],tmp2a[1],tmp2a[2],tmp2a[3],tmp2a[4],tmp2a[5])
				w1,w2,w3,w4,w5,w6=projection(tmp2b[0],tmp2b[1],tmp2b[2],tmp2b[3],tmp2b[4],tmp2b[5])
				if np.all(v4==w4) and np.all(v5==w5) and np.all(v6==w6):
					counter1+=1
					break
				else:
					counter1+=0
			if counter1==0:
				if counter2==0:
					tmp1=np.array([tmp2a[0],tmp2a[1],tmp2a[2],tmp2a[3],tmp2a[4],tmp2a[5]]).reshape(18)
				else:
					tmp1=np.append(tmp1,[tmp2a[0],tmp2a[1],tmp2a[2],tmp2a[3],tmp2a[4],tmp2a[5]])
				counter2+=1
			else:
				pass
		#if counter1!=0: # if there is intersecting points.
		if counter2!=0: # if there is intersecting points.
			#tmp3a=tmp1.reshape(counter1,6,3)
			tmp3a=tmp1.reshape(len(tmp1)/18,6,3)
			#for i in range(len(tmp3a)):
			#	z1,z2,z3,z4,z5,z6=projection(tmp3a[i][0],tmp3a[i][1],tmp3a[i][2],tmp3a[i][3],tmp3a[i][4],tmp3a[i][5])
			#	print 'Xx %8.6f %8.6f %8.6f'%((z4[0]+TAU*z4[1])/float(z4[2]),(z5[0]+TAU*z5[1])/float(z5[2]),(z6[0]+TAU*z6[1])/float(z6[2]))
			#------------------------
			#   Tetrahedralization
			#------------------------
			obj1=tetrahedralization(tetrahedron_1,tmp3a) # Tetrahedralization of tetrahedron_1
			obj2=tetrahedralization(tetrahedron_2,tmp3a) # Tetrahedralization of tetrahedron_2
			#if obj1.tolist()!=[[[[0]]]] and obj2.tolist()!=[[[[0]]]]:
			#	return obj1,obj2
			#else:
			#	return np.array([0],dtype=np.int).reshape(1,1,1,1),np.array([0],dtype=np.int).reshape(1,1,1,1)
			return obj1,obj2
		else:
			return np.array([0],dtype=np.int).reshape(1,1,1,1),np.array([0],dtype=np.int).reshape(1,1,1,1)
			# return themselves
			#return tetrahedron_1.reshape(1,4,6,3),tetrahedron_2.reshape(1,4,6,3)

#cpdef np.ndarray intersection_using_tetrahedron(np.ndarray[np.int64_t, ndim=4] obj1,np.ndarray[np.int64_t, ndim=4] obj2):
cpdef intersection_using_tetrahedron(np.ndarray[np.int64_t, ndim=4] obj1,np.ndarray[np.int64_t, ndim=4] obj2,path):
	# Common part between two sets of tetrahedra (obj1 and obj2)
	cdef int i1,i2,i3,i4,counter1,counter2,counter3,q1,q2,q3,flag
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1c
	cdef np.ndarray[np.int64_t,ndim=2] tmp2a
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b,tmp3c
	cdef np.ndarray[np.int64_t,ndim=4] tmp4_common,tmp4c
	print ' intersection_using_tetrahedron()'
	counter1=0
	for i1 in range(len(obj1)):
		tmp3a=obj1[i1]
		for i2 in range(len(obj2)):
			tmp3b=obj2[i2]
			tmp4_common=intersection_two_tetrahedron(tmp3a,tmp3b)
			if tmp4_common.tolist()!=[[[[0]]]]: # if common part is not empty
				if counter1==0:
					tmp1c=tmp4_common.reshape(72*len(tmp4_common)) # 72=4*6*3
				else:
					tmp1c=np.append(tmp1c,tmp4_common)
				counter1+=1
	if counter1==0: # if common part (obj1 and obj2) is empty
		tmp4_common=np.array([0],dtype=np.int).reshape(1,1,1,1)
	else: # if common part (obj1 and obj2) is NOT empty
		tmp4_common=tmp1c.reshape(len(tmp1c)/72,4,6,3)
		file_tmp='%s/intersecting_point.xyz'%(path)
		generator_xyz_dim4(tmp4_common,file_tmp)
		"""
		for i1 in range(len(tmp4_common)):
			tmp3a=tmp4_common[i1]
			[q1,q2,q3]=tetrahedron_volume_6d(tmp3a)
			print '  %3d-th tetrahedron, volume: %d,%d,%d'%(i1,q1,q2,q3)
		"""
		##########################
		#                        #
		# Re-tetrahedralization  #
		#                        #
		##########################
		tmp3c=remove_doubling_dim4_in_perp_space(tmp4_common)
		tmp4c=tetrahedralization_points(tmp3c)
		"""
		counter3=0
		for i2 in range(len(tmp4c)):
			counter2=0
			for i3 in range(4):
				tmp2a=tmp4c[i2][i3]
				for i4 in range(len(tmp4_common)):
					tmp3b=tmp4_common[i4]
					flag=inside_outside_tetrahedron(tmp2a,tmp3b[0],tmp3b[1],tmp3b[2],tmp3b[3])
					if flag==0:
						counter2+=1
						break
					else:
						pass
			if counter2==4:
				if counter3==0:
					tmp1a=tmp4c[i2].reshape(72) # 72=4*6*3
				else:
					tmp1a=np.append(tmp1a,tmp4c[i2])
			else:
				pass
		tmp4_common=tmp1a.reshape(len(tmp1a)/72,4,6,3)	
		"""
		tmp4_common=tmp4c
		##########################
		#                        #
		#                        #
		##########################
	return tmp4_common

cpdef intersection_using_tetrahedron_2(np.ndarray[np.int64_t, ndim=4] obj1,np.ndarray[np.int64_t, ndim=4] obj2,int option,int verbose,path):
	#
	# Common part between two sets of tetrahedra (obj1 and obj2)
	#
	cdef int i1,i2,i3,i4,counter1,counter2,counter3,counter4,counter5,q1,q2,q3,flag,num,flag1,flag2
	cdef int option1
	cdef long v01,v02,v03,v1,v2,v3,w1,w2,w3
	cdef double vol1,vol2,vol3,vol4a,vol4b,vol5a,vol5b
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b,tmp1c
	cdef np.ndarray[np.int64_t,ndim=2] tmp2a
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b,tmp3c,tmp3d
	cdef np.ndarray[np.int64_t,ndim=4] tmp4_common,tmp4a,tmp4b,tmp4c,tmp4d
	print ' intersection_using_tetrahedron_2()'
	
	option1=0
	#option1=2 #check each tetrahedron before finishing.
	
	option2=0
	#option2=1 # Remove unnecessary decompositions
	
	#verbose=0
	
	if verbose==1:
		v01,v02,v03=obj_volume_6d(obj1)
		vol1=(v01+v02*TAU)/float(v03)
		print '   obj1, volume = %d %d %d (%10.6f)'%(v01,v02,v03,vol1)
		v01,v02,v03=obj_volume_6d(obj2)
		vol2=(v01+v02*TAU)/float(v03)
		print '   obj2, volume = %d %d %d (%10.6f)'%(v01,v02,v03,vol2)
	else:
		pass
	
	counter3=0
	tmp4_common=np.array([0],dtype=np.int).reshape(1,1,1,1)
	for i1 in range(len(obj1)):
		tmp3a=obj1[i1]
		counter1=0
		
		# volume check
		w1,w2,w3=tetrahedron_volume_6d(tmp3a)
		vol5a=(w1+w2*TAU)/float(w3)
		if verbose==1:
			print '  %2d-th tetrahedron in obj1, %d %d %d (%10.6f)'%(i1,w1,w2,w3,vol5a)
		else:
			pass
			
		for i2 in range(len(obj2)):
			#print '    i2',i2
			tmp3b=obj2[i2]
			
			#tmp4_common=intersection_two_tetrahedron(tmp3a,tmp3b)
			# Attention (2020.05.26)
			#
			# Deu to "error3" in find_common_obj_2(), common part having very small volume
			# may be judged as empty.
			#
			# Probabily, this problem is deu to numerical calculation performed for this.
			#
			# In addition, intersection_two_tetrahedron() returns "tmp4_common" whose volume 
			# is out of expected value.
			#
			# Related functions:
			#  intersection_two_tetrahedron()
			#  find_common_obj_2()
			
			tmp4_common=intersection_two_tetrahedron_2(tmp3a,tmp3b)
			
			if tmp4_common.tolist()!=[[[[0]]]]: # if common part is not empty
				#print '  tmp4_common=',tmp4_common
				if counter1==0:
					tmp1c=tmp4_common.reshape(72*len(tmp4_common)) # 72=4*6*3
				else:
					tmp1c=np.append(tmp1c,tmp4_common)
				counter1+=1
				#print '   counter1=',counter1
			else:
				pass
		#print '  counter1 = ',counter1
		if counter1==0: # if common part (obj1 and obj2) is empty
			if verbose==1:
				print '                 common part, empty'
			else:
				pass
			pass
			"""
			# check whole part of i1-th tetrahedron in obj1 is inside obj2 or not
			counter4=0
			for i4 in range(len(tmp3a)):
				for i3 in range(len(obj2)):
					tmp3d=obj2[i3]
					num=inside_outside_tetrahedron(tmp3a[i4],tmp3d[0],tmp3d[1],tmp3d[2],tmp3d[3])
					if num==0: # inside
						counter4+=1
						break
					else:
						pass
			if counter4==len(tmp3a):
				#print '  oh... common part (obj1 and obj2) is NOT empty, obj1 is INSIDE obj2'
				if counter3==0:
					tmp1b=tmp3a.reshape(len(tmp3a)*18)
				else:
					tmp1b=np.append(tmp1b,tmp3a)
				counter3+=1
			"""
		else: # if common part (obj1 and obj2) is NOT empty
			#print '  common part (obj1 and obj2) is NOT empty'
			tmp4d=tmp1c.reshape(len(tmp1c)/72,4,6,3)
			# volume check
			w1,w2,w3=obj_volume_6d(tmp4d)
			vol5b=(w1+w2*TAU)/float(w3)
			if vol5a>=vol5b and vol5b>=0.0:
				if verbose==1:
					print '                 common part, %d %d %d (%10.6f)'%(w1,w2,w3,vol5b)
				else:
					pass
				if counter3==0:
					tmp1b=tmp4d.reshape(len(tmp4d)*72)
				else:
					tmp1b=np.append(tmp1b,tmp4d)
				counter3+=1
			else:
				if verbose==1:
					print '                 common part, %d %d %d (%10.6f) out of expected!'%(w1,w2,w3,vol5b)
				else:
					pass
				pass
				#
				"""
				if counter3==0:
					tmp1b=tmp4d.reshape(len(tmp4d)*72)
				else:
					tmp1b=np.append(tmp1b,tmp4d)
				counter3+=1
				"""
			#v1,v2,v3=obj_volume_6d(tmp4d)
			#vol4a=(v1+v2*TAU)/float(v3)
			#print '          volume = %d %d %d (%10.6f)'%(v1,v2,v3,vol4a)
			#file_tmp='%s/intersecting_point.xyz'%(path)
			#generator_xyz_dim4(tmp4d,file_tmp)
			"""
			#
			# Re-tetrahedralization 
			tmp3c=remove_doubling_dim4_in_perp_space(tmp4d)
			tmp4c=tetrahedralization_points(tmp3c)
			if tmp4c.tolist()==[[[[0]]]]:
				pass
			else:
				#v1,v2,v3=obj_volume_6d(tmp4c)
				#vol4b=(v1+v2*TAU)/float(v3)
				#print '          volume = %d %d %d (%10.6f)'%(v1,v2,v3,vol4b)
				if counter3==0:
					tmp1b=tmp4c.reshape(len(tmp4c)*72)
				else:
					tmp1b=np.append(tmp1b,tmp4c)
				counter3+=1
			"""
			
			"""
			if counter3==0:
				tmp1b=tmp4d.reshape(len(tmp4d)*72)
			else:
				tmp1b=np.append(tmp1b,tmp4d)
			counter3+=1
			"""
	if counter3!=0:
		tmp4_common=tmp1b.reshape(len(tmp1b)/72,4,6,3)

		if option2==0:
			v01,v02,v03=obj_volume_6d(tmp4_common)
			vol3=(v01+v02*TAU)/float(v03)
			if verbose==1:
				print '   common obj, volume = %d %d %d (%10.6f)'%(v01,v02,v03,vol3)
			else:
				pass
			if option1==2:
				# check each tetrahedron in "tmp4_common" is really inside "obj1" and "obj1"
				print '   checking common obj'
				counter5=0
				num=len(tmp4_common)
				for i1 in range(len(tmp4_common)):
					flag1=tetrahedron_inside_obj(tmp4_common[i1],obj1)
					flag2=tetrahedron_inside_obj(tmp4_common[i1],obj2)
					if flag1==0 and flag2==0:
						if counter5==0:
							tmp1a=tmp4_common[i1].reshape(72)
						else:
							tmp1a=np.append(tmp1a,tmp4_common[i1])
						counter5+=1
					else:
						pass
				tmp4_common=tmp1a.reshape(len(tmp1a)/72,4,6,3)
				print '   numbre of tetrahedron: %d -> %d'%(num,counter5)
				v01,v02,v03=obj_volume_6d(tmp4_common)
				vol3=(v01+v02*TAU)/float(v03)
				print '   new common obj, volume = %d %d %d (%10.6f)'%(v01,v02,v03,vol3)
			else:
				pass
			return tmp4_common
			
		elif option2==1:
			#
			# Remove unnecessary decompositions
			# if sum of tetrahedra is forming a tetrahedron, then merge them.
			# This makes computation time longer
			#
			#print ' remove unnecessary decompositions'
			counter1=0
			for i1 in range(len(obj2)):
				tmp3b=obj2[i1]
				v01,v02,v03=tetrahedron_volume_6d(tmp3b)# volume of tmp3b
				#print ' v01,v02,v03 = ',v01,v02,v03
				v1,v2,v3=0,0,1 # initialize
				counter2=0
				for i2 in range(len(tmp4_common)):
					tmp3a=tmp4_common[i2]
					w1,w2,w3=tetrahedron_volume_6d(tmp3a) # volume of tmp3a
					flag=tetrahedron_inside_tetrahedron(tmp3a,tmp3b)
					if flag==0: # if tmp3a is inside tmp3b
						if counter2==0:
							tmp1a=tmp3a.reshape(72) # 4*6*3
						else:
							tmp1a=np.append(tmp1a,tmp3a)
						v1,v2,v3=add(v1,v2,v3,w1,w2,w3)
						counter2+=1
					else:
						pass
				#print ' v1,v2,v3 = ',v1,v2,v3
				#
				if v01==v1 and v02==v2 and v03==v3: # if sum of tetrahedra is forming a tetrahedron, then merge them.
					if counter1==0:
						tmp1b=tmp3b.reshape(72) # 4*6*3
					else:
						tmp1b=np.append(tmp1b,tmp3b)
					counter1+=1
				else:
					if counter2!=0:
						if counter1==0:
							tmp1b=tmp1a
						else:
							tmp1b=np.append(tmp1b,tmp1a)
						counter1+=1
					else:
						pass
			tmp4_common=tmp1b.reshape(len(tmp1b)/72,4,6,3)
			print '   numbre of tetrahedron: %d -> %d'%(counter3,counter1)
			v01,v02,v03=obj_volume_6d(tmp4_common)
			vol3=(v01+v02*TAU)/float(v03)
			print '   common obj, volume = %d %d %d (%10.6f)'%(v01,v02,v03,vol3)
			#
			if option1==2:
				# check each tetrahedron in "tmp4_common" is really inside "obj1" and "obj1"
				print '   checking common obj'
				counter5=0
				for i1 in range(len(tmp4_common)):
					flag1=tetrahedron_inside_obj(tmp4_common[i1],obj1)
					flag2=tetrahedron_inside_obj(tmp4_common[i1],obj2)
					if flag1==0 and flag2==0:
						if counter5==0:
							tmp1a=tmp4_common[i1].reshape(72)
						else:
							tmp1a=np.append(tmp1a,tmp4_common[i1])
						counter5+=1
					else:
						pass
			else:
				pass
			tmp4_common=tmp1a.reshape(len(tmp1a)/72,4,6,3)
			print '   numbre of tetrahedron: %d -> %d'%(counter1,counter5)
			v01,v02,v03=obj_volume_6d(tmp4_common)
			vol3=(v01+v02*TAU)/float(v03)
			print '   new common obj, volume = %d %d %d (%10.6f)'%(v01,v02,v03,vol3)
			
			return tmp4_common
		else:
			print 'Error: no option flag'
			pass
	else:
		return np.array([0],dtype=np.int).reshape(1,1,1,1)

cpdef intersection_using_tetrahedron_3(np.ndarray[np.int64_t, ndim=4] obj1,np.ndarray[np.int64_t, ndim=4] obj2,int option,int verbose,path):
	#
	# Intersection; obj1 and obj2
	#
	cdef int num_of_cycle, num_of_shuffle, verbose_level
	cdef int i1,i2,i3,i4,counter1,counter2,counter3,counter4,counter5,q1,q2,q3,flag,num,flag1,flag2
	#cdef int option1
	cdef long v1a,v1b,v1c,v2a,v2b,v2c,v3a,v3b,v3c
	cdef double vol1,vol2,vol3
	cdef np.ndarray[np.int64_t,ndim=1] tmp1b,tmp1c
	cdef np.ndarray[np.int64_t,ndim=4] tmp4_common,tmp4a
	
	print ' intersection_using_tetrahedron_3()'
	
	# parameters for simplification()
	num_of_cycle=3
	num_of_shuffle=3
	verbose_level=0
	
	v1a,v1b,v1c=obj_volume_6d(obj1)
	vol1=(v1a+v1b*TAU)/float(v1c)
	if verbose>=1:
		print '   obj1, volume = %d %d %d (%10.6f)'%(v1a,v1b,v1c,vol1)
	else:
		pass	
	v1a,v1b,v1c=obj_volume_6d(obj2)
	vol1=(v1a+v1b*TAU)/float(v1c)
	if verbose>=1:
		print '   obj2, volume = %d %d %d (%10.6f)'%(v1a,v1b,v1c,vol1)
	else:
		pass
	
	counter3=0
	tmp1b=np.array([0])
	tmp1c=np.array([0])
	tmp4_common=np.array([0]).reshape(1,1,1,1)
	for i1 in range(len(obj1)):
		
		# volume check
		v2a,v2b,v2c=tetrahedron_volume_6d(obj1[i1])
		vol2=(v2a+v2b*TAU)/float(v2c)
		if verbose>=1:
			print '  %2d-th tetrahedron in obj1, %d %d %d (%10.6f)'%(i1,v2a,v2b,v2c,vol2)
		else:
			pass

		counter1=0
		for i2 in range(len(obj2)):
			#tmp4_common=intersection_two_tetrahedron_2(obj1[i1],obj2[i2])
			#tmp4_common=intersection_two_tetrahedron_3(obj1[i1],obj2[i2])
			tmp4_common=intersection_two_tetrahedron_4(obj1[i1],obj2[i2],verbose)
			#tmp4_common=intersection_two_tetrahedron_4(obj1[i1],obj2[i2],1,verbose)
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
				#tmp4a=simplification(tmp4a, num_of_cycle, num_of_shuffle, verbose_level)
				tmp4a=simplification(tmp4a,num_of_cycle,0,0,num_of_shuffle,0,0,verbose_level)
			else:
				pass
			#
			# volume check
			v3a,v3b,v3c=obj_volume_6d(tmp4a)
			vol3=(v3a+v3b*TAU)/float(v3c)
			if vol2>=vol3 and vol3>=0.0:
				if verbose>=1:
					print '                 common part, %d %d %d (%10.6f)'%(v3a,v3b,v3c,vol3)
				else:
					pass
				if counter3==0:
					#tmp1b=tmp1c
					tmp1b=tmp4a.reshape(len(tmp4a)*72)
				else:
					#tmp1b=np.append(tmp1b,tmp1c)
					tmp1b=np.append(tmp1b,tmp4a)
				counter3+=1
			else:
				if verbose>=1:
					print '                 common part, %d %d %d (%10.6f) out of expected!'%(v3a,v3b,v3c,vol3)
				else:
					pass
				pass
		else: # if common part (obj1 and obj2) is NOT empty
			if verbose>=1:
				print '                 common part, empty'
			else:
				pass
			pass
			
	if counter3!=0:
		tmp4_common=tmp1b.reshape(len(tmp1b)/72,4,6,3)
		v1a,v1b,v1c=obj_volume_6d(tmp4_common)
		vol1=(v1a+v1b*TAU)/float(v1c)
		if verbose>=1:
			print '   common obj, volume = %d %d %d (%10.6f)'%(v1a,v1b,v1c,vol1)
		else:
			pass
		return tmp4_common
	else:
		return np.array([0],dtype=np.int).reshape(1,1,1,1)

cpdef intersection_using_tetrahedron_4(np.ndarray[np.int64_t, ndim=4] obj1,np.ndarray[np.int64_t, ndim=4] obj2,int option,int verbose,int dummy):
	#
	# Intersection; obj1 and obj2
	#
	#cdef int num_of_cycle,num_of_shuffle#,verbose_level#,scale
	cdef int i1,i2,i3,i4,counter1,counter2,counter3,counter4,counter5,q1,q2,q3,flag,num,flag1,flag2
	#cdef int option1
	cdef long v1a,v1b,v1c,v2a,v2b,v2c,v3a,v3b,v3c
	cdef double vol1,vol2,vol3,vol4,vol5
	cdef np.ndarray[np.int64_t,ndim=1] tmp1b,tmp1c
	cdef np.ndarray[np.int64_t,ndim=4] tmp4_common,tmp4a,tmp4b
	
	if verbose>0:
		print ' intersection_using_tetrahedron_4()'
	else:
		pass
	# parameters for simplification()
	#num_of_cycle=3
	#num_of_shuffle=3
	#verbose_level=0
	
	v1a,v1b,v1c=obj_volume_6d(obj1)
	#vol1=(v1a+v1b*TAU)/float(v1c)
	vol2=obj_volume_6d_numerical(obj1)
	if verbose>1:
		#print '   obj1, volume = %d %d %d (%10.8f) (%10.8f)'%(v1a,v1b,v1c,vol1,vol2)
		print '   obj1, volume = %d %d %d (%10.8f)'%(v1a,v1b,v1c,vol2)
		print '                = ',obj_volume_6d_numerical(obj1)
	else:
		pass	
	v1a,v1b,v1c=obj_volume_6d(obj2)
	#vol1=(v1a+v1b*TAU)/float(v1c)
	vol2=obj_volume_6d_numerical(obj2)
	if verbose>1:
		#print '   obj2, volume = %d %d %d (%10.8f) (%10.8f)'%(v1a,v1b,v1c,vol1,vol2)
		print '   obj2, volume = %d %d %d (%10.8f)'%(v1a,v1b,v1c,vol2)
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
		#vol1=(v1a+v1b*TAU)/float(v1c)
		if verbose>1:
			print '  %2d-th tetrahedron in obj1, %d %d %d (%10.8f)'%(i1,v1a,v1b,v1c,vol1)
		else:
			pass
		counter1=0
		for i2 in range(len(obj2)):
			if verbose>2:
				v2a,v2b,v2c=tetrahedron_volume_6d(obj2[i2])
				vol2=tetrahedron_volume_6d_numerical(obj2[i2])
				#vol2=(v2a+v2b*TAU)/float(v2c)
				print '  %2d-th tetrahedron in obj2, %d %d %d (%10.8f)'%(i2,v2a,v2b,v2c,vol2)
			else:
				pass
			#tmp4_common=intersection_two_tetrahedron_2(obj1[i1],obj2[i2])
			#tmp4_common=intersection_two_tetrahedron_3(obj1[i1],obj2[i2])
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
				#tmp4a=simplification(tmp4a, num_of_cycle, num_of_shuffle, verbose_level)
				#tmp4a=simplification(tmp4a,num_of_cycle,0,0,num_of_shuffle,0,0,verbose_level)
				#tmp4a=simplification_obj_edges(obj, num_cycle, verbose_level) # do not use this
				#tmp4a=simplification_obj_edges(tmp4a,2,1) # do not use this
				tmp4a=simplification_convex_polyhedron(tmp4a,2,verbose-1)
			else:
				pass
			#
			# volume check
			v3a,v3b,v3c=obj_volume_6d(tmp4a)
			vol3=(v3a+v3b*TAU)/float(v3c)
			if v3a==v1a and v3b==v1b and v3c==v1c: # vol1==vol3 ＃ この場合、obj1[i1]全体が、obj2に含まれている
				if verbose>1:
					print '                 common part (all), %d %d %d (%10.8f)'%(v3a,v3b,v3c,vol3)
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
						print '                 common part (all), %d %d %d (%10.8f)'%(v1a,v1b,v1c,vol4)
					else:
						pass
					if counter3==0:
						tmp1b=obj1[i1].reshape(72)
					else:
						tmp1b=np.append(tmp1b,obj1[i1])
					counter3+=1
				#elif abs(vol1-vol4)>1e-8 and vol3>=0.0:
				elif abs(vol1-vol4)>1e-8 and vol4>=0.0:
					if abs(vol4-vol3)<1e-8:
						if verbose>1:
							print '                 common part (partial_1), %d %d %d (%10.8f)'%(v3a,v3b,v3c,vol4)
						else:
							pass
						if counter3==0:
							#tmp1b=tmp1c
							tmp1b=tmp4a.reshape(len(tmp4a)*72)
						else:
							#tmp1b=np.append(tmp1b,tmp1c)
							tmp1b=np.append(tmp1b,tmp4a)
						counter3+=1
					else:
						# simplification to avoid overflow problem which is resulted from small tetrahedra
						#tmp4b=simplification_obj_edges(tmp4a,2,1)
						#tmp4b=simplification(tmp4a,4,0,0,1,0,0,1)
						tmp4b=tmp4a
						#
						vol4=obj_volume_6d_numerical(tmp4b)
						v3a,v3b,v3c=obj_volume_6d(tmp4b)
						vol3=(v3a+v3b*TAU)/float(v3c)
						if abs(vol4-vol3)<1e-8:
							if verbose>1:
								print '                 common part (partial_2), %d %d %d (%10.8f)'%(v3a,v3b,v3c,vol4)
							else:
								pass
							if counter3==0:
								#tmp1b=tmp1c
								tmp1b=tmp4b.reshape(len(tmp4b)*72)
							else:
								#tmp1b=np.append(tmp1b,tmp1c)
								tmp1b=np.append(tmp1b,tmp4b)
							counter3+=1
						else:
							if verbose>1:
								print '                 common part (partial_3), %d %d %d (%10.8f)'%(v3a,v3b,v3c,vol4)
							else:
								pass
							if counter3==0:
								#tmp1b=tmp1c
								tmp1b=tmp4a.reshape(len(tmp4a)*72)
							else:
								#tmp1b=np.append(tmp1b,tmp1c)
								tmp1b=np.append(tmp1b,tmp4a)
							counter3+=1
				else:
					if verbose>1:
						print '                 common part, %d %d %d (%10.8f) out of expected!'%(v3a,v3b,v3c,vol3)
						print '                 numerical value, %10.8f'%(vol4)
						"""
						vol5=0
						for i3 in range(len(tmp4a)):
							print '                   %d-th tetrahedron'%(i3)
							vol4=tetrahedron_volume_6d_numerical(tmp4a[i3])
							v3a,v3b,v3c=tetrahedron_volume_6d(tmp4a[i3])
							vol3=(v3a+v3b*TAU)/float(v3c)
							vol5+=vol4
							print '                   volume, %13.12f (%13.12f) (%13.12f)'%(vol4,vol5,vol3)
							for i4 in range(4):
								print '                   v%d: %d %d %d'%(i4+1,tmp4a[i3][i4][0][0],tmp4a[i3][i4][0][1],tmp4a[i3][i4][0][2])
								print '                        %d %d %d'%(tmp4a[i3][i4][1][0],tmp4a[i3][i4][1][1],tmp4a[i3][i4][1][2])
								print '                        %d %d %d'%(tmp4a[i3][i4][2][0],tmp4a[i3][i4][2][1],tmp4a[i3][i4][2][2])
								print '                        %d %d %d'%(tmp4a[i3][i4][3][0],tmp4a[i3][i4][3][1],tmp4a[i3][i4][3][2])
								print '                        %d %d %d'%(tmp4a[i3][i4][4][0],tmp4a[i3][i4][4][1],tmp4a[i3][i4][4][2])
								print '                        %d %d %d'%(tmp4a[i3][i4][5][0],tmp4a[i3][i4][5][1],tmp4a[i3][i4][5][2])
						"""
						# tmp
						#tmp4a=simplification(tmp4a,4,0,0,1,0,0,1)
						#if counter3==0:
						#	#tmp1b=tmp1c
						#	tmp1b=tmp4a.reshape(len(tmp4a)*72)
						#else:
						#	#tmp1b=np.append(tmp1b,tmp1c)
						#	tmp1b=np.append(tmp1b,tmp4a)
						#counter3+=1
					else:
						pass
					pass
		else: # if common part (obj1_reduced and obj2) is NOT empty
			if verbose>1:
				print '                 common part, empty'
			else:
				pass
			pass
	
			
	if counter3!=0:
		#if tmp4_common.tolist()!=[[[[0]]]]:
		#	tmp1b=np.append(tmp1b,tmp4_common)
		#else:
		#	pass
		tmp4_common=tmp1b.reshape(len(tmp1b)/72,4,6,3)
		v1a,v1b,v1c=obj_volume_6d(tmp4_common)
		#vol1=(v1a+v1b*TAU)/float(v1c)
		vol2=obj_volume_6d_numerical(tmp4_common)
		if verbose>1:
			#print '   common obj, volume = %d %d %d (%10.8f)(%10.8f)'%(v1a,v1b,v1c,vol1,vol2)
			print '   common obj, volume = %d %d %d (%10.8f)'%(v1a,v1b,v1c,vol2)
		else:
			pass
		return tmp4_common
	else:
		if tmp4_common.tolist()!=[[[[0]]]]:
			return tmp4_common
		else:
			return np.array([0],dtype=np.int).reshape(1,1,1,1)

cpdef np.ndarray tetrahedron_subtraction_dev(np.ndarray[np.int64_t, ndim=3] tetrahedron,\
										np.ndarray[np.int64_t, ndim=4] obj2,\
										np.ndarray[np.int64_t, ndim=4] obj3):
	# based on object_subtraction_dev()
	cdef int flag,flag1,counter1,counter2,counter3
	cdef int i1,i2,i3
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b
	cdef np.ndarray[np.int64_t,ndim=2] tmp2a
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a,tmp4b
	
	print '     tetrahedron_subtraction_dev()'
	
	#if verbose==1:
	#	print '  get A not B = A not (A and B)'
	#	print '  obj1: A'
	#	print '  obj2: A and B'
	#	print '  obj3: B'
	#else:
	#	pass
	tmp1a=np.array([0])
	tmp1b=np.array([0])
	tmp4b=generator_surface_1(obj2)
	#tmp4b=generator_surface(obj2)
	#tmp4b=obj2
	tmp3a=remove_doubling_dim3_in_perp_space(tmp4b.reshape(len(tmp4b)*3,6,3)) # vertices of obj2
	
	# get vertices of obj2 which are inside tetrahedron
	for i2 in range(len(tmp3a)):
		flag=inside_outside_tetrahedron(tmp3a[i2],tetrahedron[0],tetrahedron[1],tetrahedron[2],tetrahedron[3])
		if flag==0: # insede
			if len(tmp1a)==1:
				tmp1a=tmp3a[i2].reshape(18)
			else:
				tmp1a=np.append(tmp1a,tmp3a[i2])
		else:
			pass
	# get vertices of tetrahedron  which are NOT inside obj2
	for i2 in range(4):
		counter1=0
		for i3 in range(len(obj2)):
			flag=inside_outside_tetrahedron(tetrahedron[i2],obj2[i2][0],obj2[i2][1],obj2[i2][2],obj2[i2][3])
			if flag==0: # insede
				counter1+=1
				break
			else:
				pass
		if counter1==0:
			if len(tmp1a)==1:
					tmp1a=tetrahedron[i2].reshape(18)
			else:
				tmp1a=np.append(tmp1a,tetrahedron[i2])
		else:
			pass
	if len(tmp1a)!=1:
		tmp3b=remove_doubling_dim3_in_perp_space(tmp1a.reshape(len(tmp1a)/18,6,3))
		tmp1a=np.array([0])
		if len(tmp3b)>=4:
			tmp4a=tetrahedralization_points(tmp3b)
			for i2 in range(len(tmp4a)):
				# geometric center, centroid of the tetrahedron, tmp2c
				tmp2a=centroid(tmp4a[i2])
				counter1=0
				for i3 in range(len(obj2)):
					# check tmp2c is out of obj2 or not
					flag=inside_outside_tetrahedron(tmp2a,obj2[i3][0],obj2[i3][1],obj2[i3][2],obj2[i3][3])
					if flag==0: # inside
						counter1+=1
						break
					else:
						pass
				if counter1==0:
					if len(tmp1a)==1:
						tmp1a=tmp4a[i2].reshape(72)
					else:
						tmp1a=np.append(tmp1a,tmp4a[i2])
				else:
					pass
			if len(tmp1b)==1:
				tmp1b=tmp1a
			else:
				tmp1b=np.append(tmp1b,tmp1a)
		else:
			pass
	else:
		pass
	
	if len(tmp1b)!=1:
		tmp4a=tmp1b.reshape(len(tmp1b)/72,4,6,3)
		# もう一度、obj2との交点を求め、四面体分割。そして、obj2に含まれない部分を取り出す。
		# ただし、複雑に凹凸部分と重なる部分では、obj2に含まれる部分を取り除くことができないので注意
		counter1=0
		for i2 in range(len(tmp4a)):
			tmp3a=intersection_two_tetrahedron_0(tetrahedron,tmp4a[i2])
			if tmp3a.tolist()!=[[[0]]]:
				if counter1==0:
					tmp1c=tmp3a.reshape(len(tmp3a)*18)
				else:
					tmp1c=np.append(tmp1c,tmp3a)
				counter1+=1
			else:
				pass
		tmp1c=np.append(tmp4a,tmp1c)
		tmp3a=remove_doubling_dim3_in_perp_space(tmp1c.reshape(len(tmp1c)/18,6,3))
		tmp4a=tetrahedralization_points(tmp3a)
		tmp1a=np.array([0])
		counter1=0
		for i2 in range(len(tmp4a)):
			counter2=0
			for i3 in range(4) :
				for i4 in range(len(obj2)):
					flag=inside_outside_tetrahedron(tmp4a[i2][i3],obj2[i4][0],obj2[i4][1],obj2[i4][2],obj2[i4][3])
					if flag==0: # inside
						counter2+=1
						break
					else:
						pass
			if counter2==4:
				# geometric center, centroid of the tetrahedron, tmp2c
				tmp2a=centroid(tmp4a[i2])
				for i5 in range(len(obj2)):
					flag1=inside_outside_tetrahedron(tmp2a,obj2[i5][0],obj2[i5][1],obj2[i5][2],obj2[i5][3])
					if flag1==0: # inside
						counter3+=1
						break
					else:
						pass
				if counter3==0:
					if counter1==0:
						tmp1a=tmp4a[i2].reshape(72)
					else:
						tmp1a=np.append(tmp1a,tmp4a[i2])
					counter1+=1
				else:
					pass
			else:
				if counter1==0:
					tmp1a=tmp4a[i2].reshape(72)
				else:
					tmp1a=np.append(tmp1a,tmp4a[i2])
				counter1+=1
		# obj2の凹凸部分との共通部分を除くために、分割した四面体ごとに、再度、obj3との交点を求める。
		# 交点があるならば、その点を含めて四面体分割する。
		# そして、各四面体の重心を用いて、tetrahedron not obj2 かを判断
		if counter1!=0:
			tmp1b=np.array([0])
			tmp4a=tmp1a.reshape(len(tmp1a)/72,4,6,3)
			for i3 in range(len(tmp4a)):
				tmp1c=np.array([0])
				for i4 in range(len(obj3)):
					tmp3a=intersection_two_tetrahedron_0(tmp4a[i3],obj3[i4])
					if tmp3a.tolist()!=[[[0]]]:
						if len(tmp1c)==1:
							tmp1c=tmp3a.reshape(len(tmp3a)*18)
						else:
							tmp1c=np.append(tmp1c,tmp3a)
					else:
						pass
				if len(tmp1c)!=1:
					tmp3a=remove_doubling_dim3_in_perp_space(tmp1c.reshape(len(tmp1c)/18,6,3))
					tmp4b=tetrahedralization(tmp4a[i3],tmp3a)
					for i4 in range(len(tmp4b)):
						tmp2a=centroid(tmp4b[i4])
						counter3=0
						for i5 in range(len(obj2)):
							flag1=inside_outside_tetrahedron(tmp2a,obj2[i5][0],obj2[i5][1],obj2[i5][2],obj2[i5][3])
							if flag1==0: # inside
								counter3+=1
								break
							else:
								pass
						if counter3==0:
							if len(tmp1b)==1:
								tmp1b=tmp4b[i4].reshape(72)
							else:
								tmp1b=np.append(tmp1b,tmp4b[i4])
						else:
							pass
				else:
					pass
			if len(tmp1b)!=0:
				return tmp1b.reshape(len(tmp1b)/72,4,6,3)
			else:
				return np.array([0]).reshape(1,1,1,1)
		else:
			return np.array([0]).reshape(1,1,1,1)
	else:
		return np.array([0]).reshape(1,1,1,1)

cpdef intersection_using_tetrahedron_5(np.ndarray[np.int64_t, ndim=4] obj1,np.ndarray[np.int64_t, ndim=4] obj2,int option,int verbose,path):

	cdef int i1,i2,i3,i4,counter1,counter2,counter3,counter4,counter5,q1,q2,q3,flag,num,flag1,flag2
	#cdef int option1
	cdef long v1a,v1b,v1c,v2a,v2b,v2c,v3a,v3b,v3c,v4a,v4b,v4c,v5a,v5b,v5c
	cdef double vol1,vol2,vol3,vol4,vol5
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b,tmp1c
	cdef np.ndarray[np.int64_t,ndim=4] tmp4_common,tmp4a,tmp4b
	
	print ' intersection_using_tetrahedron_5()'
		
	v1a,v1b,v1c=obj_volume_6d(obj1)
	vol1=(v1a+v1b*TAU)/float(v1c)
	vol2=obj_volume_6d_numerical(obj1)
	if verbose>=1:
		print '   obj1, volume = %d %d %d (%10.8f) (%10.8f)'%(v1a,v1b,v1c,vol1,vol2)
		print '                = ',obj_volume_6d_numerical(obj1)
	else:
		pass	
	v1a,v1b,v1c=obj_volume_6d(obj2)
	vol1=(v1a+v1b*TAU)/float(v1c)
	vol2=obj_volume_6d_numerical(obj2)
	if verbose>=1:
		print '   obj2, volume = %d %d %d (%10.8f) (%10.8f)'%(v1a,v1b,v1c,vol1,vol2)
	else:
		pass

	counter3=0
	tmp1a=np.array([0])
	tmp1b=np.array([0])
	tmp1c=np.array([0])
	tmp4_common=np.array([0]).reshape(1,1,1,1)
	for i1 in range(len(obj1)):
		# volume check
		v1a,v1b,v1c=tetrahedron_volume_6d(obj1[i1])
		vol1=(v1a+v1b*TAU)/float(v1c)
		if verbose>=1:
			print '  %2d-th tetrahedron in obj1, %d %d %d (%10.8f)'%(i1,v1a,v1b,v1c,vol1)
		else:
			pass
		counter1=0
		for i2 in range(len(obj2)):
			if verbose>=2:
				v2a,v2b,v2c=tetrahedron_volume_6d(obj2[i2])
				vol2=(v2a+v2b*TAU)/float(v2c)
				print '  %2d-th tetrahedron in obj2, %d %d %d (%10.8f)'%(i2,v2a,v2b,v2c,vol2)
			else:
				pass
			tmp4_common=intersection_two_tetrahedron_4(obj1[i1],obj2[i2],verbose)
			
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
				#tmp4a=simplification(tmp4a, num_of_cycle, num_of_shuffle, verbose_level)
				#tmp4a=simplification(tmp4a,num_of_cycle,0,0,num_of_shuffle,0,0,verbose_level)
				#tmp4a=simplification_obj_edges(obj, num_cycle, verbose_level)
				tmp4a=simplification_obj_edges(tmp4a,2,1)
			else:
				pass
			#
			# volume check
			v3a,v3b,v3c=obj_volume_6d(tmp4a)
			vol3=(v3a+v3b*TAU)/float(v3c)
			if v3a==v1a and v3b==v1b and v3c==v1c: # vol1==vol3 ＃ この場合、obj1[i1]全体が、obj2に含まれている
				if verbose>=1:
					print '                 common part (all), %d %d %d (%10.8f)'%(v3a,v3b,v3c,vol3)
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
					if verbose>=1:
						print '                 common part (all), %d %d %d (%10.8f)'%(v1a,v1b,v1c,vol4)
					else:
						pass
					if counter3==0:
						tmp1b=obj1[i1].reshape(72)
					else:
						tmp1b=np.append(tmp1b,obj1[i1])
					counter3+=1
				#elif abs(vol1-vol4)>1e-8 and vol3>=0.0:
				elif abs(vol1-vol4)>1e-8 and vol4>=0.0:
					
					if abs(vol4-vol3)<1e-8:
						if verbose>=1:
							print '                 common part (partial_1), %d %d %d (%10.8f)'%(v3a,v3b,v3c,vol4)
						else:
							pass
						if counter3==0:
							tmp1b=tmp4a.reshape(len(tmp4a)*72)
						else:
							tmp1b=np.append(tmp1b,tmp4a)
						counter3+=1
					else:
						#
						vol4=obj_volume_6d_numerical(tmp4a)
						v3a,v3b,v3c=obj_volume_6d(tmp4a)
						vol3=(v3a+v3b*TAU)/float(v3c)
						if abs(vol4-vol3)<1e-8:
							if verbose>=1:
								print '                 common part (partial_2), %d %d %d (%10.8f)'%(v3a,v3b,v3c,vol4)
							else:
								pass
							if counter3==0:
								tmp1b=tmp4a.reshape(len(tmp4a)*72)
							else:
								tmp1b=np.append(tmp1b,tmp4a)
							counter3+=1
						else:
							if verbose>=1:
								print '                 common part (partial_3), %d %d %d (%10.8f)'%(v3a,v3b,v3c,vol4)
							else:
								pass
							if counter3==0:
								tmp1b=tmp4a.reshape(len(tmp4a)*72)
							else:
								tmp1b=np.append(tmp1b,tmp4a)
							counter3+=1
					#
					# obj2に含まれない部分を得る
					#tmp4c=generator_surface_1(tmp4a)
					tmp4c=tetrahedron_subtraction_dev(obj1[i1],tmp4a,obj2)
					#if tmp4c.tolist!=[[[[0]]]]:
					#if len(tmp4c)!=1:
					if len(tmp4c[0])==4:
						vol5=obj_volume_6d_numerical(tmp4c)
						v4a,v4b,v4c=obj_volume_6d(tmp4c)
						v5a,v5b,v5c=add(v3a,v3b,v3c,v4a,v4b,v4c)
						if len(tmp1a)==1:
							tmp1a=tmp4c.reshape(len(tmp4c)*72)
						else:
							tmp1a=np.append(tmp1a,tmp4c)
						#
						if verbose>=1:
							print '                 uncommon part         , %d %d %d (%10.8f)'%(v4a,v4b,v4c,vol4)
							print '                 total                 , %d %d %d (%10.8f)'%(v5a,v5b,v5c,vol4+vol5)
						else:
							pass
					else:
						if verbose>=1:
							print '                 uncommon part         , empty???'
						else:
							pass
					
				else:
					if verbose>=1:
						print '                 common part, %d %d %d (%10.8f) out of expected!'%(v3a,v3b,v3c,vol3)
						print '                 numerical value, %10.8f'%(vol4)
					else:
						pass
					pass
		else: # if common part (obj1_reduced and obj2) is NOT empty
			if len(tmp1a)==1:
				tmp1a=obj1[i1].reshape(len(obj1[i1])*18)
			else:
				tmp1a=np.append(tmp1a,obj1[i1])
			if verbose>=1:
				print '                 common part, empty'
			else:
				pass
			pass
	
	if counter3!=0:
		#if tmp4_common.tolist()!=[[[[0]]]]:
		#	tmp1b=np.append(tmp1b,tmp4_common)
		#else:
		#	pass
		tmp4_common=tmp1b.reshape(len(tmp1b)/72,4,6,3)
		v1a,v1b,v1c=obj_volume_6d(tmp4_common)
		vol1=obj_volume_6d_numerical(tmp4_common)
		v2a,v2b,v2c=obj_volume_6d(tmp1a.reshape(len(tmp1a)/72,4,6,3))
		vol2=obj_volume_6d_numerical(tmp1a.reshape(len(tmp1a)/72,4,6,3))
		v3a,v3b,v3c=add(v1a,v1b,v1c,v2a,v2b,v2c)
		if verbose>=1:
			print '   obj1 AND obj2, volume = %d %d %d (%10.8f)'%(v1a,v1b,v1c,vol1)
			print '   obj1 NOT obj2, volume = %d %d %d (%10.8f)'%(v2a,v2b,v2c,vol2)
			print '             sum, volume = %d %d %d (%10.8f)'%(v3a,v3b,v3c,vol1+vol2)
		else:
			pass
		return tmp4_common,tmp1a.reshape(len(tmp1a)/72,4,6,3)
	else:
		if tmp4_common.tolist()!=[[[[0]]]]:
			return tmp4_common,np.array([0]).reshape(1,1,1,1)
		else:
			return np.array([0]).reshape(1,1,1,1),tmp1a.reshape(len(tmp1a)/72,4,6,3)

cpdef int tetrahedron_inside_tetrahedron(np.ndarray[np.int64_t, ndim=3] tetrahedron1,np.ndarray[np.int64_t, ndim=3] tetrahedron2):
	cdef int counter1
	cdef int i1,num
	counter1=0
	for i1 in range(len(tetrahedron1)):
		num=inside_outside_tetrahedron(tetrahedron1[i1],tetrahedron2[0],tetrahedron2[1],tetrahedron2[2],tetrahedron2[3])
		if num==0: # inside
			counter1+=1
			break
		else:
			pass
	if counter1==4:
		return 0 # inside
	else:
		return 1 # outside

# this function is wrong, do not use.
cpdef int tetrahedron_inside_obj(np.ndarray[np.int64_t, ndim=3] tet, np.ndarray[np.int64_t, ndim=4] obj):
	cdef int i1,i2,num
	cdef int counter1,counter2
	counter2=0
	for i1 in range(len(tet)):
		counter1=0
		for i2 in range(len(obj)):
			num=inside_outside_tetrahedron(tet[i1],obj[i2][0],obj[i2][1],obj[i2][2],obj[i2][3])
			if num==0: # inside
				counter1+=1
				break
			else:
				pass
		if counter1!=0:
			counter2+=1
		else:
			pass
	if counter2==4:
		return 0 # inside
	else:
		return 1 # outside
	
cpdef np.ndarray intersection_tetrahedron_and_surface_object(np.ndarray[np.int64_t, ndim=4] obj2_surface,\
																np.ndarray[np.int64_t, ndim=4] obj2_edge,\
																np.ndarray[np.int64_t, ndim=4] obj1_tetrahedron,\
																np.ndarray[np.int64_t, ndim=4] obj2_tetrahedron,\
																path):
	# get common part of two objects (obj1 and obj2)
	# intersection of each terahedron in obj1 and obj2 is calculated using surface method, and then, sum the common parts.
	cdef int i0,i1,i2,i3,counter,counter1,counter2,counter3,num1,num2,counter1a,counter2a,counter1b,counter2b
	cdef np.ndarray[np.int64_t,ndim=1] p,tmp,tmp1,tmp1a,tmp1b
	cdef np.ndarray[np.int64_t,ndim=2] point_tmp,point1_tmp
	cdef np.ndarray[np.int64_t,ndim=3] tet
	cdef np.ndarray[np.int64_t,ndim=3] tr1,tr2,ed1,ed2,tmp3
	cdef np.ndarray[np.int64_t,ndim=3] point_a,point_a1,point_a2
	cdef np.ndarray[np.int64_t,ndim=3] point_b,point_b1,point_b2
	cdef np.ndarray[np.int64_t,ndim=3] point,point1,point_common
	cdef np.ndarray[np.int64_t,ndim=3] tet1
	cdef np.ndarray[np.int64_t,ndim=4] obj_common,obj_1,obj_2
	cdef np.ndarray[np.int64_t,ndim=4] tet1_surface,tet1_edge
	cdef np.ndarray[np.int64_t,ndim=4] tmp4,tmp4a
	
	counter3=0
	#obj2_surface=generator_surface(obj2_tetrahedron)
	#obj2_edge=generator_edge(obj2_surface)
	for i0 in range(len(obj1_tetrahedron)):
		tet1=obj1_tetrahedron[i0]
		tet1_surface=generator_surface(tet1.reshape(1,4,6,3))
		tet1_edge=generator_edge(tet1_surface)
		#
		# intersection i0-th tetrahedron in obj1_ and obj2 (using surface intersection)
		#
		counter=0
		for i1 in range(len(tet1_surface)):
			tr1=tet1_surface[i1]
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
			for i2 in range(len(tet1_edge)):
				ed1=tet1_edge[i2]
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
			print ' No intersection'
			
			#
			# check i0-th tetrahedron is inside obj2 or not
			#
			#tmp3=remove_doubling_dim4_in_perp_space(tet1_surface) # generating vertces of 1st OD	
			tmp3=remove_doubling_dim4(tet1_surface) # generating vertces of 1st OD	
			
			counter1=0
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
				if counter!=0:
					counter1+=1
				else:
					pass
			if counter1==len(tmp3):
				if counter3==0:
					tmp4=tet1.reshape(1,4,6,3) # 4*6*3
				else:
					tmp1=np.append(tmp4,tet1)
					tmp4=tmp1.reshape(len(tmp1)/72,4,6,3) # 4*6*3
				counter3+=1
			else:
				pass
			# return np.array([0],dtype=np.int).reshape(1,1,1,1)
		else:
			point=p.reshape(len(p)/18,6,3) # 18=6*3
			point1=remove_doubling_dim3_in_perp_space(point)
			#point1=remove_doubling_dim3(point)
			"""
			print ' dividing into three PODs:'
			print '    Common   :     OD1 and     ODB'
			print '  UnCommon 1 :     OD1 and Not OD2'
			print '  UnCommon 2 : Not OD1 and     OD2'
			"""
			#
			#	 OD1 and ODB	 : common part (point_common)
			#	 OD1 and Not OD2 : ODA (point_a)
			# Not OD1 and OD2	 : ODB (point_b)
			#
			# --------------------------
			# (1) Extract vertces of 2nd OD which are insede 1st OD --> point_a1
			#	 Extract vertces of 2nd OD which are outsede 1st OD --> point_b2
			#
			counter1a=0
			counter2b=0
			#tmp3=remove_doubling_dim4_in_perp_space(tet1_surface) # generating vertces of 1st OD	
			tmp3=remove_doubling_dim4(tet1_surface) # generating vertces of 1st OD	
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
					if counter1a==0:
						tmp1a=point_tmp.reshape(18) # 18=6*3
					else:
						tmp1a=np.append(tmp1a,point_tmp)
					counter1a+=1
				else:
					if counter2b==0:
						tmp1b=point_tmp.reshape(18)
					else:
						tmp1b=np.append(tmp1b,point_tmp)
					counter2b+=1
			if counter1a!=0:
				point_a1=tmp1a.reshape(len(tmp1a)/18,6,3)
			else:
				pass
			if counter2b!=0:
				point_b2=tmp1b.reshape(len(tmp1b)/18,6,3)
			else:
				pass
			#
			# (2) Extract vertces of 1st OD which are insede 2nd OD --> point_b1
			#	 Extract vertces of 1st OD which are outsede 2nd OD --> point_a2
			#
			counter1b=0
			counter2a=0
			#tmp3=remove_doubling_dim4_in_perp_space(obj2_surface) # generating vertces of 2nd OD
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
					if counter1b==0:
						tmp1a=point_tmp.reshape(18) # 18=6*3
					else:
						tmp1a=np.append(tmp1a,point_tmp)
					counter1b+=1
				else:
					if counter2a==0:
						tmp1b=point_tmp.reshape(18)
					else:
						tmp1b=np.append(tmp1b,point_tmp)
					counter2a+=1
			if counter1b!=0:
				point_b1=tmp1a.reshape(len(tmp1a)/18,6,3)
			else:
				pass
			if counter2a!=0:
				point_a2=tmp1b.reshape(len(tmp1b)/18,6,3)
			else:
				pass
			#
			# (3) Sum point A, point B and Intersections --->>> common part
			#
			# common part = point1 + point_a1 + point_b1
			if counter1a!=0:
				tmp=np.append(point1,point_a1)
				if counter1b!=0:
					tmp=np.append(tmp,point_b1)
				else:
					pass
				point_common=tmp.reshape(len(tmp)/18,6,3) # 18=6*3
			else:
				if counter1b!=0:
					tmp=np.append(point1,point_b1)
					point_common=tmp.reshape(len(tmp)/18,6,3) # 18=6*3
				else:
					point_common=point1
			point_common=remove_doubling_dim3_in_perp_space(point_common)
			#point_common=remove_doubling_dim3(point_common)
			#
			# Tetrahedralization
			#
			if coplanar_check(point_common)==0:
				tmp4a=tetrahedralization_points(point_common)
				#print 'tmp4a=',tmp4a
				if counter3==0:
					tmp4=tmp4a
				else:
					tmp1=np.append(tmp4,tmp4a)
					tmp4=tmp1.reshape(len(tmp1)/72,4,6,3) # 4*6*3
				counter3+=1
				"""
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
				"""
			else:
				print 'coplanar'
	if counter3==0:
		return np.array([0],dtype=np.int).reshape(1,1,1,1)
	else:
		return tmp4	
	
cpdef np.ndarray intersection_two_tetrahedron(np.ndarray[np.int64_t, ndim=3] tetrahedron_1,\
											np.ndarray[np.int64_t, ndim=3] tetrahedron_2):
	cdef int i,i1,i2,j,count,counter1,counter2,t1,t2,t3,t4,t5,t6
	#cdef q1,q2,q3
	cdef np.ndarray[np.int64_t,ndim=1] p,tmp,tmp1,v1,v2,v3,v4,v5,v6,w1,w2,w3,w4,w5,w6
	cdef np.ndarray[np.int64_t,ndim=2] comb,tmp2a,tmp2b
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b,tri
	cdef np.ndarray[np.int64_t,ndim=4] obj1,obj2,obj_a,obj_b,obj_common,obj_uncommon_1,obj_uncommon_2
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
	#
	#print '     intersection_two_tetrahedron()'
	counter1=0
	for i in range(len(comb)): # len(combination_index) = 24
		# case 1: intersection betweem
		# 6 edges of tetrahedron_1
		# 4 surfaces of tetrahedron_2
		segment_1=tetrahedron_1[comb[i][0]] 
		segment_2=tetrahedron_1[comb[i][1]]
		surface_1=tetrahedron_2[comb[i][2]]
		surface_2=tetrahedron_2[comb[i][3]]
		surface_3=tetrahedron_2[comb[i][4]]
		tmp=intersection_segment_surface(segment_1,segment_2,surface_1,surface_2,surface_3)
		if len(tmp)!=1:
			#print 'p = ',tmp
			if counter1==0 :
				p=tmp # intersection
			else:
				p=np.append(p,tmp) # intersecting points
			counter1+=1
		else:
			pass
		# case 2: intersection betweem
		# 6 edges of tetrahedron_2
		# 4 surfaces of tetrahedron_1
		segment_1=tetrahedron_2[comb[i][0]]
		segment_2=tetrahedron_2[comb[i][1]]
		surface_1=tetrahedron_1[comb[i][2]]
		surface_2=tetrahedron_1[comb[i][3]]
		surface_3=tetrahedron_1[comb[i][4]]
		tmp=intersection_segment_surface(segment_1,segment_2,surface_1,surface_2,surface_3)
		if len(tmp)!=1:
			#print 'p = ',tmp
			if counter1==0:
				p=tmp # intersection
			else:
				p=np.append(p,tmp) # intersecting points
			counter1+=1
		else:
			pass
	#print 'counter1',counter1
	if counter1==0:
		#print '   end intersection_two_tetrahedron()'
		return np.array([0],dtype=np.int).reshape(1,1,1,1)
	else:
		# working in perp space. 
		# remove doubling in 'p' (intersecting points)
		# remove any points in 'p' which are also vertces of tetrahedron_1 and/or tetrahedron_2
		tmp=np.append(tetrahedron_1,tetrahedron_2)
		tmp3b=tmp.reshape(len(tmp)/18,6,3)
		tmp3a=p.reshape(len(p)/18,6,3)
		tmp3a=remove_doubling_dim3_in_perp_space(tmp3a)
		counter2=0
		for i1 in range(len(tmp3a)):
			tmp2a=tmp3a[i1]
			#print tmp2a[0],tmp2a[1],tmp2a[2],tmp2a[3],tmp2a[4],tmp2a[5]
			v1,v2,v3,v4,v5,v6=projection(tmp2a[0],tmp2a[1],tmp2a[2],tmp2a[3],tmp2a[4],tmp2a[5])
			counter1=0
			for i2 in range(len(tmp3b)):
				tmp2b=tmp3b[i2]
				w1,w2,w3,w4,w5,w6=projection(tmp2b[0],tmp2b[1],tmp2b[2],tmp2b[3],tmp2b[4],tmp2b[5])
				if np.all(v4==w4) and np.all(v5==w5) and np.all(v6==w6):
					counter1+=1
					break
				else:
					pass
			if counter1==0:
				if counter2==0:
					tmp1=tmp2a.reshape(18)
					#tmp1=np.array([tmp2a[0],tmp2a[1],tmp2a[2],tmp2a[3],tmp2a[4],tmp2a[5]]).reshape(18)
				else:
					tmp1=np.append(tmp1,tmp2a)
					#tmp1=np.append(tmp1,[tmp2a[0],tmp2a[1],tmp2a[2],tmp2a[3],tmp2a[4],tmp2a[5]])
				counter2+=1
			else:
				pass
		if counter2!=0: # if there is intersecting point.
			tmp3a=tmp1.reshape(len(tmp1)/18,6,3)
			#
			# coplanar_check
			#if coplanar_check(tmp3a)==0:
			#
			#
			#for i in range(len(tmp3a)):
			#	z1,z2,z3,z4,z5,z6=projection(tmp3a[i][0],tmp3a[i][1],tmp3a[i][2],tmp3a[i][3],tmp3a[i][4],tmp3a[i][5])
			#	print 'Xx %8.6f %8.6f %8.6f'%((z4[0]+TAU*z4[1])/float(z4[2]),(z5[0]+TAU*z5[1])/float(z5[2]),(z6[0]+TAU*z6[1])/float(z6[2]))
			#------------------------
			#   Tetrahedralization
			#------------------------
			obj1=tetrahedralization(tetrahedron_1,tmp3a) # Tetrahedralization of tetrahedron_1
			"""
			t1,t2,t3=tetrahedron_volume_6d(tetrahedron_1)
			t4,t5,t6=obj_volume_6d(obj1)
			if t1==t4 and t2==t5 and t3==t6:
				print '      tet1 ok'
				pass
			else:
				print '      tet1 fail'
			"""
			"""
			for i in range(len(obj1)):
				[q1,q2,q3]=tetrahedron_volume_6d(obj1[i])
				print '   %3d-th tetrahedron in obj1, volume: %d,%d,%d = %8.6f'%(i,q1,q2,q3,(q1+q2*TAU)/float(q3))
			"""
			#	print '%3d-th obj1:'%(i)
			#	for j in range(len(obj1[i])):
			#		print ' %3d-th obj in %3d-th obj1'%(j,i)
			#		z1,z2,z3,z4,z5,z6=projection(obj1[i][j][0],obj1[i][j][1],obj1[i][j][2],obj1[i][j][3],obj1[i][j][4],obj1[i][j][5])
			#		print '	Xx %8.6f %8.6f %8.6f'%((z4[0]+TAU*z4[1])/float(z4[2]),(z5[0]+TAU*z5[1])/float(z5[2]),(z6[0]+TAU*z6[1])/float(z6[2]))
			obj2=tetrahedralization(tetrahedron_2,tmp3a) # Tetrahedralization of tetrahedron_2
			"""
			t1,t2,t3=tetrahedron_volume_6d(tetrahedron_2)
			t4,t5,t6=obj_volume_6d(obj2)
			if t1==t4 and t2==t5 and t3==t6:
				print '      tet2 ok'
				pass
			else:
				print '      tet2 fail'
			"""
			"""
			for i in range(len(obj2)):
				[q1,q2,q3]=tetrahedron_volume_6d(obj2[i])
				print '   %3d-th tetrahedron in obj2 volume: %d,%d,%d = %8.6f'%(i,q1,q2,q3,(q1+q2*TAU)/float(q3))
			"""
			# common part: tetrahedron_1 and tetrahedron_2
			#if obj1.tolist()!=[[[[0]]]] and obj2.tolist()!=[[[[0]]]]:
			#	obj_common=find_common_obj(tetrahedron_1,obj1,tetrahedron_2,obj2)
			#	return obj_common
			#else:
			#	return np.array([0],dtype=np.int).reshape(1,1,1,1)
			#print 'obj1 =',obj1
			#print 'obj2 =',obj2
			if obj1.tolist()!=[[[[0]]]] and obj2.tolist()!=[[[[0]]]]:
				# obj_common     : tetrahedron_1 AND tetrahedron_2
				# obj_uncommon_1 : tetrahedron_1 NOT tetrahedron_2
				# obj_uncommon_2 : tetrahedron_2 NOT tetrahedron_1
				#
				obj_common=find_common_obj(tetrahedron_1,obj1,tetrahedron_2,obj2)
				#return obj_common
				#
				#obj_common,obj_uncommon_1,obj_uncommon_2=find_common_obj_2(tetrahedron_1,obj1,tetrahedron_2,obj2)
				#return obj_common,obj_uncommon_1,obj_uncommon_2
				#
				#obj_common=find_common_obj_2(tetrahedron_1,obj1,tetrahedron_2,obj2)
				#
				# Attention (2020.05.26)
				#
				# See a remark in find_common_obj_2()
				#
				#
				return obj_common
			else:
				#print '   end intersection_two_tetrahedron()'
				return np.array([0],dtype=np.int).reshape(1,1,1,1)
		else:
			#print '   end intersection_two_tetrahedron()'
			return np.array([0],dtype=np.int).reshape(1,1,1,1)

cpdef np.ndarray intersection_two_tetrahedron_2(np.ndarray[np.int64_t, ndim=3] tetrahedron_1,\
											np.ndarray[np.int64_t, ndim=3] tetrahedron_2):
	cdef int i,i1,i2,j,count,counter1,counter2,counter3,t1,t2,t3,t4,t5,t6,flag
	#cdef int q1,q2,q3
	cdef np.ndarray[np.int64_t,ndim=1] p,tmp,tmp1,v1,v2,v3,v4,v5,v6,w1,w2,w3,w4,w5,w6
	cdef np.ndarray[np.int64_t,ndim=2] comb,tmp2a,tmp2b
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b,tri
	cdef np.ndarray[np.int64_t,ndim=4] obj1,obj2,obj_a,obj_b,obj_common,obj_uncommon_1,obj_uncommon_2
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
	#
	#print '     intersection_two_tetrahedron()'
	counter1=0
	for i in range(len(comb)): # len(combination_index) = 24
		# case 1: intersection betweem
		# 6 edges of tetrahedron_1
		# 4 surfaces of tetrahedron_2
		segment_1=tetrahedron_1[comb[i][0]] 
		segment_2=tetrahedron_1[comb[i][1]]
		surface_1=tetrahedron_2[comb[i][2]]
		surface_2=tetrahedron_2[comb[i][3]]
		surface_3=tetrahedron_2[comb[i][4]]
		tmp=intersection_segment_surface(segment_1,segment_2,surface_1,surface_2,surface_3)
		if len(tmp)!=1:
			#print 'p = ',tmp
			if counter1==0 :
				p=tmp # intersection
			else:
				p=np.append(p,tmp) # intersecting points
			counter1+=1
		else:
			pass
		# case 2: intersection betweem
		# 6 edges of tetrahedron_2
		# 4 surfaces of tetrahedron_1
		segment_1=tetrahedron_2[comb[i][0]]
		segment_2=tetrahedron_2[comb[i][1]]
		surface_1=tetrahedron_1[comb[i][2]]
		surface_2=tetrahedron_1[comb[i][3]]
		surface_3=tetrahedron_1[comb[i][4]]
		tmp=intersection_segment_surface(segment_1,segment_2,surface_1,surface_2,surface_3)
		if len(tmp)!=1:
			#print 'p = ',tmp
			if counter1==0:
				p=tmp # intersection
			else:
				p=np.append(p,tmp) # intersecting points
			counter1+=1
		else:
			pass
	#print 'counter1',counter1
	if counter1==0:
		#print '   end intersection_two_tetrahedron()'
		return np.array([0],dtype=np.int).reshape(1,1,1,1)
	else:
		# working in perp space. 
		# remove doubling in 'p' (intersecting points)
		# remove any points in 'p' which are also vertces of tetrahedron_1 and/or tetrahedron_2
		tmp=np.append(tetrahedron_1,tetrahedron_2)
		tmp3b=tmp.reshape(len(tmp)/18,6,3)
		tmp3a=p.reshape(len(p)/18,6,3)
		tmp3a=remove_doubling_dim3_in_perp_space(tmp3a)
		counter2=0
		for i1 in range(len(tmp3a)):
			tmp2a=tmp3a[i1]
			#print tmp2a[0],tmp2a[1],tmp2a[2],tmp2a[3],tmp2a[4],tmp2a[5]
			v1,v2,v3,v4,v5,v6=projection(tmp2a[0],tmp2a[1],tmp2a[2],tmp2a[3],tmp2a[4],tmp2a[5])
			counter1=0
			for i2 in range(len(tmp3b)):
				tmp2b=tmp3b[i2]
				w1,w2,w3,w4,w5,w6=projection(tmp2b[0],tmp2b[1],tmp2b[2],tmp2b[3],tmp2b[4],tmp2b[5])
				if np.all(v4==w4) and np.all(v5==w5) and np.all(v6==w6):
					counter1+=1
					break
				else:
					pass
			if counter1==0:
				if counter2==0:
					tmp1=tmp2a.reshape(18)
					#tmp1=np.array([tmp2a[0],tmp2a[1],tmp2a[2],tmp2a[3],tmp2a[4],tmp2a[5]]).reshape(18)
				else:
					tmp1=np.append(tmp1,tmp2a)
					#tmp1=np.append(tmp1,[tmp2a[0],tmp2a[1],tmp2a[2],tmp2a[3],tmp2a[4],tmp2a[5]])
				counter2+=1
			else:
				pass
		if counter2!=0: # if there is intersecting point (not vertices of tetrahedron 1 and 2).
			tmp3a=tmp1.reshape(len(tmp1)/18,6,3)
			
			# get vertces of tetrahedron_1 which are inside tetrahedron_2
			counter3=0
			for i2 in range(len(tetrahedron_1)):
				flag=inside_outside_tetrahedron(tetrahedron_1[i2],tetrahedron_2[0],tetrahedron_2[1],tetrahedron_2[2],tetrahedron_2[3])
				if flag==0:
					if counter3==0:
						tmp1=tetrahedron_1[i2].reshape(18)
					else:
						tmp1=np.append(tmp1,tetrahedron_1[i2])
					counter3+=1
				else:
					pass
			# get vertces of tetrahedron_2 which are inside tetrahedron_1
			for i2 in range(len(tetrahedron_2)):
				flag=inside_outside_tetrahedron(tetrahedron_2[i2],tetrahedron_1[0],tetrahedron_1[1],tetrahedron_1[2],tetrahedron_1[3])
				if flag==0:
					if counter3==0:
						tmp1=tetrahedron_2[i2].reshape(18)
					else:
						tmp1=np.append(tmp1,tetrahedron_2[i2])
					counter3+=1
				else:
					pass
			if counter3!=0:
				tmp1=np.append(tmp3a,tmp1)
				tmp3a=tmp1.reshape(len(tmp1)/18,6,3)
			else:
				pass
			if len(tmp3a)>=4:
				#------------------------
				#   Tetrahedralization
				#------------------------
				# coplanar_check
				if coplanar_check(tmp3a)==0:
					#print 'tmp3a=',tmp3a
					obj_common=tetrahedralization_points(tmp3a)
					return obj_common
				else:
					return np.array([0],dtype=np.int).reshape(1,1,1,1)
				#obj_common=tetrahedralization_points(tmp3a)
				#return obj_common
			else:
				return np.array([0],dtype=np.int).reshape(1,1,1,1)
		else:
			return np.array([0],dtype=np.int).reshape(1,1,1,1)
# dev
cpdef np.ndarray intersection_two_tetrahedron_3(np.ndarray[np.int64_t, ndim=3] tetrahedron_1,\
												np.ndarray[np.int64_t, ndim=3] tetrahedron_2):
	cdef int i1,i2,counter1,counter2
	cdef np.ndarray[np.int64_t,ndim=1] p,tmp1a,tmp1b,tmp1c
	cdef np.ndarray[np.int64_t,ndim=2] comb
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b
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

	#print '     intersection_two_tetrahedron_3()'
	tmp1a=np.array([0])
	tmp1b=np.array([0])
	tmp1c=np.array([0])

	counter1=0
	for i1 in range(len(comb)): # len(combination_index) = 24
		# case 1: intersection betweem
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
				p=tmp1c # intersection
			else:
				p=np.append(p,tmp1c) # intersecting points
			counter1+=1
		else:
			pass
		# case 2: intersection betweem
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
				p=tmp1c # intersection
			else:
				p=np.append(p,tmp1c) # intersecting points
			counter1+=1
		else:
			pass

	if counter1!=0:
		# get vertces of tetrahedron_1 which are inside tetrahedron_2
		counter1=0
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
		
		if counter1!=0:
			p=np.append(p,tmp1a)
		else:
			pass
		
		# remove doubling in 'p'
		tmp3a=p.reshape(len(p)/18,6,3)
		tmp3a=remove_doubling_dim3_in_perp_space(tmp3a)
		
		print 'len(tmp3a)',len(tmp3a)
		
		### Tetrahedralization ##
		if coplanar_check(tmp3a)==0:
			#print 'tmp3a=',tmp3a
			tmp4a=tetrahedralization_points(tmp3a)
			# geometric center, centroid of the tetrahedron, tmp2c
			for i1 in range(len(tmp4a)):
				tmp2c=centroid(tmp4a[i1])
				# check tmp2c is inside both tetrahedron_1 and 2
				flag1=inside_outside_tetrahedron(tmp2c,tetrahedron_1[0],tetrahedron_1[1],tetrahedron_1[2],tetrahedron_1[3])
				flag2=inside_outside_tetrahedron(tmp2c,tetrahedron_2[0],tetrahedron_2[1],tetrahedron_2[2],tetrahedron_2[3])
				if flag1==0 and flag2==0: # inside
					if len(tmp1b)==1:
						tmp1b=tmp4a[i1].reshape(72)
					else:
						tmp1b=np.append(tmp1b,tmp4a[i1])
				else:
					pass
		if len(tmp1b)!=1:
			return tmp1b.reshape(len(tmp1b)/72,4,6,3)
		else:
			return np.array([0],dtype=np.int).reshape(1,1,1,1)
		"""	
		tmp4a=tetrahedralization(tetrahedron_2,tmp3a) # Tetrahedralization of tetrahedron_2, using 'tmp3a'
		for i1 in range(len(tmp4a)):
			# geometric center, centroid of the tetrahedron, tmp2c
			tmp2c=centroid(tmp4a[i1])
			counter2=0
			# check tmp2c is out of obj2 or not
			flag1=inside_outside_tetrahedron(tmp2c,tetrahedron_1[0],tetrahedron_1[1],tetrahedron_1[2],tetrahedron_1[3])
			if flag1==0: # inside
				if len(tmp1b)==1:
					tmp1b=tmp4a[i1].reshape(72)
				else:
					tmp1b=np.append(tmp1b,tmp4a[i1])
			else:
				pass
		if len(tmp1b)!=1:
			return tmp1b.reshape(len(tmp1b)/72,4,6,3)
		else:
			return np.array([0],dtype=np.int).reshape(1,1,1,1)
		"""
	else:
		return np.array([0],dtype=np.int).reshape(1,1,1,1)

# dev
cpdef np.ndarray intersection_two_tetrahedron_4(np.ndarray[np.int64_t, ndim=3] tetrahedron_1,\
												np.ndarray[np.int64_t, ndim=3] tetrahedron_2,\
												#int scale,\
												int verbose):
	cdef int i1,i2,counter1,counter2
	cdef long a1,b1,c1,a2,b2,c2
	cdef float vol1
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b,tmp1c,v1,v2,v3,v4,v5,v6
	cdef np.ndarray[np.int64_t,ndim=2] comb
	cdef np.ndarray[np.int64_t,ndim=2] tmp2a,tmp2b,tmp2c
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a
	
	if verbose>0:
		print '      intersection_two_tetrahedron_4()'
	else:
		pass

	"""
	if scale==1:
		pass
	else:
		if verbose>=2:
			print '         scale = %d'%(scale)
		else:
			pass
		tmp2a=tetrahedron_1.reshape(24,3)
		tmp2b=tetrahedron_2.reshape(24,3)
		for i1 in range(len(tmp2a)):
			a1,b1,c1=mul(tmp2a[i1][0],tmp2a[i1][1],tmp2a[i1][2],scale,0,1)
			a2,b2,c2=mul(tmp2b[i1][0],tmp2b[i1][1],tmp2b[i1][2],scale,0,1)
			if i1==0:
				tmp1a=np.array([a1,b1,c1])
				tmp1b=np.array([a2,b2,c2])
			else:
				tmp1a=np.append(tmp1a,[a1,b1,c1])
				tmp1b=np.append(tmp1b,[a2,b2,c2])
		if verbose>=2:
			tetrahedron_1=tmp1a.reshape(4,6,3)
			print 'tetrahedron_1:'
			print tetrahedron_1
			tetrahedron_2=tmp1b.reshape(4,6,3)
			a1,b1,c1=tetrahedron_volume_6d(tetrahedron_1)
			a2,b2,c2=tetrahedron_volume_6d(tetrahedron_2)
			print '         volume of tetrahedron_1, %d %d %d (%10.8f)'%(a1,b1,c1,(a1+b1*TAU)/float(c1))
			print '         volume of tetrahedron_2, %d %d %d (%10.8f)'%(a2,b2,c2,(a2+b2*TAU)/float(c2))
		else:
			pass
	"""
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
	
	tmp4a=np.array([0]).reshape(1,1,1,1)
	if counter1!=0:
		# remove doubling
		tmp3a=tmp1a.reshape(len(tmp1a)/18,6,3)
		tmp3a=remove_doubling_dim3_in_perp_space(tmp3a)
		
		if len(tmp3a)>=4:
			if verbose>2:
				print '       number of points for tetrahedralization, %d'%(len(tmp3a))
			else:
				pass
			### Tetrahedralization ##
			if coplanar_check(tmp3a)==0:
				#print 'tmp3a=',tmp3a
				#
				if len(tmp3a)==4:
					tmp4a=tmp3a.reshape(1,4,6,3)
				else:
					tmp4a=tetrahedralization_points(tmp3a)
					#
				if tmp4a.tolist()!=[[[[0]]]]:
					if verbose>2:
						print '       -> number of tetrahedron,  %d'%(len(tmp4a))
					else:
						pass
						# geometric center, centroid of the tetrahedron, tmp2c
					for i1 in range(len(tmp4a)):
						tmp2c=centroid(tmp4a[i1])
						# check tmp2c is inside both tetrahedron_1 and 2
						flag1=inside_outside_tetrahedron(tmp2c,tetrahedron_1[0],tetrahedron_1[1],tetrahedron_1[2],tetrahedron_1[3])
						flag2=inside_outside_tetrahedron(tmp2c,tetrahedron_2[0],tetrahedron_2[1],tetrahedron_2[2],tetrahedron_2[3])
						"""
						#
						print 'tetrahedron_1'
						v1,v2,v3,v4,v5,v6=projection(tetrahedron_1[0][0],tetrahedron_1[0][1],tetrahedron_1[0][2],tetrahedron_1[0][3],tetrahedron_1[0][4],tetrahedron_1[0][5])
						print 'Yy %8.6f %8.6f %8.6f'%((v4[0]+v4[1]*TAU)/float(v4[2]),(v5[0]+v5[1]*TAU)/float(v5[2]),(v6[0]+v6[1]*TAU)/float(v6[2]))
						v1,v2,v3,v4,v5,v6=projection(tetrahedron_1[1][0],tetrahedron_1[1][1],tetrahedron_1[1][2],tetrahedron_1[1][3],tetrahedron_1[1][4],tetrahedron_1[1][5])
						print 'Yy %8.6f %8.6f %8.6f'%((v4[0]+v4[1]*TAU)/float(v4[2]),(v5[0]+v5[1]*TAU)/float(v5[2]),(v6[0]+v6[1]*TAU)/float(v6[2]))
						v1,v2,v3,v4,v5,v6=projection(tetrahedron_1[2][0],tetrahedron_1[2][1],tetrahedron_1[2][2],tetrahedron_1[2][3],tetrahedron_1[2][4],tetrahedron_1[2][5])
						print 'Yy %8.6f %8.6f %8.6f'%((v4[0]+v4[1]*TAU)/float(v4[2]),(v5[0]+v5[1]*TAU)/float(v5[2]),(v6[0]+v6[1]*TAU)/float(v6[2]))
						v1,v2,v3,v4,v5,v6=projection(tetrahedron_1[3][0],tetrahedron_1[3][1],tetrahedron_1[3][2],tetrahedron_1[3][3],tetrahedron_1[3][4],tetrahedron_1[3][5])
						print 'Yy %8.6f %8.6f %8.6f'%((v4[0]+v4[1]*TAU)/float(v4[2]),(v5[0]+v5[1]*TAU)/float(v5[2]),(v6[0]+v6[1]*TAU)/float(v6[2]))
						print ''
						print 'tetrahedron_2'
						v1,v2,v3,v4,v5,v6=projection(tetrahedron_2[0][0],tetrahedron_2[0][1],tetrahedron_2[0][2],tetrahedron_2[0][3],tetrahedron_2[0][4],tetrahedron_2[0][5])
						print 'Zz %8.6f %8.6f %8.6f'%((v4[0]+v4[1]*TAU)/float(v4[2]),(v5[0]+v5[1]*TAU)/float(v5[2]),(v6[0]+v6[1]*TAU)/float(v6[2]))
						v1,v2,v3,v4,v5,v6=projection(tetrahedron_2[1][0],tetrahedron_2[1][1],tetrahedron_2[1][2],tetrahedron_2[1][3],tetrahedron_2[1][4],tetrahedron_2[1][5])
						print 'Zz %8.6f %8.6f %8.6f'%((v4[0]+v4[1]*TAU)/float(v4[2]),(v5[0]+v5[1]*TAU)/float(v5[2]),(v6[0]+v6[1]*TAU)/float(v6[2]))
						v1,v2,v3,v4,v5,v6=projection(tetrahedron_2[2][0],tetrahedron_2[2][1],tetrahedron_2[2][2],tetrahedron_2[2][3],tetrahedron_2[2][4],tetrahedron_2[2][5])
						print 'Zz %8.6f %8.6f %8.6f'%((v4[0]+v4[1]*TAU)/float(v4[2]),(v5[0]+v5[1]*TAU)/float(v5[2]),(v6[0]+v6[1]*TAU)/float(v6[2]))
						v1,v2,v3,v4,v5,v6=projection(tetrahedron_2[3][0],tetrahedron_2[3][1],tetrahedron_2[3][2],tetrahedron_2[3][3],tetrahedron_2[3][4],tetrahedron_2[3][5])
						print 'Zz %8.6f %8.6f %8.6f'%((v4[0]+v4[1]*TAU)/float(v4[2]),(v5[0]+v5[1]*TAU)/float(v5[2]),(v6[0]+v6[1]*TAU)/float(v6[2]))
						print ''
						#
						"""
						if verbose>2:
							#print '         coordinate'
							print '         tetraheddron %d'%(i1)
							v1,v2,v3,v4,v5,v6=projection(tmp4a[i1][0][0],tmp4a[i1][0][1],tmp4a[i1][0][2],tmp4a[i1][0][3],tmp4a[i1][0][4],tmp4a[i1][0][5])
							print '          Qq %8.6f %8.6f %8.6f'%((v4[0]+v4[1]*TAU)/float(v4[2]),(v5[0]+v5[1]*TAU)/float(v5[2]),(v6[0]+v6[1]*TAU)/float(v6[2]))
							v1,v2,v3,v4,v5,v6=projection(tmp4a[i1][1][0],tmp4a[i1][1][1],tmp4a[i1][1][2],tmp4a[i1][1][3],tmp4a[i1][1][4],tmp4a[i1][1][5])
							print '          Qq %8.6f %8.6f %8.6f'%((v4[0]+v4[1]*TAU)/float(v4[2]),(v5[0]+v5[1]*TAU)/float(v5[2]),(v6[0]+v6[1]*TAU)/float(v6[2]))
							v1,v2,v3,v4,v5,v6=projection(tmp4a[i1][2][0],tmp4a[i1][2][1],tmp4a[i1][2][2],tmp4a[i1][2][3],tmp4a[i1][2][4],tmp4a[i1][2][5])
							print '          Qq %8.6f %8.6f %8.6f'%((v4[0]+v4[1]*TAU)/float(v4[2]),(v5[0]+v5[1]*TAU)/float(v5[2]),(v6[0]+v6[1]*TAU)/float(v6[2]))
							v1,v2,v3,v4,v5,v6=projection(tmp4a[i1][3][0],tmp4a[i1][3][1],tmp4a[i1][3][2],tmp4a[i1][3][3],tmp4a[i1][3][4],tmp4a[i1][3][5])
							print '          Qq %8.6f %8.6f %8.6f'%((v4[0]+v4[1]*TAU)/float(v4[2]),(v5[0]+v5[1]*TAU)/float(v5[2]),(v6[0]+v6[1]*TAU)/float(v6[2]))
							# volume of tetrahedron
							a1,b1,c1=tetrahedron_volume_6d(tmp4a[i1])
							vol1=(a1+b1*TAU)/float(c1)
							print '         volume, %d %d %d (%8.6f)'%(a1,b1,c1,vol1)
							print '         volume (numerical value)',tetrahedron_volume_6d_numerical(tmp4a[i1])
						else:
							pass
					
						if flag1==0 and flag2==0: # inside
							if verbose>2:
								print '              in'
							else:
								pass
							if len(tmp1b)==1:
								tmp1b=tmp4a[i1].reshape(72)
								#print '   in %d %d %d (%8.6f)'%(a1,b1,c1,vol1)
							else:
								tmp1b=np.append(tmp1b,tmp4a[i1])
								#print '   in %d %d %d (%8.6f)'%(a1,b1,c1,vol1)
						else:
							if verbose>2:
								print '              out (%d,%d)'%(flag1,flag2)
							else:
								pass
							#print '  out %d %d %d (%8.6f)'%(a1,b1,c1,vol1)
							pass
					if len(tmp1b)!=1:
						if len(tmp1b)/72==len(tmp4a): # 全体が入っている場合
							return tmp4a
						else:  # 一部が交差している場合
							#tmp2c=tmp1b.reshape(len(tmp1b)/3,3)
							#for i2 in range(len(tmp2c)):
							#	a1,b1,c1=div(tmp2c[i2][0],tmp2c[i2][1],tmp2c[i2][2],scale,0,1)
							#	if i2==0:
							#		tmp1c=np.array([a1,b1,c1])
							#	else:
							#		tmp1c=np.append(tmp1c,[a1,b1,c1])
							#return tmp1c.reshape(len(tmp1c)/72,4,6,3)
							return tmp1b.reshape(len(tmp1b)/72,4,6,3)
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

cpdef np.ndarray find_common_obj(np.ndarray[np.int64_t, ndim=3] tetrahedron_1,\
								np.ndarray[np.int64_t, ndim=4] obj1,\
								np.ndarray[np.int64_t, ndim=3] tetrahedron_2,\
								np.ndarray[np.int64_t, ndim=4] obj2):
	#
	# This get common part of tetrahedron_1 and tetrahedron_2 into  parts:
	#
	cdef int i1,i2,counter1,counter2,counter3,counter4,flag
	cdef int t1,t2,t3,t4,t5,t6
	cdef np.ndarray[np.int64_t,ndim=1] tmp1,tmp1a,obj_common,obj_a,obj_b
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a
	cdef np.ndarray[np.int64_t,ndim=4] tmp4c
	#print '      find_common_obj()'
	
	# Volume
	#t1,t2,t3=tetrahedron_volume_6d(tetrahedron_1)
	#t4,t5,t6=tetrahedron_volume_6d(tetrahedron_2)
	
	counter1=0 # counter for common part 
	for i1 in range(len(obj1)): # obj1 is part of tetrahedron_1
		counter2=0
		for i2 in range(4):
			flag=inside_outside_tetrahedron(obj1[i1][i2],tetrahedron_2[0],tetrahedron_2[1],tetrahedron_2[2],tetrahedron_2[3])
			if flag==0: # inside tetrahedron_2
				counter2+=1
			else: # outside tetrahedron_2
				break
		if counter2==4: #  obj1[i1] : tetrahedron_1 and tetrahedron_2
			tmp1a=np.append(obj1[i1][0],obj1[i1][1])
			tmp1a=np.append(tmp1a,obj1[i1][2])
			tmp1a=np.append(tmp1a,obj1[i1][3])
			if counter1==0:
				obj_common=tmp1a
			else:
				obj_common=np.append(obj_common,tmp1a)
			counter1+=1
		else:
			pass
	#print '        hello 1'
	for i1 in range(len(obj2)): # obj2 is part of tetrahedron_2
		#print '          len(obj2)=',len(obj2)
		counter2=0
		for i2 in range(4):
			#print '           i2=',i2
			flag=inside_outside_tetrahedron(obj2[i1][i2],tetrahedron_1[0],tetrahedron_1[1],tetrahedron_1[2],tetrahedron_1[3])
			#print '           flag=',flag
			if flag==0: # inside tetrahedron_1
				counter2+=1
			else:
				break
		if counter2==4: #  obj2[i1] : tetrahedron_1 and tetrahedron_2
			tmp1a=np.append(obj2[i1][0],obj2[i1][1])
			tmp1a=np.append(tmp1a,obj2[i1][2])
			tmp1a=np.append(tmp1a,obj2[i1][3])
			if counter1==0:
				obj_common=tmp1a
			else:
				obj_common=np.append(obj_common,tmp1a)
			counter1+=1
		else:
			pass
	#print '        hello 2'
	# COMMON
	if counter1!=0:
		tmp4c=obj_common.reshape(len(obj_common)/72,4,6,3) # 72=4*6*3
	else:
		tmp4c=np.array([0],dtype=np.int).reshape(1,1,1,1)
	#print '        end find_common_obj()'
	return tmp4c

cpdef np.ndarray find_common_obj_2(np.ndarray[np.int64_t, ndim=3] tetrahedron_1,\
									np.ndarray[np.int64_t, ndim=4] obj1,\
									np.ndarray[np.int64_t, ndim=3] tetrahedron_2,\
									np.ndarray[np.int64_t, ndim=4] obj2):
	#
	# updated version of "find_common_obj()"
	#
	# This decomposes tetrahedron_1 and tetrahedron_2 into three parts:
	# (1) common part 
	# (2) uncommon part (tetrahedron_1 NOT tetrahedron_2)
	# (3) uncommon part (tetrahedron_2 NOT tetrahedron_1)
	#
	cdef int i1,i2,counter1a,counter1b,counter2,counter3,counter4,flag
	cdef long t1,t2,t3,t4,t5,t6,v1,v2,v3,v4,v5,v6,v7,v8,v9,t01,t02,t03,t04,t05,t06,w4,w5,w6,w7,w8,w9
	cdef long v1a,v2a,v3a,v1b,v2b,v3b
	cdef np.ndarray[np.int64_t,ndim=1] tmp1,tmp1a,obj_common,obj_common_a,obj_common_b,obj_a,obj_b,obj_1,obj_2
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a,tmp4b,tmp4c,tmp4d,tmp4e

	# Volume check
	t1,t2,t3=tetrahedron_volume_6d(tetrahedron_1)
	t01,t02,t03=obj_volume_6d(obj1)
	if t1==t01 and t2==t02 and t3==t03:
		pass
	else: 
		print '   error1 in find_common_obj_2()'
	t4,t5,t6=tetrahedron_volume_6d(tetrahedron_2)
	t04,t05,t06=obj_volume_6d(obj2)
	if t4==t04 and t5==t05 and t6==t06:
		pass
	else: 
		print '   error2 in find_common_obj_2()'

	counter1a=0 # counter for common part_a 
	counter1b=0 # counter for common part _b
	counter3=0 # counter for uncommon part (tetrahedron_1 NOT tetrahedron_2)
	counter4=0 # counter for uncommon part (tetrahedron_2 NOT tetrahedron_1)
	for i1 in range(len(obj1)): # obj1 is part of tetrahedron_1
		counter2=0
		tmp1a=np.append(obj1[i1][0],obj1[i1][1])
		tmp1a=np.append(tmp1a,obj1[i1][2])
		tmp1a=np.append(tmp1a,obj1[i1][3])
		for i2 in range(4):
			flag=inside_outside_tetrahedron(obj1[i1][i2],tetrahedron_2[0],tetrahedron_2[1],tetrahedron_2[2],tetrahedron_2[3])
			#flag=inside_outside_tetrahedron_new(obj1[i1][i2],tetrahedron_2[0],tetrahedron_2[1],tetrahedron_2[2],tetrahedron_2[3])
			if flag==0: # inside tetrahedron_2
				counter2+=1
			else: # outside tetrahedron_2
				break
		if counter2==4: #  obj1[i1] : tetrahedron_1 and tetrahedron_2
			#tmp1a=np.append(obj1[i1][0],obj1[i1][1])
			#tmp1a=np.append(tmp1a,obj1[i1][2])
			#tmp1a=np.append(tmp1a,obj1[i1][3])
			if counter1a==0:
				obj_common_a=tmp1a
			else:
				obj_common_a=np.append(obj_common_a,tmp1a)
			counter1a+=1
		else:
			if counter3==0:
				obj_1=tmp1a # obj_1: tetrahedron_1 NOT tetrahedron_2
			else:
				obj_1=np.append(obj_1,tmp1a)
			counter3+=1
	for i1 in range(len(obj2)): # obj2 is part of tetrahedron_2
		counter2=0
		tmp1a=np.append(obj2[i1][0],obj2[i1][1])
		tmp1a=np.append(tmp1a,obj2[i1][2])
		tmp1a=np.append(tmp1a,obj2[i1][3])
		for i2 in range(4):
			flag=inside_outside_tetrahedron(obj2[i1][i2],tetrahedron_1[0],tetrahedron_1[1],tetrahedron_1[2],tetrahedron_1[3])
			#flag=inside_outside_tetrahedron_new(obj2[i1][i2],tetrahedron_1[0],tetrahedron_1[1],tetrahedron_1[2],tetrahedron_1[3])
			if flag==0: # inside tetrahedron_1
				counter2+=1
			else:
				break
		if counter2==4: #  obj2[i1] : tetrahedron_1 and tetrahedron_2
			#tmp1a=np.append(obj2[i1][0],obj2[i1][1])
			#tmp1a=np.append(tmp1a,obj2[i1][2])
			#tmp1a=np.append(tmp1a,obj2[i1][3])
			if counter1b==0:
				obj_common_b=tmp1a
			else:
				obj_common_b=np.append(obj_common_b,tmp1a)
			counter1b+=1
		else:
			if counter4==0:
				obj_2=tmp1a # obj_1: tetrahedron_1 NOT tetrahedron_2
			else:
				obj_2=np.append(obj_2,tmp1a)
			counter4+=1
	
	# COMMON: tetrahedron_1 AND tetrahedron_2
	if counter1a!=0:
		tmp4a=obj_common_a.reshape(len(obj_common_a)/72,4,6,3) # 72=4*6*3
		v1a,v2a,v3a=obj_volume_6d(tmp4a)# obj_common_a
	else:
		tmp4a=np.array([0],dtype=np.int).reshape(1,1,1,1)
		v1a,v2a,v3a=0,0,1
	if counter1b!=0:
		tmp4b=obj_common_b.reshape(len(obj_common_b)/72,4,6,3) # 72=4*6*3
		v1b,v2b,v3b=obj_volume_6d(tmp4b)# obj_common_b
	else:
		tmp4b=np.array([0],dtype=np.int).reshape(1,1,1,1)
		v1b,v2b,v3b=0,0,1
	#
	#                          Attension
	#
	# Remark
	# "obj_common_a" and "obj_common_b" should be identical, however, it is not ture in the present version. 
	# This problem may be resulted form "inside_outside_tetrahedron_new()". 
	# note that both "inside_outside_tetrahedron_new()" and "inside_outside_tetrahedron()"
	# behave same.
	#
	# So, here, I allow to set a summation of "obj_common_a" and "obj_common_b" as "obj_common".
	# This means that the behaviors of "find_common_obj()" and "find_common_obj_2()" is the same.
	# However, "find_common_obj_2()" is slower.
	#
	if counter1a+counter1b!=0:
		obj_common=np.append(tmp4a,tmp4b)
		tmp4c=obj_common.reshape(len(obj_common)/72,4,6,3) # 72=4*6*3
		v1,v2,v3=obj_volume_6d(tmp4c)
	else:
		tmp4c=np.array([0],dtype=np.int).reshape(1,1,1,1)
		v1,v2,v3=0,0,1
	
	# obj_1: tetrahedron_1 NOT tetrahedron_2
	if counter3!=0:
		tmp4d=obj_1.reshape(len(obj_1)/72,4,6,3) # 72=4*6*3
		v4,v5,v6=obj_volume_6d(tmp4d)
	else:
		tmp4d=np.array([0],dtype=np.int).reshape(1,1,1,1)
		v4,v5,v6=0,0,1
	
	# obj_2: tetrahedron_2 NOT tetrahedron_1
	if counter4!=0:
		tmp4e=obj_2.reshape(len(obj_2)/72,4,6,3) # 72=4*6*3
		v7,v8,v9=obj_volume_6d(tmp4e)
	else:
		tmp4e=np.array([0],dtype=np.int).reshape(1,1,1,1)
		v7,v8,v9=0,0,1
			
	w4,w5,w6=add(v1a,v2a,v3a,v4,v5,v6) # COMMON A + obj_1 = tetrahedron_1
	w7,w8,w9=add(v1b,v2b,v3b,v7,v8,v9) # COMMON B + obj_2 = tetrahedron_2
	
	if w4==t1 and w5==t2 and w6==t3 and w7==t4 and w8==t5 and w9==t6:
		if v1a==v1b and v2a==v2b and v3a==v3b:
			pass
		else:
			#print '                    COMMON,%d %d %d (%8.6f)'%(v1,v2,v3,(v1+v2*TAU)/float(v3))
			print '                  COMMON A,%d %d %d (%8.6f)'%(v1a,v2a,v3a,(v1a+v2a*TAU)/float(v3a))
			print '                  COMMON B,%d %d %d (%8.6f)'%(v1b,v2b,v3b,(v1b+v2b*TAU)/float(v3b))
			print '                     obj_1,%d %d %d (%8.6f)'%(v4,v5,v6,(v4+v5*TAU)/float(v6))
			print '                     obj_2,%d %d %d (%8.6f)'%(v7,v8,v9,(v7+v8*TAU)/float(v9))
			print '             tetrahedron_1, %d %d %d (%8.6f)'%(t1,t2,t3,(t1+t2*TAU)/float(t3))
			print '          COMMON A + obj_1, %d %d %d (%8.6f)'%(w4,w5,w6,(w4+w5*TAU)/float(w6))
			print '             tetrahedron_2, %d %d %d (%8.6f)'%(t4,t5,t6,(t4+t5*TAU)/float(t6))
			print '          COMMON B + obj_2, %d %d %d (%8.6f)'%(w7,w8,w9,(w7+w8*TAU)/float(w9))
		return tmp4c
	else:
		return np.array([0],dtype=np.int).reshape(1,1,1,1)
		#return np.array([0],dtype=np.int).reshape(1,1,1,1),np.array([0],dtype=np.int).reshape(1,1,1,1),np.array([0],dtype=np.int).reshape(1,1,1,1)
		
cpdef np.ndarray tetrahedralization_dim4(np.ndarray[np.int64_t, ndim=4] obj ,np.ndarray[np.int64_t, ndim=3] intersecting_point):
	cdef int i1,i2,flag,counter1,counter2
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b
	cdef np.ndarray[np.int64_t,ndim=2] tmp2a
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a,tmp4b
	counter2=0
	for i1 in range(len(obj)):
		tmp3a=obj[i1]
		counter1=0
		for i2 in range(len(intersecting_point)):
			tmp2a=intersecting_point[i2]
			counter3=0
			flag=inside_outside_tetrahedron(tmp2a,tmp3a[0],tmp3a[1],tmp3a[2],tmp3a[3])
			if flag==0: # inside
				if counter3==0:
					tmp1a=tmp2a.reshape(18) # 18=6*3
				else:
					tmp1a=np.append(tmp1a,tmp2a)
				counter3+=1
				counter1+=1
			else: # outside
				pass
		if counter1!=0:
			tmp3b=tmp1a.reshape(len(tmp1a)/18,6,3)
			tmp4a=tetrahedralization(tmp3a,tmp3b)
		else:
			pass
		if counter2==0:
			tmp1b=tmp4a.reshape(len(tmp4a)*72)
		else:
			tmp1b=np.append(tmp1b,tmp4a)
		counter2+=1
	if counter2!=0:
		return tmp1b.reshape(len(tmp1b)/72,4,6,3)
	else:
		return obj

cdef list tetrahedron_volume_6d(np.ndarray[np.int64_t, ndim=3] tetrahedron):
	cdef long v1,v2,v3
	cdef np.ndarray[np.int64_t,ndim=1] x1e,y1e,z1e,x1i,y1i,z1i,x2i,y2i,z2i,x3i,y3i,z3i,x4i,y4i,z4i
	x1e,y1e,z1e,x1i,y1i,z1i=projection(tetrahedron[0][0],tetrahedron[0][1],tetrahedron[0][2],tetrahedron[0][3],tetrahedron[0][4],tetrahedron[0][5])
	x1e,y1e,z1e,x2i,y2i,z2i=projection(tetrahedron[1][0],tetrahedron[1][1],tetrahedron[1][2],tetrahedron[1][3],tetrahedron[1][4],tetrahedron[1][5])
	x1e,y1e,z1e,x3i,y3i,z3i=projection(tetrahedron[2][0],tetrahedron[2][1],tetrahedron[2][2],tetrahedron[2][3],tetrahedron[2][4],tetrahedron[2][5])
	x1e,y1e,z1e,x4i,y4i,z4i=projection(tetrahedron[3][0],tetrahedron[3][1],tetrahedron[3][2],tetrahedron[3][3],tetrahedron[3][4],tetrahedron[3][5])
	[v1,v2,v3]=tetrahedron_volume(np.array([x1i,y1i,z1i]),np.array([x2i,y2i,z2i]),np.array([x3i,y3i,z3i]),np.array([x4i,y4i,z4i]))
	return [v1,v2,v3]

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

cpdef np.ndarray merge_two_triangle(np.ndarray[np.int64_t, ndim=3] triangle_1, np.ndarray[np.int64_t, ndim=3] triangle_2):
	print '         merge_two_triangle()'
	return 0

"""	
cpdef np.ndarray merge_two_tetrahedra(np.ndarray[np.int64_t, ndim=3] tetrahedron_1, \
										np.ndarray[np.int64_t, ndim=3] tetrahedron_2, \
										int verbose):
	# merge two tetrahedra
	cdef int flag,i1,i2,counter1
	cdef long a1,b1,c1,a2,b2,c2,w1,w2,w3,v1,v2,v3
	cdef list comb
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b
	cdef np.ndarray[np.int64_t,ndim=2] tmp2a,tmp2b
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a
	
	if verbose>=4:
		print '   merge_two_tetrahedra()'
	else:
		pass
	# volume
	a1,b1,c1=tetrahedron_volume_6d(tetrahedron_1)
	a2,b2,c2=tetrahedron_volume_6d(tetrahedron_2)
	v1,v2,v3=add(a1,b1,c1,a2,b2,c2)
	
	# three triangles of tetrahedron: 1-2-3, 1-2-4, 1-3-4, 2-3-4
	comb=[[0,1,2],[0,1,3],[0,2,3],[1,2,3]] 
	# other vertx: 3,2,1,0
	comb1=[3,2,1,0] 
	
	# check whether tetrahedron_1 and _2 are sharing a triangle surface or not.
	# Four triangles of tetrahedron_1
	for i1 in range(len(comb)): # i1-th triangle of tetrahedron1
		tmp1a=np.append(tetrahedron_1[comb[i1][0]],tetrahedron_1[comb[i1][1]])
		tmp1a=np.append(tmp1a,tetrahedron_1[comb[i1][2]])
		counter1=0
		for i2 in range(len(comb)): # i2-th triangle of tetrahedron2
			tmp1b=np.append(tetrahedron_2[comb[i2][0]],tetrahedron_2[comb[i2][1]])
			tmp1b=np.append(tmp1b,tetrahedron_2[comb[i2][2]])
			# equivalent_triangle()
			# flag=0 (equivalent), 1 (non equivalent)
			flag=equivalent_triangle(tmp1a,tmp1b)
			if flag==0:
				counter1+=1
				tmp3a=tmp1a.reshape(3,6,3) # common triangle
				tmp2a=tetrahedron_1[comb1[i1]] # a vertx of tetrahedron_1 which is not vertices of the common triangle.
				tmp2b=tetrahedron_2[comb1[i2]] # a vertx of tetrahedron_2 which is not vertices of the common triangle.
				break
			else:
				pass
		if counter1==1:
			break
		else:
			pass
	if counter1==1:
		tmp1a=np.append(tmp2a,tmp2b)
		counter1=0
		for i1 in range(3):
			flag=point_on_segment(tmp3a[i1],tmp2a,tmp2b)
			# a vertex of common triangle is on the line segment between tmp2a and tmp2b
			# this means that sum of tetrahedron_1 and _2 forms a tetrahedron.
			if flag==0:
				counter1+=1
			else:
				tmp1a=np.append(tmp1a,tmp3a[i1])
		if counter1==1:
			if  verbose>=4:
				w1,w2,w3=tetrahedron_volume_6d(tmp1a.reshape(4,6,3))
				print '      volume merged tet = %d %d %d (%8.6f)'%(w1,w2,w3,(w1+TAU*w2)/float(w3))
				print '      volume sum 2  tet = %d %d %d (%8.6f)'%(v1,v2,v3,(v1+TAU*v2)/float(v3))
				# sum of two tetrahedra is forming a tetrahedron
				print '     merged'
			else:
				pass
			return tmp1a.reshape(4,6,3)
		elif  counter1==0:
			# sum of two tetrahedra is not tetrahedron
			return np.array([0]).reshape(1,1,1)
		else:
			# something strange
			return np.array([0]).reshape(1,1,1)
	else:
		# no common triangle
		return np.array([0]).reshape(1,1,1)
"""
# new
cpdef np.ndarray merge_two_tetrahedra(np.ndarray[np.int64_t, ndim=3] tetrahedron_1, \
										np.ndarray[np.int64_t, ndim=3] tetrahedron_2, \
										int verbose):
	# merge two tetrahedra
	cdef int flag,i1,i2,counter1
	cdef long a1,b1,c1,a2,b2,c2,w1,w2,w3,v1,v2,v3
	cdef list comb
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b
	cdef np.ndarray[np.int64_t,ndim=2] tmp2a,tmp2b
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a
	
	if verbose>=4:
		print '         merge_two_tetrahedra()'
	else:
		pass
	# volume
	a1,b1,c1=tetrahedron_volume_6d(tetrahedron_1)
	a2,b2,c2=tetrahedron_volume_6d(tetrahedron_2)
	v1,v2,v3=add(a1,b1,c1,a2,b2,c2)
	
	tmp1a=np.array([0])
	tmp3a=np.array([0]).reshape(1,1,1)
	tmp3a=check_two_tetrahedra(tetrahedron_1,tetrahedron_2)
	if tmp3a.tolist()!=[[[0]]]:
		tmp1a=np.append(tmp3a[3],tmp3a[4])
		counter1=0
		for i1 in range(3):
			flag=point_on_segment(tmp3a[i1],tmp3a[3],tmp3a[4])
			# a vertex of common triangle is on the line segment between tmp2a and tmp2b
			# this means that sum of tetrahedron_1 and _2 forms a tetrahedron.
			if flag==0:
				counter1+=1
			else:
				tmp1a=np.append(tmp1a,tmp3a[i1])
		if counter1==1:
			if  verbose>=4:
				w1,w2,w3=tetrahedron_volume_6d(tmp1a.reshape(4,6,3))
				print '         volume merged tet = %d %d %d (%8.6f)'%(w1,w2,w3,(w1+TAU*w2)/float(w3))
				print '         volume sum 2  tet = %d %d %d (%8.6f)'%(v1,v2,v3,(v1+TAU*v2)/float(v3))
				# sum of two tetrahedra is forming a tetrahedron
				print '         merged'
			else:
				pass
			return tmp1a.reshape(4,6,3)
		elif  counter1==0:
			# sum of two tetrahedra is not tetrahedron
			return np.array([0]).reshape(1,1,1)
		else:
			# something strange
			return np.array([0]).reshape(1,1,1)
	else:
		# no common triangle
		return np.array([0]).reshape(1,1,1)

cpdef np.ndarray check_two_tetrahedra(np.ndarray[np.int64_t, ndim=3] tetrahedron_1, \
										np.ndarray[np.int64_t, ndim=3] tetrahedron_2):
	# check whether tetrahedron_1 and _2 are sharing a triangle surface or not.
	cdef int flag,i1,i2,counter1
	cdef list comb
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b
	cdef np.ndarray[np.int64_t,ndim=2] tmp2a,tmp2b
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a
	
	# three triangles of tetrahedron: 1-2-3, 1-2-4, 1-3-4, 2-3-4
	comb=[[0,1,2],[0,1,3],[0,2,3],[1,2,3]] 
	# other vertx: 3,2,1,0
	comb1=[3,2,1,0] 
	
	# Four triangles of tetrahedron_1
	for i1 in range(len(comb)): # i1-th triangle of tetrahedron1
		tmp1a=np.append(tetrahedron_1[comb[i1][0]],tetrahedron_1[comb[i1][1]])
		tmp1a=np.append(tmp1a,tetrahedron_1[comb[i1][2]])
		counter1=0
		for i2 in range(len(comb)): # i2-th triangle of tetrahedron2
			tmp1b=np.append(tetrahedron_2[comb[i2][0]],tetrahedron_2[comb[i2][1]])
			tmp1b=np.append(tmp1b,tetrahedron_2[comb[i2][2]])
			# equivalent_triangle()
			# flag=0 (equivalent), 1 (non equivalent)
			flag=equivalent_triangle(tmp1a,tmp1b)
			if flag==0:
				counter1+=1
				tmp3a=tmp1a.reshape(3,6,3) # common triangle
				tmp2a=tetrahedron_1[comb1[i1]] # a vertx of tetrahedron_1 which is not vertices of the common triangle.
				tmp2b=tetrahedron_2[comb1[i2]] # a vertx of tetrahedron_2 which is not vertices of the common triangle.
				break
			else:
				pass
		if counter1==1:
			break
		else:
			pass
	if counter1==1:
		tmp1a=np.append(tmp2a,tmp2b)
		tmp1a=np.append(tmp3a,tmp1a)
		return tmp1a.reshape(len(tmp1a)/18,6,3)
	else:
		return np.array([0]).reshape(1,1,1)
			
cpdef np.ndarray merge_three_tetrahedra(np.ndarray[np.int64_t, ndim=3] tetrahedron_1, \
										np.ndarray[np.int64_t, ndim=3] tetrahedron_2, \
										np.ndarray[np.int64_t, ndim=3] tetrahedron_3, \
										int verbose):
	# merge three tetrahedra
	cdef int flag,i1,i2,counter1,counter2,counter3
	cdef long a1,b1,c1,a2,b2,c2,a3,b3,c3,w1,w2,w3,v1,v2,v3
	cdef list comb
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b,tmp1c
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b,tmp3c,tmp3d,tmp3e,tmp3f
	
	if verbose>=2:
		print '         merge_three_tetrahedra()'
	else:
		pass
	# volume
	a1,b1,c1=tetrahedron_volume_6d(tetrahedron_1)
	a2,b2,c2=tetrahedron_volume_6d(tetrahedron_2)
	a3,b3,c3=tetrahedron_volume_6d(tetrahedron_3)
	v1,v2,v3=add(a1,b1,c1,a2,b2,c2)
	v1,v2,v3=add(v1,v2,v3,a3,b3,c3)
		
	tmp1b=np.append(tetrahedron_1,tetrahedron_2)
	tmp1b=np.append(tmp1b,tetrahedron_3)
	tmp3e=remove_doubling_dim3_in_perp_space(tmp1b.reshape(len(tmp1b)/18,6,3))
	
	flag=0
	tmp3d=np.array([[[0]]])
	tmp3a=check_two_tetrahedra(tetrahedron_1,tetrahedron_2)
	if tmp3a.tolist()!=[[[0]]]:
		tmp3b=check_two_tetrahedra(tetrahedron_1,tetrahedron_3)
		if tmp3b.tolist()!=[[[0]]]:
			tmp3c=check_two_tetrahedra(tetrahedron_2,tetrahedron_3)
			if tmp3c.tolist()!=[[[0]]]:
				flag=1
			else:
				pass
		else:
			pass
	else:
		pass
	
	if verbose>=4:
		print '         flag=1'
	else:
		pass

	tmp1b=np.array([0])
	tmp1c=np.array([0])
	if flag==1:
		tmp1a=np.append(tmp3a[3],tmp3a[4])
		tmp1a=np.append(tmp1a,tmp3b[3])
		tmp1a=np.append(tmp1a,tmp3b[4])
		tmp1a=np.append(tmp1a,tmp3c[3])
		tmp1a=np.append(tmp1a,tmp3c[4])
		tmp3f=tmp1a.reshape(len(tmp1a)/18,6,3)
		tmp3d=remove_doubling_dim3_in_perp_space(tmp3f)
		if verbose>=4:
			print '         len(tmp3d)=',len(tmp3d)
		else:
			pass
			
		for i1 in range(len(tmp3e)):
			counter1=0
			for i2 in range(len(tmp3d)):
				if np.all(tmp3e[i1]==tmp3d[i2]):
					counter1+=1
					break
				else:
					pass
			if counter1==0:
				if len(tmp1b)==1:
					tmp1b=tmp3e[i1].reshape(18)
				else:
					tmp1b=np.append(tmp1b,tmp3e[i1])
			else:
				pass
		
		#print 'len(tmp1b)/18=',len(tmp1b)/18
		
		tmp3f=np.array([0]).reshape(1,1,1)
		tmp3e=tmp1b.reshape(len(tmp1b)/18,6,3)
		counter2=0
		counter3=0
		for i1 in range(len(tmp3e)): # len(tmp3e)=2
			tmp1b=np.append(tmp3d,tmp3e[i1])
			if coplanar_check(tmp1b.reshape(len(tmp1b)/18,6,3))==0: # not coplanar
				tmp3f=tmp1b.reshape(len(tmp1b)/18,6,3)
				counter2+=1
			else:
				counter3+=1
		if counter2==1 and counter3==1: 
			if verbose>=2:
				w1,w2,w3=tetrahedron_volume_6d(tmp3f)
				print '         volume merged tet = %d %d %d (%8.6f)'%(w1,w2,w3,(w1+TAU*w2)/float(w3))
				print '         volume sum 3  tet = %d %d %d (%8.6f)'%(v1,v2,v3,(v1+TAU*v2)/float(v3))
				# sum of two tetrahedra is forming a tetrahedron
				print '         merged'
			else:
				pass
			return tmp3f.reshape(4,6,3)
		else:
			return np.array([0]).reshape(1,1,1)
	else:
		return np.array([0]).reshape(1,1,1)

cpdef np.ndarray merge_five_tetrahedra(np.ndarray[np.int64_t, ndim=3] tetrahedron_1,\
										np.ndarray[np.int64_t, ndim=3] tetrahedron_2,\
										np.ndarray[np.int64_t, ndim=3] tetrahedron_3,\
										np.ndarray[np.int64_t, ndim=3] tetrahedron_4,\
										np.ndarray[np.int64_t, ndim=3] tetrahedron_5,\
										int verbose):
		return 0
	
cpdef np.ndarray merge_four_tetrahedra(np.ndarray[np.int64_t, ndim=3] tetrahedron_1,\
										np.ndarray[np.int64_t, ndim=3] tetrahedron_2,\
										np.ndarray[np.int64_t, ndim=3] tetrahedron_3,\
										np.ndarray[np.int64_t, ndim=3] tetrahedron_4,\
										int verbose):
	# merge four tetrahedra
	cdef int flag,i1,i2,counter1,counter2,counter3
	cdef long a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,w1,w2,w3,v1,v2,v3
	cdef list comb
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a,tmp1b,tmp1c
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a,tmp3b,tmp3c,tmp3d,tmp3e,tmp3f,tmp3g,tmp3h
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a,tmp4b
	
	if verbose>=2:
		print '         merge_four_tetrahedra()'
	else:
		pass
	"""
	# volume
	a1,b1,c1=tetrahedron_volume_6d(tetrahedron_1)
	a2,b2,c2=tetrahedron_volume_6d(tetrahedron_2)
	a3,b3,c3=tetrahedron_volume_6d(tetrahedron_3)
	a4,b4,c4=tetrahedron_volume_6d(tetrahedron_4)
	v1,v2,v3=add(a1,b1,c1,a2,b2,c2)
	v1,v2,v3=add(v1,v2,v3,a3,b3,c3)
	v1,v2,v3=add(v1,v2,v3,a4,b4,c4)
	
	tmp1b=np.append(tetrahedron_1,tetrahedron_2)
	tmp1b=np.append(tmp1b,tetrahedron_3)
	tmp1b=np.append(tmp1b,tetrahedron_4)
	# all vertices of four tetrahedra
	tmp3e=remove_doubling_dim3_in_perp_space(tmp1b.reshape(len(tmp1b)/18,6,3))
	"""
	flag=0
	tmp3d=np.array([[[0]]])
	tmp3a=check_two_tetrahedra(tetrahedron_1,tetrahedron_2)
	if tmp3a.tolist()!=[[[0]]]:
		tmp3b=check_two_tetrahedra(tetrahedron_1,tetrahedron_3)
		if tmp3b.tolist()!=[[[0]]]:
			tmp3c=check_two_tetrahedra(tetrahedron_1,tetrahedron_4)
			if tmp3c.tolist()!=[[[0]]]:
				tmp3d=check_two_tetrahedra(tetrahedron_2,tetrahedron_3)
				if tmp3d.tolist()!=[[[0]]]:
					tmp3e=check_two_tetrahedra(tetrahedron_2,tetrahedron_4)
					if tmp3e.tolist()!=[[[0]]]:
						tmp3f=check_two_tetrahedra(tetrahedron_3,tetrahedron_4)
						if tmp3f.tolist()!=[[[0]]]:
							#
							tmp1a=np.append(tmp3a[0],tmp3a[1])
							tmp1a=np.append(tmp1a,tmp3a[2])
							tmp1a=np.append(tmp1a,tmp3b[0])
							tmp1a=np.append(tmp1a,tmp3b[1])
							tmp1a=np.append(tmp1a,tmp3b[2])
							tmp1a=np.append(tmp1a,tmp3c[0])
							tmp1a=np.append(tmp1a,tmp3c[1])
							tmp1a=np.append(tmp1a,tmp3c[2])
							tmp1a=np.append(tmp1a,tmp3d[0])
							tmp1a=np.append(tmp1a,tmp3d[1])
							tmp1a=np.append(tmp1a,tmp3d[2])
							tmp1a=np.append(tmp1a,tmp3e[0])
							tmp1a=np.append(tmp1a,tmp3e[1])
							tmp1a=np.append(tmp1a,tmp3e[2])
							tmp1a=np.append(tmp1a,tmp3f[0])
							tmp1a=np.append(tmp1a,tmp3f[1])
							tmp1a=np.append(tmp1a,tmp3f[2])
							tmp3g=tmp1a.reshape(len(tmp1a)/18,6,3)
							tmp3h=remove_doubling_dim3_in_perp_space(tmp3g)
							#
							flag=1
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
	else:
		pass
	
	if flag==1:
		tmp1a=np.array([0])
		for i1 in range(len(tmp3h)):
			counter1=0
			for i2 in range(len(tmp3g)):
				if np.all(tmp3h[i1]==tmp3g[i2]):
					counter1+=1
				else:
					pass
			#print '%d %d'%(i1,counter1)
			if counter1==6:
				pass
			elif counter1!=6:
				if len(tmp1a)==1:
					tmp1a=tmp3h[i1].reshape(18)
				else:
					tmp1a=np.append(tmp1a,tmp3h[i1])
		
		if verbose>=2:
			tmp4b=tmp1a.reshape(4,6,3)
			vol2=tetrahedron_volume_6d_numerical(tmp4b)

			tmp1a=np.append(tetrahedron_1,tetrahedron_2)
			tmp1a=np.append(tmp1a,tetrahedron_3)
			tmp1a=np.append(tmp1a,tetrahedron_4)
			tmp4a=tmp1a.reshape(4,4,6,3)
			# Total volume of five tetrahedra
			vol1=obj_volume_6d_numerical(tmp4a)
			
			print '         volume merged tet = %10.8f'%(vol1)
			print '         volume sum 4  tet = %10.8f'%(vol2)
			# sum of two tetrahedra is forming a tetrahedron
			print '         merged'
		else:
			pass
		return tmp4b
	else:
		return np.array([0]).reshape(1,1,1)
"""
cpdef np.ndarray merge_4_tetrahedra_in_obj(np.ndarray[np.int64_t, ndim=4] obj,int verbose):
	# this simplificates obj (set of tetrahedra)
	cdef int i1,i2,i3,i4,i5,counter1,counter2,counter3,counter4,counter5,num1,num2,num3
	cdef long v1,v2,v3
	cdef list a1
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a
	
	if len(obj)>=3:
		
		if verbose>=2:
			print '   merge_4_tetrahedra_in_obj()'
		else:
			pass
	
		# volume of initial obj
		v1,v2,v3=obj_volume_6d(obj)
		if verbose>=3:
			print '      volume initial obj = %d %d %d (%8.6f)'%(v1,v2,v3,(v1+TAU*v2)/float(v3))
		else:
			pass
	
		a1=[0]
		counter4=0
		tmp1a=np.array([0])
		tmp1b=np.array([0])
		for i1 in range(len(obj)-3):
			counter3=0
			for i2 in range(i1+1,len(obj)-2):
				counter2=0
				for i3 in range(i2+1,len(obj)-1):
					counter1=0
					for i4 in range(i3+1,len(obj)):
						counter0=0
						for i5 in a1: # skip tetrahedron already merged
							if i1==0:
								pass 
							elif i1!=i5 and i2!=i5 and i3!=i5 and i4!=i5:
								pass
							else:
								counter0+=1
								break
						if counter0==0:
							tmp3a=merge_four_tetrahedra(obj[i1],obj[i2],obj[i3],obj[i4],verbose)
							if tmp3a.tolist()!=[[[0]]]:
								if verbose>=3:
									print '  %d %d %d %d (merged)'%(i1,i2,i3,i4) 
								else:
									pass
								a1.append(i2)
								a1.append(i3)
								a1.append(i4)
								if len(tmp1a)==1:
									tmp1a=tmp3a.reshape(72)
								else:
									tmp1a=np.append(tmp1a,tmp3a)
								counter1+=1
								counter2+=1
								counter3+=1
								break
							else:
								if verbose>=3:
									print '  %d %d %d %d'%(i1,i2,i3,i4) 
								else:
									pass
						else:
							pass
					if counter1!=0:
						break
					else:
						pass
				if counter2!=0:
					break
				else:
					pass
			if counter3!=0:
				break
			else:
				pass
			
			if i1==0:
				if counter1==0 and counter2==0 and counter3==0:
					if len(tmp1a)==1:
						tmp1a=obj[i1].reshape(72)
					else:
						tmp1a=np.append(tmp1a,obj[i1])
					counter2+=1
				else:
					pass
			else:
				pass
			if counter1==0 and counter2==0 and counter3==0:
				counter5=0
				for i5 in a1:
					if i1==i5:
						counter5+=1
						break
					else:
						pass
				if counter5==0:
					if len(tmp1a)==1:
						tmp1a=obj[i1].reshape(72)
					else:
						tmp1a=np.append(tmp1a,obj[i1])
				else:
					pass
			else:
				pass
				
		#tmp4a=tmp1a.reshape(len(tmp1a)/72,4,6,3)
	
		# 1番目から最後から4つ目の四面体については、mergeされた・されていない場合に
		# 関わらず、tmp1aに加えられている。
		counter1=0
		for i1 in a1:
			if i1==len(obj)-3: # 最後から3番目の四面体がmergeされていない場合
				counter1+=1
				break
			else:
				pass
		if counter1==0:
			tmp1a=np.append(tmp1a,obj[len(obj)-3])
		else:
			pass

		counter1=0
		for i1 in a1:
			if i1==len(obj)-2: # 最後から2番目の四面体がmergeされていない場合
				counter1+=1
				break
			else:
				pass
		if counter1==0:
			tmp1a=np.append(tmp1a,obj[len(obj)-2])
		else:
			pass
	
		counter1=0
		for i1 in a1:
			if i1==len(obj)-1: # 最後の四面体がmergeされていない場合
				counter1+=1
				break
			else:
				pass
		if counter1==0:
			tmp1a=np.append(tmp1a,obj[len(obj)-1])
		else:
			pass

		tmp4a=tmp1a.reshape(len(tmp1a)/72,4,6,3)
	
		# volume of simplized obj
		w1,w2,w3=obj_volume_6d(tmp4a)
		if verbose>=2:
			print '      volume simplified obj = %d %d %d (%8.6f)'%(w1,w2,w3,(w1+TAU*w2)/float(w3))
		else:
			pass
	
		if v1==w1 and v2==w2 and v3==w3:
			return tmp4a
		else:
			if verbose>=2:
				print '      fail'
			else:
				pass	
			return np.array([0]).reshape(1,1,1,1)
	else:
		return obj
"""

cpdef np.ndarray merge_5_tetrahedra_in_obj(np.ndarray[np.int64_t, ndim=4] obj,int verbose):
	# this simplificates obj (set of tetrahedra)
	cdef int i1,i2,i3,i4,i5,i6,counter0,counter1,counter2,counter3,counter4,counter5,num1,num2,num3
	cdef long v1,v2,v3
	cdef double vol1,vol2
	cdef list a1
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a
	#cdef np.ndarray[np.int64_t,ndim=3] tmp3a
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a
	
	if len(obj)>=5:
		
		if verbose>=2:
			print '         merge_5_tetrahedra_in_obj()'
		else:
			pass
	
		# volume of initial obj
		#v1,v2,v3=obj_volume_6d(obj)
		vol1=obj_volume_6d_numerical(obj)
		if verbose>=3:
			#print '         volume initial obj = %d %d %d (%8.6f)'%(v1,v2,v3,(v1+TAU*v2)/float(v3))
			print '         volume initial obj = %10.8f'%(vol1)
		else:
			pass
	
		a1=[]
		counter5=0
		tmp1a=np.array([0])
		tmp1b=np.array([0])
		for i1 in range(len(obj)-4):
			counter4=0
			for i2 in range(i1+1,len(obj)-3):
				counter3=0
				for i3 in range(i2+1,len(obj)-2):
					counter2=0
					for i4 in range(i3+1,len(obj)-1):
						counter1=0
						for i5 in range(i4+1,len(obj)):
							counter0=0
							if len(a1)!=0:
								for i6 in a1: # skip tetrahedron already merged
									if i1!=i6 and i2!=i6 and i3!=i6 and i4!=i6 and i5!=i6:
										pass
									else:
										counter0+=1
										break
							if counter0==0:
								tmp4a=merge_five_tetrahedra(obj[i1],obj[i2],obj[i3],obj[i4],obj[i5],verbose)
								if tmp4a.tolist()!=[[[[0]]]]:
									if verbose>=3:
										print '         %d %d %d %d %d(merged)'%(i1,i2,i3,i4,i5) 
									else:
										pass
									a1.append(i1)
									a1.append(i2)
									a1.append(i3)
									a1.append(i4)
									a1.append(i5)
									if len(tmp1a)==1:
										tmp1a=tmp4a.reshape(72)
									else:
										tmp1a=np.append(tmp1a,tmp4a)
									counter1+=1
									counter2+=1
									counter3+=1
									counter4+=1
									break
								else:
									if verbose>=3:
										print '         %d %d %d %d %d'%(i1,i2,i3,i4,i5) 
									else:
										pass
							else:
								pass
						if counter1!=0:
							break
						else:
							pass
					if counter2!=0:
						break
					else:
						pass
				if counter3!=0:
					break
				else:
					pass
			if counter4!=0:
				break
			else:
				pass
		
		if len(a1)!=0:
			for i1 in range(len(obj)):
				counter1=0
				for i2 in a1: # skip tetrahedron already merged
					if i1==i2:
						counter1+=1
						break
					else:
						pass
				if counter1==0:
					tmp1a=np.append(tmp1a,obj[i1])
				else:
					pass
		else:
			tmp1a=obj[0].reshape(72)
			for i1 in range(1,len(obj)):
				tmp1a=np.append(tmp1a,obj[i1])
		tmp4a=tmp1a.reshape(len(tmp1a)/72,4,6,3)
		# volume of simplized obj
		#w1,w2,w3=obj_volume_6d(tmp4a)
		vol2=obj_volume_6d_numerical(tmp4a)
		if verbose>=2:
			#print '         volume simplified obj = %d %d %d (%8.6f)'%(w1,w2,w3,(w1+TAU*w2)/float(w3))
			print '         volume simplified obj = %10.8f'%(vol2)
		else:
			pass
		#if v1==w1 and v2==w2 and v3==w3:
		if abs(vol1-vol2)<1e-8:
			return tmp4a
		else:
			if verbose>=2:
				print '         fail'
			else:
				pass	
			return np.array([0]).reshape(1,1,1,1)
	else:
		return obj

cpdef np.ndarray merge_4_tetrahedra_in_obj(np.ndarray[np.int64_t, ndim=4] obj,int verbose):
	# this simplificates obj (set of tetrahedra)
	cdef int i1,i2,i3,i4,i5,counter0,counter1,counter2,counter3,counter4,counter5,num1,num2,num3
	cdef long v1,v2,v3
	cdef double vol1,vol2
	cdef list a1
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a
	
	if len(obj)>=4:
		
		if verbose>=2:
			print '         merge_4_tetrahedra_in_obj()'
		else:
			pass
	
		# volume of initial obj
		#v1,v2,v3=obj_volume_6d(obj)
		vol1=obj_volume_6d_numerical(obj)
		if verbose>=3:
			#print '         volume initial obj = %d %d %d (%8.6f)'%(v1,v2,v3,(v1+TAU*v2)/float(v3))
			print '         volume initial obj = %10.8f'%(vol1)
		else:
			pass
	
		a1=[]
		counter4=0
		tmp1a=np.array([0])
		tmp1b=np.array([0])
		for i1 in range(len(obj)-3):
			counter3=0
			for i2 in range(i1+1,len(obj)-2):
				counter2=0
				for i3 in range(i2+1,len(obj)-1):
					counter1=0
					for i4 in range(i3+1,len(obj)):
						counter0=0
						if len(a1)!=0:
							for i5 in a1: # skip tetrahedron already merged
								if i1!=i5 and i2!=i5 and i3!=i5 and i4!=i5:
									pass
								else:
									counter0+=1
									break
						if counter0==0:
							tmp3a=merge_four_tetrahedra(obj[i1],obj[i2],obj[i3],obj[i4],verbose)
							if tmp3a.tolist()!=[[[0]]]:
								if verbose>=3:
									print '         %d %d %d %d (merged)'%(i1,i2,i3,i4) 
								else:
									pass
								a1.append(i1)
								a1.append(i2)
								a1.append(i3)
								a1.append(i4)
								if len(tmp1a)==1:
									tmp1a=tmp3a.reshape(72)
								else:
									tmp1a=np.append(tmp1a,tmp3a)
								counter1+=1
								counter2+=1
								counter3+=1
								break
							else:
								if verbose>=3:
									print '         %d %d %d %d'%(i1,i2,i3,i4) 
								else:
									pass
						else:
							pass
					if counter1!=0:
						break
					else:
						pass
				if counter2!=0:
					break
				else:
					pass
			if counter3!=0:
				break
			else:
				pass
		if len(a1)!=0:
			for i1 in range(len(obj)):
				counter1=0
				for i2 in a1: # skip tetrahedron already merged
					if i1==i2:
						counter1+=1
						break
					else:
						pass
				if counter1==0:
					tmp1a=np.append(tmp1a,obj[i1])
				else:
					pass
		else:
			tmp1a=obj[0].reshape(72)
			for i1 in range(1,len(obj)):
				tmp1a=np.append(tmp1a,obj[i1])
		

		tmp4a=tmp1a.reshape(len(tmp1a)/72,4,6,3)
	
		# volume of simplized obj
		#w1,w2,w3=obj_volume_6d(tmp4a)
		vol2=obj_volume_6d_numerical(tmp4a)
		if verbose>=2:
			#print '         volume simplified obj = %d %d %d (%8.6f)'%(w1,w2,w3,(w1+TAU*w2)/float(w3))
			print '         volume simplified obj = %10.8f'%(vol2)
		else:
			pass
	
		#if v1==w1 and v2==w2 and v3==w3:
		if abs(vol1-vol2)<1e-8:
			return tmp4a
		else:
			if verbose>=2:
				print '         fail'
			else:
				pass	
			return np.array([0]).reshape(1,1,1,1)
	else:
		return obj

cpdef np.ndarray merge_3_tetrahedra_in_obj(np.ndarray[np.int64_t, ndim=4] obj,int verbose):
	# this simplificates obj (set of tetrahedra)
	cdef int i1,i2,i3,i4,i5,counter1,counter2,counter3,counter4,counter5,num1,num2
	cdef long v1,v2,v3
	cdef double vol1,vol2
	cdef list a1
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a
	
	# volume of initial obj
	#v1,v2,v3=obj_volume_6d(obj)
	vol1=obj_volume_6d_numerical(obj)
	if verbose>=3:
		#print '         volume initial obj = %d %d %d (%8.6f)'%(v1,v2,v3,(v1+TAU*v2)/float(v3))
		print '         volume initial obj = %10.8f'%(vol1)
	else:
		pass
	
	if len(obj)>=3:
		if verbose>=2:
			print '        merge_3_tetrahedra_in_obj()'
		else:
			pass
		
		a1=[]
		counter4=0
		tmp1a=np.array([0])
		tmp1b=np.array([0])
		for i1 in range(len(obj)-2):
			counter3=0
			for i2 in range(i1+1,len(obj)-1):
				counter2=0
				for i3 in range(i2+1,len(obj)):
					counter1=0
					if len(a1)!=0:
						for i4 in a1: # skip tetrahedron already merged
							if i1!=i4 and i2!=i4 and i3!=i4:
								pass
							else:
								counter1+=1
								break
					else:
						pass
					if counter1==0:
						tmp3a=merge_three_tetrahedra(obj[i1],obj[i2],obj[i3],verbose)
						if tmp3a.tolist()!=[[[0]]]:
							if verbose>=3:
								print '         %d %d %d (merged)'%(i1,i2,i3) 
							else:
								pass
							a1.append(i1)
							a1.append(i2)
							a1.append(i3)
							if len(tmp1a)==1:
								tmp1a=tmp3a.reshape(72)
							else:
								tmp1a=np.append(tmp1a,tmp3a)
							counter2+=1
						else:
							if verbose>=3:
								print '         %d %d %d'%(i1,i2,i3) 
							else:
								pass
					else:
						pass
				if counter2!=0:
					break
				else:
					pass
					
		if len(a1)!=0:
			for i1 in range(len(obj)):
				counter1=0
				for i2 in a1: # skip tetrahedron already merged
					if i1==i2:
						counter1+=1
						break
					else:
						pass
				if counter1==0:
					tmp1a=np.append(tmp1a,obj[i1])
				else:
					pass
		else:
			tmp1a=obj[0].reshape(72)
			for i1 in range(1,len(obj)):
				tmp1a=np.append(tmp1a,obj[i1])
		
		tmp4a=tmp1a.reshape(len(tmp1a)/72,4,6,3)
	
		# volume of simplized obj
		#w1,w2,w3=obj_volume_6d(tmp4a)
		vol2=obj_volume_6d_numerical(tmp4a)
		if verbose>=2:
			#print '         volume simplified obj = %d %d %d (%8.6f)'%(w1,w2,w3,(w1+TAU*w2)/float(w3))
			print '         volume simplified obj = %10.8f'%(vol2)
		else:
			pass
	
		#if v1==w1 and v2==w2 and v3==w3:
		if abs(vol1-vol2)<1e-8:
			return tmp4a
		else:
			if verbose>=2:
				print '         fail'
			else:
				pass	
			return np.array([0]).reshape(1,1,1,1)
	else:
		return obj

cpdef np.ndarray merge_2_tetrahedra_in_obj(np.ndarray[np.int64_t, ndim=4] obj,int verbose):
	# this simplificates obj (set of tetrahedra)
	cdef int i1,i2,i3,i4,counter1,counter2,counter3,counter4,num
	cdef long v1,v2,v3,w1,w2,w3
	cdef double vol1,vol2
	cdef list a1
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a
	cdef np.ndarray[np.int64_t,ndim=4] tmp4a
	
	if len(obj)>=2:
		if verbose>=2:
			print '         merge_2_tetrahedra_in_obj()'
		else:
			pass
	
		# volume of initial obj
		v1,v2,v3=obj_volume_6d(obj)
		vol1=obj_volume_6d_numerical(obj)
		if verbose>=3:
			print '         volume initial obj = %d %d %d (%10.8f)'%(v1,v2,v3,(v1+TAU*v2)/float(v3))
			print '                            = %10.8f'%(vol1)
		else:
			pass
	
		a1=[]
		counter3=0
		tmp1a=np.array([0])
		tmp1b=np.array([0])
		for i1 in range(len(obj)-1):
			counter2=0
			for i2 in range(i1+1,len(obj)):
				counter1=0
				if len(a1)!=0:
					for i3 in a1: # skip tetrahedron already merged
						if i1!=i3 and i2!=i3:
							pass
						else:
							counter1+=1
							break
				else:
					pass
				if counter1==0:
					tmp3a=merge_two_tetrahedra(obj[i1],obj[i2],verbose)
					if tmp3a.tolist()!=[[[0]]]:
						if verbose>=3:
							print '         %d %d (merged)'%(i1,i2) 
						else:
							pass
						a1.append(i1)
						a1.append(i2)
						if len(tmp1a)==1:
							tmp1a=tmp3a.reshape(72)
						else:
							tmp1a=np.append(tmp1a,tmp3a)
						counter2+=1
						break
					else:
						if verbose>=3:
							print '         %d %d'%(i1,i2) 
						else:
							pass
				else:
					pass
				if counter2!=0:
					break
				else:
					pass
		
		if len(a1)!=0:
			for i1 in range(len(obj)):
				counter1=0
				for i2 in a1: # skip tetrahedron already merged
					if i1==i2:
						counter1+=1
						break
					else:
						pass
				if counter1==0:
					tmp1a=np.append(tmp1a,obj[i1])
				else:
					pass
		else:
			tmp1a=obj[0].reshape(72)
			for i1 in range(1,len(obj)):
				tmp1a=np.append(tmp1a,obj[i1])
		
		tmp4a=tmp1a.reshape(len(tmp1a)/72,4,6,3)
	
		# volume of simplized obj
		w1,w2,w3=obj_volume_6d(tmp4a)
		vol2=obj_volume_6d_numerical(tmp4a)
		if verbose>=2:
			print '         volume simplified obj = %d %d %d (%8.6f)'%(w1,w2,w3,(w1+TAU*w2)/float(w3))
			print '                               = %10.8f'%(vol2)
		else:
			pass
	
		#if v1==w1 and v2==w2 and v3==w3:
		if abs(vol1-vol2)<1e-8:
			return tmp4a
		else:
			if verbose==2:
				print '         fail'
			else:
				pass	
			return np.array([0]).reshape(1,1,1,1)
	else:
		return obj

cdef int equivalent_tetrahedron(np.ndarray[np.int64_t, ndim=3] tetrahedron1,\
								np.ndarray[np.int64_t, ndim=3] tetrahedron2):
	cdef int i1,i2
	cdef np.ndarray[np.int64_t,ndim=1] tmp1a
	cdef np.ndarray[np.int64_t,ndim=3] tmp3a
	#
	tmp1a=np.append(tetrahedron1,tetrahedron2)
	tmp3a=remove_doubling_dim3_in_perp_space(tmp1a.reshape(len(tmp1a)/18,6,3))
	if len(tmp3a)==4:
		return 0 # equivalent
	else:
		return 1 # not equivalent

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
	#
	
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
		print '   -> N of tetrahedron: %3d'%(len(ltmp))
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
					print '     %d-th tet, empty'%(i)
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
						print '     %d-th tet, volume : %d %d %d (%8.6f)'%(i,v1,v2,v3,(v1+v2*TAU)/float(v3))
					else:
						pass
				else:
					tmp1a=np.append(tmp1a,tmp3b)
					if verbose==1:
						print '     %d-th tet, volume : %d %d %d (%8.6f)'%(i,v1,v2,v3,(v1+v2*TAU)/float(v3))
					else:
						pass
				w1,w2,w3=add(v1,v2,v3,w1,w2,w3)
				counter+=1
		if counter!=0:
			tmp4a=tmp1a.reshape(len(tmp1a)/72,4,6,3) # 4*6*3=72	
			if verbose==1:
				print '     -> Total : %d %d %d (%8.6f)'%(w1,w2,w3,(w1+w2*TAU)/float(w3))
			else:
				pass
		else:
			tmp4a=np.array([0],dtype=np.int).reshape(1,1,1,1)
		return tmp4a
	else:
		print 'tmp2v',tmp2v
		return np.array([0],dtype=np.int).reshape(1,1,1,1)

cpdef np.ndarray tetrahedralization(np.ndarray[np.int64_t, ndim=3] tetrahedron,np.ndarray[np.int64_t, ndim=3] intersecting_point):
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
			tmp3b=np.array([tmp3a[ltmp[i][0]],tmp3a[ltmp[i][1]],tmp3a[ltmp[i][2]],tmp3a[ltmp[i][3]]],dtype=np.int).reshape(4,6,3)
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
			tmp4a=np.array([0],dtype=np.int).reshape(1,1,1,1)
		return tmp4a
	else:
		return np.array([0],dtype=np.int).reshape(1,1,1,1)

cdef list decomposition(np.ndarray[DTYPE_t, ndim=2] tmp2v):
	cdef int i
	cdef list tmp=[]
	try:
		tri=Delaunay(tmp2v)
	except:
		print 'error in decomposition()'
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
	return np.array([[c1,c2,c3],[c4,c5,c6],[c7,c8,c9]],dtype=np.int)

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
		print '              intersection_segment_surface()'
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
	#if abs(val1)<EPS:
	if f1==0 and f2==0:
		if verbose>=2:
			print '   line segment and triangle are parallel'
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
			print '   Intersectiong point:',tmp1
		else:
			pass
		return tmp1
	#elif f1!=0 or f2!=0:
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
			#g4,g5,g6 = div(f4,f5,f6,f1,f2,f3) # val2/val1 in TAU-style
			#if (g4>=0 and g5>=0)   g4+TAU*g5<=g6
			f7,f8,f9=det_matrix(vecCD,vecCA,vecBA)
			val3=(f7+f8*TAU)/float(f9)
			if val3/val1>=0.0 and (val2+val3)/val1<=1.0:
				#g7,g8,g9 = div(f7,f8,f9,f1,f2,f3) # val3/val1 in TAU-style
				f10,f11,f12=det_matrix(vecCD,vecCE,vecCA)
				val4=(f10+f11*TAU)/float(f12)
				if val4/val1>=0.0 and val4/val1<=1.0: # t = val4/val1
					
					#print 'f=',f10,f11,f12,f1,f2,f3
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
					#if j3!=0 and j6!=0 and j9!=0 and j12!=0 and j15!=0 and j18!=0:
					#	tmp1=np.array([j1,j2,j3,j4,j5,j6,j7,j8,j9,j10,j11,j12,j13,j14,j15,j16,j17,j18],dtype=np.int)
					#	if np.all(tmp1<1000000):
					#		return np.array([j1,j2,j3,j4,j5,j6,j7,j8,j9,j10,j11,j12,j13,j14,j15,j16,j17,j18],dtype=np.int)
					#	else:
					#		return np.array([0],dtype=np.int)
					#else:
					#	return np.array([0],dtype=np.int)
					tmp1=np.array([j1,j2,j3,j4,j5,j6,j7,j8,j9,j10,j11,j12,j13,j14,j15,j16,j17,j18])					
					#
					# to avoid overfllow
					#
					"""
					if abs(f10)>1000000 or abs(f11)>1000000 or abs(f12)>1000000 or abs(f1)>1000000 or abs(f2)>1000000 or abs(f3)>1000000:
						return np.array([0],dtype=np.int)
					else:
						#print 'f=',f10,f11,f12,f1,f2,f3
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
						#if j3!=0 and j6!=0 and j9!=0 and j12!=0 and j15!=0 and j18!=0:
						#	tmp1=np.array([j1,j2,j3,j4,j5,j6,j7,j8,j9,j10,j11,j12,j13,j14,j15,j16,j17,j18],dtype=np.int)
						#	if np.all(tmp1<1000000):
						#		return np.array([j1,j2,j3,j4,j5,j6,j7,j8,j9,j10,j11,j12,j13,j14,j15,j16,j17,j18],dtype=np.int)
						#	else:
						#		return np.array([0],dtype=np.int)
						#else:
						#	return np.array([0],dtype=np.int)
						return np.array([j1,j2,j3,j4,j5,j6,j7,j8,j9,j10,j11,j12,j13,j14,j15,j16,j17,j18],dtype=np.int)
					"""
				else:
					pass
			else:
				pass
		else:
			pass
		return tmp1

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
		print '                        intersection_two_segment()'
	else:
		pass
		
	if verbose>=2:
		print 'segment_1',segment_1_A
		print 'segment_1',segment_1_B
		print 'segment_2',segment_2_C
		print 'segment_2',segment_2_D
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
			print '                           s = %8.6f'%(s)
		else:
			pass
		if c1!=0 or c2!=0:
			t1,t2,t3=div(h1,h2,h3,c1,c2,c3)
			t=(t1+TAU*t2)/float(t3)
			if verbose>=2:
				print '                           t = %8.6f'%(t)
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
					print '      ddx1,ddx2 = %d %d'%(ddx1,ddx2)
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
					print '      ddy1,ddy2 = %d %d'%(ddy1,ddy2)
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
					print '      ddz1,ddz2 = %d %d'%(ddz1,ddz2)
				else:
					pass
					
				z1,z2,z3=add(ddx1,ddx2,ddx3,ddy1,ddy2,ddy3)
				z1,z2,z3=add(z1,z2,z3,ddz1,ddz2,ddz3)
				if verbose>=2:
					print '      z1,z2 = %d %d'%(z1,z2)
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

cpdef np.ndarray intersection_two_triangle(np.ndarray[np.int64_t, ndim=3] triangle_1,np.ndarray[np.int64_t, ndim=3] triangle_2):
	cdef int i,j,counter,num1,num2,num3
	cdef np.ndarray[np.int64_t,ndim=1] p,tmp,tmp1
	cdef np.ndarray[np.int64_t,ndim=2] combination_index
	#
	# -----------------
	# triangle_1
	# -----------------
	# vertex 1: triangle_1[0],  consist of (a1+b1*TAU)/c1, ... (a6+b6*TAU)/c6	a_i,b_i,c_i = tetrahedron_1[0][i:0~5][0],triangle_1[0][i:0~5][1],triangle_1[0][i:0~5][2]
	# vertex 2: triangle_1[1]
	# vertex 3: triangle_1[2]
	#
	# triangle
	# triangle 1: v1,v2,v3
	#
	# 3 edges
	# edge 1: v1,v2
	# edge 2: v1,v3
	# edge 3: v2,v3
	#
	# -----------------
	# triangle_2
	# -----------------
	# vertex 1: triangle_2[0]
	# vertex 2: triangle_2[1]
	# vertex 3: triangle_2[2]
	#
	# triangle
	# triangle 1: w1,w2,w3
	#
	# 3 edges of triangle_2
	# edge 1: w1,w2
	# edge 2: w1,w3
	# edge 3: w2,w3

	#
	# case 1: intersection between (edge of triangle_1) and (triangle_2)
	# case 2: intersection between (edge of triangle_2) and (triangle_1)
	#
	# combination_index
	# e.g. v1,v2,w1,w2,w3 (edge 1 and triangle 1) ...
	combination_index=np.array([\
	[0,1,0,1,2],\
	[0,2,0,1,2],\
	[1,2,0,1,2]])
	#
	count=0
	for i in range(len(combination_index)): # len(combination_index) = 3
		# case 1: intersection betweem
		# 3 edges of triangle_1
		# triangle_2
		segment_1=triangle_1[combination_index[i][0]] 
		segment_2=triangle_1[combination_index[i][1]]
		surface_1=triangle_2[combination_index[i][2]]
		surface_2=triangle_2[combination_index[i][3]]
		surface_3=triangle_2[combination_index[i][4]]
		tmp=intersection_segment_surface(segment_1,segment_2,surface_1,surface_2,surface_3)
		if len(tmp)!=1:
			if count==0 :
				p=tmp # intersection
			else:
				p=np.append(p,tmp) # intersecting points
			count+=1
		else:
			pass
		# case 2: intersection betweem
		# 3 edges of triangle_2
		# triangle_1
		segment_1=triangle_2[combination_index[i][0]]
		segment_2=triangle_2[combination_index[i][1]]
		surface_1=triangle_1[combination_index[i][2]]
		surface_2=triangle_1[combination_index[i][3]]
		surface_3=triangle_1[combination_index[i][4]]
		tmp=intersection_segment_surface(segment_1,segment_2,surface_1,surface_2,surface_3)
		if len(tmp)!=1:
			if count==0:
				p=tmp # intersection
			else:
				p=np.append(p,tmp) # intersecting points
			count+=1
		else:
			pass
	if count==0: # if no intersection
		return np.array([0],dtype=np.int)
	else:	
		return p
	
###########################
#	 Numerical Calc	  #
###########################

cdef intersection_segment_surface_cy(double ea0,double ea1,double ea2,double ea3,double ea4,double ea5,double eb0,double eb1,double eb2,double eb3,double eb4,double eb5,double sa0,double sa1,double sa2,double sa3,double sa4,double sa5,double sb0,double sb1,double sb2,double sb3,double sb4,double sb5,double sc0,double sc1,double sc2,double sc3,double sc4,double sc5):
	# calc intersection between a line segment (line1a,line1b) and a plane (plane2a,plane2b,plane2c)
	# Reference: https://shikousakugo.wordpress.com/2012/06/27/ray-intersection-2/

	#cdef double s,t,u,a1x,a1y,a1z,a2x,a2y,a2z,b1x,b1y,b1z,b2x,b2y,b2z,b3x,b3y,b3z,bunbo
	cdef double s,t,u,a2x,a2y,a2z,b1x,b1y,b1z,b2x,b2y,b2z,b3x,b3y,b3z,bunbo
	cdef np.ndarray[DTYPE_t,ndim=1] line1a,line1b,interval
	cdef np.ndarray[DTYPE_t,ndim=1] vec1,vecBA,vecCD,vecCE,vecCA

	#print ea0,ea1,ea2,ea3,ea4,ea5,eb0,eb1,eb2,eb3,eb4,eb5,sa0,sa1,sa2,sa3,sa4,sa5,sb0,sb1,sb2,sb3,sb4,sb5,sc0,sc1,sc2,sc3,sc4,sc5
	line1a=np.array([ea0,ea1,ea2,ea3,ea4,ea5], dtype=np.float64)
	line1b=np.array([eb0,eb1,eb2,eb3,eb4,eb5], dtype=np.float64)
	aa1=projection_numerical(ea0,ea1,ea2,ea3,ea4,ea5)
	aa2=projection_numerical(eb0,eb1,eb2,eb3,eb4,eb5)
	bb1=projection_numerical(sa0,sa1,sa2,sa3,sa4,sa5)
	bb2=projection_numerical(sb0,sb1,sb2,sb3,sb4,sb5)
	bb3=projection_numerical(sc0,sc1,sc2,sc3,sc4,sc5)
	# line segment
	#a1x,a1y,a1z=aa1[3],aa1[4],aa1[5] # set this as an origin. A
	a2x,a2y,a2z=aa2[3]-aa1[3],aa2[4]-aa1[4],aa2[5]-aa1[5] # AB
	# plane CDE
	b1x,b1y,b1z=bb1[3]-aa1[3],bb1[4]-aa1[4],bb1[5]-aa1[5] # AC
	b2x,b2y,b2z=bb2[3]-bb1[3],bb2[4]-bb1[4],bb2[5]-bb1[5] # CD
	b3x,b3y,b3z=bb3[3]-bb1[3],bb3[4]-bb1[4],bb3[5]-bb1[5] # CE
	#print 'AB = %10.8f %10.8f %10.8f'%(a2x,a2y,a2z)
	#print 'AC = %10.8f %10.8f %10.8f'%(b1x,b1y,b1z)
	#print 'CD = %10.8f %10.8f %10.8f'%(b2x,b2y,b2z)
	#print 'CE = %10.8f %10.8f %10.8f'%(b3x,b3y,b3z)
	
	vecBA=np.array([-a2x,-a2y,-a2z]) # line segment BA
	vecCD=np.array([ b2x, b2y, b2z]) # edge segment of triangle CDE, CD
	vecCE=np.array([ b3x, b3y, b3z]) # edge segment of triangle CDE, CE
	vecCA=np.array([-b1x,-b1y,-b1z]) # CA

	bunbo=det_matrix_cy(vecCD,vecCE,vecBA)
	#print ' bunbo = ',bunbo
	if abs(bunbo)<EPS:
		return 1
	else:
		u=det_matrix_cy(vecCA,vecCE,vecBA)/bunbo
		#print ' u = ',u
		if u>=0.0 and u<=1.0:
			v=det_matrix_cy(vecCD,vecCA,vecBA)/bunbo
			#print ' v = ',v
			if v>=0.0 and u+v<=1.0:
				t=det_matrix_cy(vecCD,vecCE,vecCA)/bunbo
				#print ' t = ',t
				if t>=0.0 and t<=1.0:
					interval=line1a+t*(line1b-line1a)
					interval_1=projection_numerical(interval[0],interval[1],interval[2],interval[3],interval[4],interval[5])
					#print 'intersection_segment_surface: interval=',interval_1
					#print 's,t,u	= ',s,t,u
					return interval[0],interval[1],interval[2],interval[3],interval[4],interval[5],interval_1[3],interval_1[4],interval_1[5],u,v,t,bunbo
				else:
					return 1
			else:
				return 1
		else:
			return 1

cpdef det_matrix_cy(np.ndarray a, np.ndarray b, np.ndarray c):
	cdef double a1,a2,a3,b1,b2,b3,c1,c2,c3
	return a[0]*b[1]*c[2]+a[2]*b[0]*c[1]+a[1]*b[2]*c[0]-a[2]*b[1]*c[0]-a[1]*b[0]*c[2]-a[0]*b[2]*c[1]

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
	#a1,a2,a3,a4,a5,a6=projection(n1,n2,n3,n4,n5,n6)
	#return [(a4[0]+TAU*a4[1])/float(a4[2]),(a5[0]+TAU*a5[1])/float(a5[2]),(a6[0]+TAU*a6[1])/float(a6[2])]

cpdef double tetrahedron_volume_6d_numerical(np.ndarray[np.int64_t, ndim=3] tetrahedron):
	cdef double x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3,z3
	cdef np.ndarray[np.int64_t,ndim=1] x1e,y1e,z1e,x1i,y1i,z1i,x2i,y2i,z2i,x3i,y3i,z3i,x4i,y4i,z4i
	x0,y0,z0=get_internal_component_numerical(tetrahedron[0])
	x1,y1,z1=get_internal_component_numerical(tetrahedron[1])
	x2,y2,z2=get_internal_component_numerical(tetrahedron[2])
	x3,y3,z3=get_internal_component_numerical(tetrahedron[3])
	return tetrahedron_volume_numerical(x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3,z3)
"""
cdef double tetrahedron_volume_numerical(double x0, double y0, double z0, double x1, double y1, double z1, double x2, double y2, double z2, double x3, double y3, double z3):
	# This function returns volume of a tetrahedron
	#input: 
	# vertex coordinates of the tetrahedron
	# (x0,y0,z0),(x1,y1,z1),(x2,y2,z2),(x3,y3,z3)  
	cdef double detm,vol
	# cdef np.ndarray m = np.zeros((0,6), dtype=np.float)
	cdef np.ndarray[np.float64_t, ndim=2] m

	m = np.array([[x1-x0,y1-y0,z1-z0],[x2-x0,y2-y0,z2-z0],[x3-x0,y3-y0,z3-z0]])
	detm = np.linalg.det(m)
	vol = abs(detm)/6.0
	return vol
"""

cpdef double obj_volume_6d_numerical(np.ndarray[np.int64_t, ndim=4] obj):
	cdef int i
	cdef double volume,vol
	volume=0.0
	for i in range(len(obj)):
		vol=tetrahedron_volume_6d_numerical(obj[i])
		volume+=vol	
	return volume

	