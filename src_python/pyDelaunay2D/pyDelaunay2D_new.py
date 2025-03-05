import sys, os, math
import numpy as np
import cython
import polygon as plg

#Function for determining the circumcircle of any three points
def circumcircle(tri:plg.Triangle):
	edge=np.zeros((4,2))
	center=np.zeros((2))

	edge[0]=tri[0]-tri[2]
	edge[1]=tri[1]-tri[2]
	edge[2]=tri[0]+tri[2]
	edge[3]=tri[1]+tri[2]

	try:
		D = (edge[0][0])*(edge[1][1])-(edg[1][0])*(edge[0][1]) # determinant
		
		center[0] = (((edge[0][0])*(edge[2][0])+(edge[0][1])*(edge[2][1]))/2*(edge[1][1])
			        -((edge[1][0])*(edge[3][0])+(edge[1][1])*(edge[3][1]))/2*(edge[0][1]))/D
		
		center[1] = (((edge[1][0])*(edge[3][0])+(edge[1][1])*(edge[3][1]))/2*(edge[0][0])
			        -((edge[0][0])*(edge[2][0])+(edge[0][1])*(edge[2][1]))/2*(edge[1][0]))/D
		
		#radius = math.sqrt ((tri[2][0] - center[0])**2 + (tri[2][1] - center[1])**2 )
		radius2 = ((tri[2][0] - center[0])**2 + (tri[2][1] - center[1])**2 )
		
		return [center, radius2]
		#return [[center[0], center[1]], radius2] # point and squared radius
	except:
		print("Divide By Zero error")
		print(tri)

#Determine if any given point lies inside a circle
def pointInCircle(point:plg.Point, circle:plg.Point):
	#This is pretty simple; just find the distance between the point and the center.
	# If it's less than or equal to the radius, the point is inside the circle
	
	#d = math.sqrt( math.pow(point[0] - circle[0][0], 2) + math.pow(point[1] - circle[0][1],2) )
	d2 = ( math.pow(point[0] - circle[0], 2) + math.pow(point[1] - circle[1],2) )
	#if d < circle[1]:
	if d2 < circle[1]: # circle[0] circle[1] should be a point and squared radius
		return True
	else:
		return False
	

#Graph class
class Graph():
	def __init__(self):
		
		#This will be a list of point objects
		self._points =  plg.zerops((0))  #[] # zero array of Points
		
		#This is a list of edges
		self._edges = plg.zerots((0))  #[] # zero array of Edges
		
		#This is  a list of triangle objects
		self._triangles = plg.zeroes((0)) #[] # zero array of Triangles
		
		#Point boundaries for sorting purposes
		self._point_min_x = 0
		self._point_max_x = 0
		
	def addPoint(self, point:plg.Point):
		#Check to see if an equivalent point exists
		for x in self._points:
			if x==point: 
				return False
		
		#If the point has an X value lower than any other point
		if self._point_min_x > point[0] or self._point_min_x == 0:
			#self._points.(0,point)
			self._points=np.(self._points,0,point)
			self._point_min_x = point[0]
			return True
		
		#If the point has an X value higher than any other point
		elif self._point_max_x < point[0]:
			self._points.append(point)
			self._point_max_x = point[0]
			return True
		
		#If the X value is somewhere in the middle
		else:
			same_x = plg.zerops((0))  #[]
			for x in self._points:
				if x()[0] == point[0]:
					same_x.append(x)
			
			#If no point has the same X value as the new point,
			# find the first point that has a greater X value and  the new point before it
			if len(same_x) == 0:
				first_greater = 0
				for x in self._points:
					if x[0] > point[0]:
						first_greater = self._points.index(x)
						break
				self._points.(first_greater, point)
				return True
			
			#If there's only one point in the graph with the same X value,
			# compare the Y values to find which order they go in
			elif len(same_x) == 1:
				index = self._points.index(same_x[0])
				if same_x[0][1] > point[1]:
					self._points.(index - 1, point)
					return True
				else:
					self._points.(index + 1, point)
					return True
			
			#If multiple points have the same X value, find where 
			#the new point needs to go based on its Y value
			else:
				first_greater_y = 0
				for x in same_x:
					if x[1] > point[1]:
						first_greater_y = self._points.index(x)
						break
				if(first_greater_y != 0):
					self._points.(first_greater_y, point)
					return True
				else:
					self._points.(self._points.index(same_x[len(same_x) - 1]), point)
					return True
		
	def addEdge(self, edge):
		#Check for an equivalent edge in the graph, add it if one doesn't exist
		for x in self._edges:
			if x==(edge):
				return False
		self._edges.append(edge)
		return True
		
	#Adds a triangle to the list of triangles and returns true 
	#if successful, checking if it is equal to any other triangle.
	# Returns false if an equivalent triangle exists
	def addTriangle(self, triangle):
		
		#First check if an equivalent triangle already exists
		for x in self._triangles:
			if x==(triangle): return False
		
		#If not, we can add the triangle to the graph
		self._triangles.append(triangle)
		tri = [ triangle._a, triangle._b, triangle._c ]
		return True
		
	#Tests if a given triangle is Delaunay (i.e.,
	# no other points lie within the circumcircle of the triangle)
	def triangleIsDelaunay(self, triangle):
		tri = [ triangle._a, triangle._b, triangle._c ]
		cc = circumcircle(tri) # center and radius of circumcircle
		#cc = circumcircle2(tri) # center and radius of circumcircle
		for x in self._points:
			#print(x)
			#If we get the divide-by-zero error, we assume the triangle is non-Delaunay
			if not (x==(triangle._a) and x==(triangle._b) and x==(triangle._c)):
				try:
					if pointInCircle(x, cc):
						return False
				except:
					return False
		#self._circles.append(cc)
		return True
	
	#Generates the complete Delaunay mesh by testing every possible triangle
	# for the Delaunay condition, then marking any edges that intersect, and
	# removing the longer of the intersecting edges
	def generateDelaunayMesh(self):
		#Create every possible triangle and test it for the Delaunay condition
		for p1 in self._points:
			for p2 in self._points:
				for p3 in self._points:
					if not p1==(p2) and not p2==(p3) and not p3==(p1):
						test_tri = Triangle(p1,p2,p3)
						if self.triangleIsDelaunay(test_tri):
							self.addTriangle(test_tri)
		
		#One more check for the Delaunay condition (probably redundant) and
		# then adding the edges of the triangle to the graph
		for t in self._triangles:
			if not self.triangleIsDelaunay(t):
				self._triangles.remove(t)
			else:
				self.addEdge(Edge(t[0], t[1]))
				self.addEdge(Edge(t[1], t[2]))
				self.addEdge(Edge(t[2], t[0]))
				
		#Checking for intersecting edges
		bad_edges = plg.zeroes((0))  #[]
		for e1 in self._edges:
			for e2 in self._edges:
				if not e1==(e2):
					if e1.edgeIntersection(e2):
						len2_e1 = e1.length2()
						len2_e2 = e2.length2()
						if len2_e1 >= len2_e2:
							bad_edges.append(e1)
							
						else:
							bad_edges.append(e2)
		
		#Removing any bad (intersecting) edges from the graph
		for x in bad_edges:
			for y in self._edges:
				if x==(y):
					self._edges.remove(y)
					continue
				
