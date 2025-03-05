import sys, os, math
import numpy as np
import cython

import qnnum as qnn   # for qnnumber
import qnvec as qnv
import qnmath as qmt # for dot product
import qnndarray as qna
import cython

from typing import Self


#Basic Point class
class Point(qna.QnNdarray):
    def __new__(cls, n:np.int64, N:np.int64):
        global shape
        shape=(n)
        return super().__new__(cls,shape,N)
    
    def __init__(self, n:np.int64, N:np.int64): # x and y coordinates of a point
        qn0=qnn.Qnnum(n,N)
        self[0] = qn0
        self[1] = qn0
    
    #Position of the point
    def pos(self):
        return self  #[self[0], self[1]]
            
    #Determines if two points are equivalent
    def isEqual(self, other_point:Self):
        if(self[0] == other_point[0] and self[1] == other_point[1]): return True
        else: return False
    
    #Convert the point into a string (for debugging purposes)
    def pointToStr(self):
        return str(self)

    def anyp(n:np.int64, N:np.int64,vec:qnn.Qnnum)->qnv.Qnvec:
        qnvt=qnv.Qnvec(n,N)
        for i in range(n):
            qnvt[i]=vec[i]
            return qnvt

#Basic Edge class
class Edge(qna.QnNdarray):
    def __init__(self, a:qnv.Qnvec, b:qnv.Qnvec): # two points
        if a is not b:
            self[0] = a
            self[1] = b
    
    #Tests if two edges are equivalent to each other
    def isEqual(self, other_edge:Self):
        if (self[0].isEqual(other_edge[0]) or self[1].isEqual(other_edge[0])) and \
        (self[0].isEqual(other_edge[1]) or self[1].isEqual(other_edge[1])):
            return True
        elif self == other_edge:
            return True
        else:
            return False
    
    #Converts an edge to a string (for debugging purposes)
    def edgeToStr(self):
        return str([self[0], self[1]])
    
    #Calculate squared length of an edge
    def length(self):
        return math.sqrt( math.pow(self[1][0] - self[0][0],2) + \
            qmt.pow(self[1][1] - self[0][1],2))
        
    def length2(self):
        return ( qmt.pow(self[1][0] - self[0][0],2) + \
            qmt.pow(self[1][1] - self[0][1],2))
    
    #Determine if two edges intersect
    def edgeIntersection(self, other_edge:Self):

        if self.isEqual(other_edge):
            return False
        else:
            try:
                x1 = self[0][0]
                x2 = self[1][0]
                x3 = other_edge[0][0]
                x4 = other_edge[1][0]
                y1 = self[0][1]
                y2 = self[1][1]
                y3 = other_edge[0][1]
                y4 = other_edge[1][1]
                t = (((x1 - x3)*(y3 - y4)) - ((y1 - y3)*(x3 - x4))) / (((x1 - x2)*(y3 - y4)) - ((y1 - y2)*(x3 - x4)))
                u = (((x2 - x1)*(y1 - y3)) - ((y2 - y1)*(x1 - x3))) / (((x1 - x2)*(y3 - y4)) - ((y1 - y2)*(x3 - x4)))
                
                #If 0 <= t <= 1 or 0 <= u <= 1, then an intersection occurs. 
                if (t >= 0 and t <= 1) and (u >= 0 and u <= 1):
                    int_x = int(x1 + t*(x2 - x1))
                    int_y = int(y1 + t*(y2 - y1))
                    int_point = Point(int_x, int_y)
                    
                    #If the intersection point is one of the edge points, then
                    # an intersection is not considered to have occurred (i.e.,
                    # these are edges connected at the same point)
                    if self[0].isEqual(int_point) or self[1].isEqual(int_point) or \
                     other_edge[0].isEqual(int_point) or other_edge[1].isEqual(int_point):
                        return False
                    
                    #If there is no point, these edges intersect
                    else:
                        return True
                    
                else:
                    return False
            except:
                #A divide-by-zero error is interpreted as the edges not intersecting
                return False

#Basic Triangle class
class Triangle(qna.QnNdarray):
    
    #Cannot create a triangle if any two points are the same
    def __init__(self, a:qnv.Qnvec, b:qnv,Qnvec, c:qnv.Qnvec):  # three points
        if a is not b and a is not c:
            self[0] = a
        if b is not a and b is not c:
            self[1] = b
        if c is not a and c is not b:
            self[2] = c
    
    #Test if any two triangles are equal (defined as sharing all three points)
    def isEqual(self, other_tri:Self):
        if (self[0] is other_tri[0] or self[0] is other_tri[1] or \
        self[0] is other_tri[2]) and (self[1] is other_tri[0] or \
        self[1] is other_tri[1] or self[1] is other_tri[2]) and \
        (self[2] is other_tri[0] or self[2] is other_tri[1] or \
        self[2] is other_tri[2]): return True
        else: return False
    
    #Prints the triangle in a neat format (for debugging purposes)
    def printTriangle(self):
        print("A: " + self[0].pointToStr() + " B: " + self[1].pointToStr() + " C: " + self[2].pointToStr())

#Graph class
class Graph():
    def __init__(self):
        
        #This will be a list of point objects as defined above
        self._points = []
        
        #This will be a list of triangle objects as defined above
        self._triangles = []
        
        #This is a list of edges as defined above
        self._edges = []
        
        #Point boundaries for sorting purposes
        self._point_min_x = 0
        self._point_max_x = 0
        
    def addPoint(self, point:Point):
    
        #Check to see if an equivalent point exists
        for x in self._points:
            if x.isEqual(point): 
                return False
        
        #If the point has an X value lower than any other point
        if self._point_min_x > point[0] or self._point_min_x == 0:
            self._points.insert(0,point)
            self._point_min_x = point[0]
            return True
        
        #If the point has an X value higher than any other point
        elif self._point_max_x < point[0]:
            self._points.append(point)
            self._point_max_x = point[0]
            return True
        
        #If the X value is somewhere in the middle
        else:
            same_x = []
            for x in self._points:
                if x[0] == point[0]:
                    same_x.append(x)
            
            #If no point has the same X value as the new point,
            # find the first point that has a greater X value and insert the new point before it
            if len(same_x) == 0:
                first_greater = 0
                for x in self._points:
                    if x[0] > point[0]:
                        first_greater = self._points.index(x)
                        break
                self._points.insert(first_greater, point)
                return True
            
            #If there's only one point in the graph with the same X value,
            # compare the Y values to find which order they go in
            elif len(same_x) == 1:
                index = self._points.index(same_x[0])
                if same_x[0][1] > point[1]:
                    self._points.insert(index - 1, point)
                    return True
                else:
                    self._points.insert(index + 1, point)
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
                    self._points.insert(first_greater_y, point)
                    return True
                else:
                    self._points.insert(self._points.index(same_x[len(same_x) - 1]), point)
                    return True
        
    def addEdge(self, edge:Edge):
        
        #Check for an equivalent edge in the graph, add it if one doesn't exist
        for x in self._edges:
            if x.isEqual(edge):
                return False
        self._edges.append(edge)
        return True
        
    #Adds a triangle to the list of triangles and returns true 
    #if successful, checking if it is equal to any other triangle.
    # Returns false if an equivalent triangle exists
    def addTriangle(self, triangle:Triangle):
        
        #First check if an equivalent triangle already exists
        for x in self._triangles:
            if x.isEqual(triangle): return False
        
        #If not, we can add the triangle to the graph
        self._triangles.append(triangle)
        tri = [ triangle[0], triangle[1], triangle[2] ]
        return True
        
    #Tests if a given triangle is Delaunay (i.e.,
    # no other points lie within the circumcircle of the triangle)
    def triangleIsDelaunay(self, triangle:Triangle):
        tri = [ triangle[0], triangle[1], triangle[2] ]
        #cc = circumcircle(tri) # center and radius of circumcircle
        cc = circumcircle2(tri) # center and radius of circumcircle
        for x in self._points:
            #print(x)
            #If we get the divide-by-zero error, we assume the triangle is non-Delaunay
            if not (x.isEqual(triangle[0]) and x.isEqual(triangle[1]) and x.isEqual(triangle[2])):
                try:
                    if pointInCircle(x, cc):
                        return False
                except:
                    return False
        #self[2]ircles.append(cc)
        return True
    
    #Generates the complete Delaunay mesh by testing every possible triangle
    # for the Delaunay condition, then marking any edges that intersect, and
    # removing the longer of the intersecting edges
    def generateDelaunayMesh(self):
    
        #Create every possible triangle and test it for the Delaunay condition
        for p1 in self._points:
            for p2 in self._points:
                for p3 in self._points:
                    if not p1.isEqual(p2) and not p2.isEqual(p3) and not p3.isEqual(p1):
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
        bad_edges = []
        for e1 in self._edges:
            for e2 in self._edges:
                if not e1.isEqual(e2):
                    if e1.edgeIntersection(e2):
                        len_e1 = e1.length()
                        len_e2 = e2.length()
                        if len_e1 >= len_e2:
                            bad_edges.append(e1)
                            
                        else:
                            bad_edges.append(e2)
        
        #Removing any bad (intersecting) edges from the graph
        for x in bad_edges:
            for y in self._edges:
                if x.isEqual(y):
                    self._edges.remove(y)
                    continue
                
    
#***** this uses qnnumbers for x y coordinates ******
# an edge is represented by 2D vectors with shape (2,2)
# a triangle is represented by 2D vectors with shpae (3,2) 

#Function for determining the circumcircle of any three points
def circumcircle(tri:Triangle):
    n=tri[0][0].n
    N=tri[0][0].N
    center=qnv.zerov(n,N)
    try:
        D = ((tri[0][0]-tri[2][0])*(tri[1][1]-tri[2][1])-(tri[1][0]-tri[2][0])*(tri[0][1]-tri[2][1]))
        
        center[0] = (((tri[0][0]-tri[2][0])*(tri[0][0]+tri[2][0])+(tri[0][1]-tri[2][1])*(tri[0][1]+tri[2][1]))/ \
                 2*(tri[1][1]-tri[2][1])-((tri[1][0]-tri[2][0])*(tri[1][0]+tri[2][0])+(tri[1][1]-tri[2][1]) * \
                 (tri[1][1]+tri[2][1]))/2*(tri[0][1]-tri[2][1]))/D
        
        center[1] = (((tri[1][0]-tri[2][0])*(tri[1][0]+tri[2][0])+(tri[1][1]-tri[2][1])*(tri[1][1]+tri[2][1]))/ \
                 2*(tri[0][0]-tri[2][0])-((tri[0][0]-tri[2][0])*(tri[0][0]+tri[2][0])+(tri[0][1]-tri[2][1]) * \
                (tri[0][1]+tri[2][1]))/ 2*(tri[1][0]-tri[2][0]))/D
        
        #radius = qmt.sqrt ((tri[2][0] - center_x)**2 + (tri[2][1] - center_y)**2 )
        radius2 = ((tri[2][0] - center[0])**2 + (tri[2][1] - center[1])**2 )
        
        #return [[center_x, center_y], radius]
        return [center, radius2] # point and squared radius
    except:
        print("Divide By Zero error")
        print(tri)
 
def circumcircle2(tri:Triangle):            
    #tri[0]-tri[2] and tri[1]-tri[2] are edige vectors form tri[2]
    center=qnn.zeros((2))
    edg=qnv.zeros((3))
    edg[0]=tri[0]-tri[2]; edg[1]=tri[1]-tri[2]; edg[2]=tri[1]+tri[2]
    try:
        D = ((edg[0][0])*(edg[1][0])-(edg[1][0])*(edg[0][1])) # 2 times the area of tri 
        
        center[0] = ((edg[0][0]*edg[2][0]+edg[0][1]*edg[2][1])/ \
            2*edg[1][1]-(edg[1][0]*edg[2][0]+edg[0][1] * \
            edg[2][1])/2*edg[0][1])/D
        
        center[1] = ((edg[1][0]*edg[2][0]+edg[0][1]*edg[1][1])/ \
            2*edg[0][0]-(edg[0][0]*edg[2][0]+edg[0][1] * \
            edg[2][1])/2*edg[1][0])/D
        radius2=(center[0]**2+center[1]**2)
        #return [[center[0], center[1]], radius]
        return [center, radius2] # point and squared radius
    except:
        print("Divide By Zero error")
        
#Determine if any given point lies inside a circle
def pointInCircle(point:Point, circle:Point):
    #This is pretty simple; just find the distance between the point and the center.
    # If it's less than or equal to the radius, the point is inside the circle
    
    #d = qmt.sqrt( qmt.pow(point[0] - circle[0][0], 2) + qmt.pow(point[1] - circle[0][1],2) )
    d2 = ( qmt.pow(point[0] - circle[0][0], 2) + qmt.pow(point[1] - circle[0][1],2) )
    #if d < circle[1]:
    if d2 < circle[1]: # circle[0] circle[1] should be a point and squared radius
        return True
    else:
        return False
   
