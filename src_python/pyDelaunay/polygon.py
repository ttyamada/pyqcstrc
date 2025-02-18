import sys, os, math
import numpy as np
from typing import Self

#Basic Point class
class Point(np.ndarray):
    def __new__(cls, n:np.int64):
        global shape
        shape=(2)
        return super().__new__(cls,shape)
    
    def __init__(self, p:np.ndarray): # x and y coordinates of a point
        self[0] = p[0]  #self._x = x
        self[1] = p[1]  #self._y = y
    
    def __eq__(self,b:Self):
        return isEqual_p(self,b)
    
    def __ne__(self,b:Self):
        return not isEuqual_p(self,b)
    
    def __add__(self,b:Self):
        return add_p(self,b)
    
    def __sub__(self,b:Self):
        return sub_p(self,b)
    
    #Position of the point
    def pos(self):
        return self  #return [self._x, self._y]
            
    #Convert the point into a string (for debugging purposes)
    def pointToStr(self):
        return str(self.pos())
    
def zerops(shape) -> Point:  # Point array
    return np.zeros(shape,dtype=Point)
    
#Determines if two points are equivalent
def isEqual_p(self:Point, other_point:Point):
    #if(self._x == other_point._x and self._y == other_point._y): return True
    if(self[0] == other_point[0] and self[1] == other_point[1]): return True
    else: return False
    
def add_p(self:Point,b:Point):
    for i in range(2):
        self[i]=self[i]+b[i]
    return self
    
def sub_p(self:Point,b:Point):
    for i in range(2):
        self[i]=self[i]-b[i]
    return self

#Basic Edge class
class Edge(np.ndarray):
    def __new__(cls, n:np.int64):
        global shape
        shape=(2)
        return super().__new__(cls,shape)

    def __init__(self, a:Point, b:Point): # two points
        if a != b:
            self[0]=a  #self._a = a
            self[1]=b  #self._b = b
            
    def __eq__(self,b:Self):
        return isEqual_e(self,b)
    
    def __ne__(self,b:Self):
        return not isEqual_e(self,b)
    
    def __add__(self,b):
        return add_e(self,b)
    
    def __sub__(self,b):
        return sub_e(self,b)
          
    #Converts an edge to a string (for debugging purposes)
    def edgeToStr(self):
        #return str([self._a.pos(), self._b.pos()])
        return str([self[0], self[1]])
    
    #Calculate the squared length of an edge
    def length2(self):
        #return math.sqrt( math.pow(self._b.pos()[0] - self._a.pos()[0],2) + \
        #    math.pow(self._b.pos()[1] - self._a.pos()[1],2))
        return ( math.pow(self[1][0] - self[0][0],2) + \
            math.pow(self[1][1] - self[0][1],2))

    def insert(self,p,a):
        self=np.insert(self,p,a)
        return self
    
    #Determine if two edges intersect
    def edgeIntersection(self, other_edge):

        if self==(other_edge):
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
                    if self[0]==int_point or self[1]==int_point or \
                     other_edge[0]==int_point or other_edge[1]==int_point:
                        return False
                    
                    #If there is no point, these edges intersect
                    else:
                        return True
                    
                else:
                    return False
            except:
                #A divide-by-zero error is interpreted as the edges not intersecting
                return False
            
def zeroes(shape) -> Edge:  # Edge array
    return np.zeros(shape,dtype=Edge)

#Tests if two edges are equivalent to each other         
def isEqual_e(self, other_edge):
    #if (self._a==(other_edge._a) or self._b==(other_edge._a)) and \
    #(self._a==(other_edge._b) or self._b==(other_edge._b)):
    if (self[0]==other_edge[0] or self[1]==other_edge[0]) and \
    (self[0]==other_edge[1] or self[1]==other_edge[1]):
        return True
    elif self == other_edge:
        return True
    else:
        return False
        
def add_e(self,b):
        for i in range(2):
            self[i]=self[i]+b[i]
        return self
    
def sub_e(self,b):
    for i in range(2):
        self[i]=self[i]-b[i]
    return self
  

#Basic Triangle class
class Triangle(np.ndarray):
    
    def __new__(cls, n:np.int64):
        global shape
        shape=(3)
        return super().__new__(cls,shape)
    
    #Cannot create a triangle if any two points are the same
    def __init__(self, a:Point, b:Point, c:Point):  # three points
        if not a==b and not a==c:
            self[0] = a
        if b is not a and b is not c:
            self[1] = b
        if c is not a and c is not b:
            self[2] = c
            
    def __eq__(self,b):
        return isEuqual_t(self,b)
        
    #Prints the triangle in a neat format (for debugging purposes)
    def printTriangle(self):
        print("A: " + self[0].pointToStr() + " B: " + self[1].pointToStr() + " C: " + self[2].pointToStr())

    def insert(self,p,a:np.ndarray):
        np.insert(self,p,a)
        return self
    
def zerots(shape) -> Triangle:  # Triangle array
    return np.zeros(shape,dtype=Triangle)
#Test if any two triangles are equal (defined as sharing all three points)
def isEqual_t(self:Triangle, other_tri:Triangle):
    if (self[0] is other_tri[0] or self[0] is other_tri[1] or \
      self[0] is other_tri[2]) and (self[1] is other_tri[0] or \
      self[1] is other_tri[1] or self[1] is other_tri[2]) and \
      (self[2] is other_tri[0] or self[2] is other_tri[1] or \
      self[2] is other_tri[2]):
        return True
    else: 
        return False
