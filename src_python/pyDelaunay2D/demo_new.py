from pyDelaunay import Graph
from polygon import Point, Edge, Triangle

import random
import sys
import pygame
import numpy as np
import cython

import polygon as plg

graph = Graph()
random.seed(1)

print("Adding points...")
p = np.zeros((2))
for x in range(0,100):
	p[0]=random.randint(50,974)
	p[1]=random.randint(50,718)	
	print("p",p[0],p[1])  # for test
	point=plg.Point(p)
	print("point",point[0],point[1])  # for test
	#while graph.addPoint(Point(random.randint(50,974), random.randint(50,718))) is False:
	while graph.addPoint(point) is False:
		print("Couldn't add point")

print("Generating Delaunay Mesh...")
graph.generateDelaunayMesh()

#Displaying all points and edges
pygame.init()
screen = pygame.display.set_mode([1024,768])
screen.fill((0,0,0))

for p in graph._points:
	pygame.draw.circle(screen, (255,255,255), p.pos(), 3)
	
for e in graph._edges:
	pygame.draw.line(screen, (0,255,0), e._a.pos(), e._b.pos())

pygame.display.update()	

while True:
	events = pygame.event.get()
	for e in events:
		if e.type == pygame.KEYDOWN:
			sys.exit()