from pyDelaunay import Graph, Point, Edge, Triangle
import random
import sys
import pygame
import pyqcstrc.qnnum.qnnum as qnn   # for qnnumber
import pyqcstrc.qnmath.qnmath as qmt # for dot product
import pyqcstrc.qnnDelaunay.qnnDelaynay as qnnD


graph = Graph()
random.seed(1)

print("Adding points...")
for x in range(0,100):
	while graph.addPoint(Point(random.randint(50,974), random.randint(50,718))) is False:
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