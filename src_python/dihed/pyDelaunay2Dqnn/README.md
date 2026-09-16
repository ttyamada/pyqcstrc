PyDelaunay2D
==============

A Simple Delaunay triangulation and Voronoi diagram constructor in 2D. Written by [Jose M. Espadero](https://github.com/jmespadero/pyDelaunay2D)

![](output-delaunay2D.png)

Just pretend to be a simple and didactic implementation of the 
[Bowyer-Watson algorithm](https://en.wikipedia.org/wiki/Bowyer-Watson_algorithm)
to compute the [Delaunay triangulation](https://en.wikipedia.org/wiki/Delaunay_triangulation)
and the [Voronoi diagram](https://en.wikipedia.org/wiki/Voronoi_diagram) of a set o 2D points.

It is written in pure python + [numpy](http://www.numpy.org/) (tested with 
python2.7 and python3). A test example is provided showing how to call and 
plot the results using matplotlib.

It support the [robust inCircle2D predicate](https://www.cs.cmu.edu/~quake/robust.html)
from Jonathan Richard Shewchuk, but it is disabled by default due to perfomance
penalties, so do not expect to work on degenerate set of points.
If you really need to compute triangulation on big or degenerate set of points, 
try [scipy.spatial.Delaunay](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html) 
instead.

## How can I use it in my projects?

Here is a minimal example of building a triangulation and dump the result to console.

```python 
import numpy as np
from delaunay2D import Delaunay2D

# Create a random set of points
seeds = np.random.random((10, 2))

# Create Delaunay Triangulation and insert points one by one
dt = Delaunay2D()
for s in seeds:
    dt.addPoint(s)

# Dump points and triangles to console
print("Input points:\n", seeds)
print ("Delaunay triangles:\n", dt.exportTriangles())

```

There is some improvements we can do to that example. For example, it is recommended to compute center and radius of the circle that contains our set of points, to build a initial frame taylored for our data. 

```python 
import numpy as np
from delaunay2D import Delaunay2D

# Create a random set of points
seeds = np.random.random((10, 2))

# Compute center and radius of our input set of points 
center = np.mean(seeds, axis=0)
radius = np.max(np.linalg.norm((seeds - center), axis=1))

# Build a taylored frame for our input points
dt = Delaunay2D(center, 2 * radius)

# Sometimes, it is useful to insert seeds sorted by X-coordinate    
perm = sorted(range(len(seeds)), key=lambda i: seeds[i][0])
for i in perm:
    dt.addPoint(seeds[i])

# Dump points and triangles to console
print("Input points:\n", seeds)
print ("Delaunay triangles:\n", dt.exportTriangles())

```

## Is it fast?

No, it isn't fast. This code has been written to remain simple, easy to read by 
beginners and with minimal dependencies needed. There is a section in ```addPoint()``` 
method that performs specially bad if you have a big set of input points: 

```python
    # Search the triangle(s) whose circumcircle contains p 
    for T in self.triangles:
        if self.inCircle(T, p):
            bad_triangles.append(T)
```

Here, we absolutely should avoid iterating over the complete list of triangles. Best way is to use a 
structure that allows a spatial search (as a [QuadTree](https://en.wikipedia.org/wiki/Quadtree)). 
Then, continue the search over the neighbours of the initial search.

Despite that, it will compute DT of less than 1000 points in a reasonable time. If you really 
need to compute triangulation on huge or degenerate sets of points, try
[scipy.spatial.Delaunay](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html),
which is based on [Qhull library](http://www.qhull.org/)


## Why did you write it?

Mainly, to provide a didactic implementation of the algorithm. Also, because sometimes it is 
not possible/worth to import the complete scipy.spatial package (for example, when running a 
script inside of python interpreter included in [blender](https://www.blender.org/) )


## References:
* https://en.wikipedia.org/wiki/Bowyer-Watson_algorithm
* https://en.wikipedia.org/wiki/Delaunay_triangulation
* http://www.geom.uiuc.edu/~samuelp/del_project.html
* https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html
* https://www.cs.cmu.edu/~quake/robust.html
