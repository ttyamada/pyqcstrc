#!/usr/bin/env python3
# -*- coding: ascii -*-
"""
Simple delaunay2D demo with mathplotlib
Written by Jose M. Espadero < http://github.com/jmespadero/pyDelaunay2D >
"""
import numpy as np
from delaunay2D import Delaunay2D
import matplotlib.pyplot as plt
import matplotlib.tri
import matplotlib.collections

if __name__ == '__main__':

    ###########################################################
    # Generate 'numSeeds' random seeds in a square of size 'radius'
    numSeeds = 24
    radius = 100
    seeds = radius * np.random.random((numSeeds, 2))

    print("seeds:\n", seeds)
    print("BBox Min:", np.amin(seeds, axis=0),
          "Bbox Max: ", np.amax(seeds, axis=0))

    """
    Compute our Delaunay triangulation of seeds.
    """
    # Compute center and radius of input points 
    center = np.mean(seeds, axis=0)
    radius = np.max(np.linalg.norm((seeds - center), axis=1))

    # Sort seeds by X-coordinate    
    #perm = sorted(range(len(seeds)), key=lambda x: seeds[x][0])
    #seeds=[seeds[i] for i in perm]

    # Starts from a new Delaunay2D frame
    dt2 = Delaunay2D(center, 50 * radius)    
    for i,s in enumerate(seeds):
        print("Inserting seed", i, s)
        dt2.addPoint(s)
        if i > 1:
            fig, ax = plt.subplots()
            ax.margins(0.1)
            ax.set_aspect('equal')            
            plt.axis([center[0]-radius, center[0]+radius,center[1]-radius, center[1]+radius])
            for ii, v in enumerate(seeds[0:i+1]):
                plt.annotate(ii, xy=v)              # Plot all seeds
            for t in dt2.exportTriangles():
                polygon = [seeds[ii] for ii in t]     # Build polygon for each region
                plt.fill(*zip(*polygon), fill=False, color="b")  # Plot filled polygon

            print("Press q to show next step")
            plt.show()
            #plt.savefig(f'output-{i:02d}.png', dpi=150, bbox_inches='tight')

    
