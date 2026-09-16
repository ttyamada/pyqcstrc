# qnnumber version of vesta
# qnnumber should be transformed to float by qnn.qn2frt 

import timeit
import os
import sys
import numpy as np
import cython

import crsys as crs
import qnnum as qnn
import qnndarray as qna
import qnvec as qnv
import qnmath as qmt
import utils as utl
import numeric as num
import intsct as ints
import prjop as prj
import math1 as mth

def write_off(obj,path='.',basename='tmp',select='triangle',verbose=0):
    """
    Export occupation domains in XYZ format.
    
    Args:
        #obj (numpy.ndarray): the occupation domain
        #    The shape is (num,3,6,3), where num=numbre_of_triangles.
        obj (qna.qnndarray): the occupation domain
            The shape is (num,3,n), where num=numbre_of_triangles n=5 or 6 for dihed or icos.
        path (str): Path of the output XYZ file
        basename (str): Basename of the output XYZ file
        select (str)
            'triangle'   : set of triangles (default)
            'edge'       : set of edges
            'vertex'      : set of vertices
            (default, select = 'triangle')
    
    Returns:
        int: 0 (succeed), 1 (fail)
    """
    
    def generator_off_dim4_triangle(obj,path,filename):
        """
        Generate object (set of triangles) object in XYZ format.
    
        Args:
            #obj (numpy.ndarray): the occupation domain
            #    The shape is (num,3,6,3), where num=numbre_of_triangle. (original)
            obj (qna.qnndarray): the occupation domain
                The shape is (num,3,n), where num=numbre_of_triangle, n=5 or 5 for dihed or icos.
            filename (str): filename of the output XYZ file
        
        Returns:
            int: 0 (succeed), 1 (fail)
        
        """
        f=open('%s/%s.xyz'%(path,filename),'w', encoding="utf-8", errors="ignore")
        f.write('%d\n'%(len(obj)*3))
        f.write('%s\n'%(filename))
        i1=0
        n=crs.n
        isys=crs.isys
        if isys>2:
            ni=2
        else:
            ni=3
        scly=crs.scly
        for i1,triangle in enumerate(obj):  # i1-th triangle
            for i2,vt in enumerate(triangle): # i2-th vertex
                #qnv.printqnv("vt",vt)  # for test
                #vi=prj.projection3(vt)  # projection into internal space
                vi=vt
                #qnv.printqnv("vi",vi)  # for test
                vif=qnv.qnv2flt(vi)

                if n==5:
                    f.write('Xx %8.6f %8.6f %8.6f'%(vif[0],vif[1],0.0))
                else:
                    f.write('Xx %8.6f %8.6f %8.6f'%(vif[0],vif[1],vif[2]))
                f.write(' # %s-th triangle %s-th vertex '%(i1,i2))
                f.write(' #')
                for i in range(ni-1):
                    f.write('  %d %d %d'%(vt[i].n[0],vt[i].n[1],vt[i].n[2]))
                f.write('  %d %d %d\n'%(vt[ni-1].n[0],vt[ni-1].n[1],vt[ni-1].n[2]))
 
        v=utl.obj_area_nd(obj)
        f.write('volume = %d %d %d (%8.6f)\n'%(v.n[0],v.n[1],v.n[2],qnn.qn2flt(v)))
        for i1,triangle in enumerate(obj):
            v=utl.triangle_area_nd(triangle)
            f.write('%3d-the triangle, %d %d %d (%8.6f)\n'\
                    %(i1,v.n[0],v.n[1],v.n[2],qnn.qn2flt(v)))
        f.closed
        return 0
    
    def generator_off_dim4_edge(obj,path,filename):
        """
        Generate object (set of edges) object in off format.
    
        Args:
            obj (numpy.ndarray): the occupation domain
                #The shape is (num,2,6,3), where num=numbre_of_edges. (original)
                The shape is (num,2,n), where num=numbre_of_edges
            filename (str): filename of the output XYZ file
        
        Returns:
            int: 0 (succeed), 1 (fail)
        
        """
        f=open('%s/%s.xyz'%(path,filename),'w', encoding="utf-8", errors="ignore")
        f.write('%d\n'%(len(obj)*2))
        f.write('%s\n'%(filename))
        n=crs.n
        for i1,edge in enumerate(obj):
            for i2,vt in enumerate(edge):
                v=prj.projection3(vt)
                vf=qnv.qnv2flt(v)
                f.write('Xx %8.6f %8.6f %8.6f'%(vf[0],vf[1],vf[2]))
                f.write(' # %s-th triang %s-th vertex '%(i1,i2))
                for i in range(n-1):
                    f.write(' %d %d %d'%(vt[0].n[0],vt[0].n[1],vt[0].n[2]))
                f.write(' %d %d %d \n'%(vt[n-1].n[0],vt[n-1].n[1],vt[n-1].n[2]))
                
        f.closed
        return 0
    
    def generator_off_dim4_vertex(obj,path,filename):
        """
        Generate object (set of vertexs) object in XYZ format.
    
        Args:
            #obj (numpy.ndarray): the occupation domain
            obj (qna.qnndarray): the occupation domain
                #The shape is (num,3,6,3), where num=numbre_of_triangles  (original)
                The shape is (num,3,n), where num=numbre_of_triangles (n=2)
            filename (str): filename of the output off file
        
        Returns:
            int: 0 (succeed), 1 (fail)
        
        """
        f=open('%s/%s.off'%(path,filename),'w', encoding="utf-8", errors="ignore")
        f.write('%d\n'%(len(obj)))
        f.write('%s\n'%(filename))
        counter=0
        n=crs.n
        for tri in range(len(obj)):
            for point in tri:
                v=prj.projection3(point) # internal space components
                vf=qnv.qnv2flt(v) # qnv to float vector for off
                f.write('Xx %8.6f %8.6f %8.6f'%(vf[0],vf[1],vf[2]))
                f.write(counter)
                #f.write(' #')
                #for i in range(n-1):
                #    f.write(' %d %d %d'%(point[i].n[0],point[i].n[1],point[i].n[2]))
                #f.write(' %d %d %d \n'%(point[n-1].n[0],point[n-1].n[1],point[n-1].n[2]))
                counter+=1
            # add edge indices of each edge 2 3 5 etc
        f.closed
        return 0
    
    def generator_off_dim3_vertex(obj,path,filename):
        """
        Generate object (set of vertexs) object in off format.
    
        Args:
            #obj (numpy.ndarray): the occupation domain
            obj (qna.qnndarray)): the occupation domain
                #The shape is (num,6,3), where num=numbre_of_vertices. (original)
                The shape is (num,3), where num=numbre_of_vertices. (including z)
            filename (str): filename of the output off file
        
        Returns:
            int: 0 (succeed), 1 (fail)
        
        """
        f=open('%s/%s.xyz'%(path,filename),'w', encoding="utf-8", errors="ignore")
        f.write('%d\n'%(len(obj)))
        f.write('%s\n'%(filename))
        n=crs.n #5 or 6
        for i1,point in enumerate(obj):
            #print("i1",i1)  # for test
            #qnv.printqnv("point",point) # for test
            v=prj.projection3(point)  # projection onto the internal space
            vf=qnv.qnv2flt(v)
            f.write('Xx %8.6f %8.6f %8.6f'%(vt[0],vt[1],vt[2]))
            #f.write(i1)
            #for i in range(n-1):
            #    f.write(' # %d %d %d '%(point[i].n[0],point[i][1],point[i].n[2]))
            #f.write(' # %d %d %d \n'%(point[n-1].n[0],point[n-1].n[1],point[n-1].n[2]))

        # add here the edge indices of each edge 2 5 6 etc. in each line for off
        f.closed
        return 0
    
    shape=obj.shape
    #print("shape in write_xyz",shape)  # for test
    ndim=len(shape)
 
    #if np.all(objt==None):
    #    print('empty objt')
    #    return 
    #elif ndim<3 or ndim>4:
    # ndim=1, 2 or 3 for vertex, triangle/tetrahedron (edge) or triangles/tetrahedra 
    if ndim<1 or ndim>4:
        print('object has an incorrect shape!')
        return 
    elif ndim==1:
        if select=='vertex':
            generator_xyz_dim3_vertex(obj,path,basename)
            if verbose>0:
                print('    written in %s/%s.xyz'%(path,basename))
            return 0
        else:
            return 
    else:
        file_name='%s/%s.off'%(path,basename)
        if select=='triangle':
            generator_off_dim4_triangle(obj,path,basename)
            if verbose>0:
                print('    written in %s/%s.off'%(path,basename))
            return 0
        elif select=='edge':
            generator_off_dim4_edge(obj,path,basename)
            if verbose>0:
                print('    written in %s/%s.off'%(path,basename))
            return 0
        elif select=='vertex':
            generator_xyz_dim4_vertex(obj,path,basename)
            if verbose>0:
                print('    written in %s/%s.off'%(path,basename))
            return 0
        else:
            if verbose>0:
                print('    error')
            return 
