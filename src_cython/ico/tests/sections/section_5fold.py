#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pyqcstrc.ico.math1 as math1

TAU=(1+np.sqrt(5))/2.0

def projection(n1,n2,n3,n4,n5,n6):
   """Direct lattice vector
   """
   n=np.array([n1,n2,n3,n4,n5,n6])
   n.shape=(1,6)
   c1=1.0/np.sqrt(2.0+TAU)
   c2=1.0/np.sqrt(2.0+TAU)
   
   # lattice spaceing with a_{par}
   m1=c1*np.array([1.0,  TAU,  0.0,\
               TAU,  0.0,  1.0,\
               TAU,  0.0, -1.0,\
               0.0,  1.0, -TAU,\
               -1.0,  TAU,  0.0,\
               0.0,  1.0,  TAU])
   m1.shape=(6,3)      
   
   # lattice spaceing with a_{perp}     
   m2=c2*np.array([ TAU, -1.0,  0.0,\
               -1.0,  0.0,  TAU,\
               -1.0,  0.0, -TAU,\
                0.0,  TAU,  1.0,\
               -TAU, -1.0,  0.0,\
                0.0,  TAU, -1.0])
   m2.shape=(6,3)
   
   # Join a sequence of arrays       
   m1=np.concatenate((m1,m2), axis=1)
   
   m2=m1[:,[0,1,2]]
   val=np.dot(n,m2) # projected n vector onto Epar
   val1=val[0,0] # x in Epar
   val2=val[0,1] # y in Epar
   val3=val[0,2] # z in Epar
   val7=np.sqrt(val1**2+val2**2+val3**2)

   m3=m1[:,[3,4,5]]
   val=np.dot(n,m3) # projected n vector onto Eperp
   val4=val[0,0] # x in Eperp
   val5=val[0,1] # y in Eperp
   val6=val[0,2] # z in Eperp
   val8=np.sqrt(val4**2+val5**2+val6**2)

   return val1,val2,val3,val4,val5,val6,val7,val8

def projection_perp(n1,n2,n3,n4,n5,n6):
   # This returns 6D indeces of a projection of 6D vector (n1,n2,n3,n4,n5,n6) onto Eperp
   # Direct lattice vector
   const=1/(2.0+TAU)
   m1 =((TAU+2)*n1 -     TAU*n2 -     TAU*n3 -     TAU*n4 -     TAU*n5 -     TAU*n6)/2.0*const
   m2 = ( - TAU*n1 + (TAU+2)*n2 -     TAU*n3 +     TAU*n4 +     TAU*n5 -     TAU*n6)/2.0*const
   m3 = ( - TAU*n1 -     TAU*n2 + (TAU+2)*n3 -     TAU*n4 +     TAU*n5 +     TAU*n6)/2.0*const
   m4 = ( - TAU*n1 +     TAU*n2 -     TAU*n3 + (TAU+2)*n4 -     TAU*n5 +     TAU*n6)/2.0*const
   m5 = ( - TAU*n1 +     TAU*n2 +     TAU*n3 -     TAU*n4 + (TAU+2)*n5 -     TAU*n6)/2.0*const
   m6 = ( - TAU*n1 -     TAU*n2 +     TAU*n3 +     TAU*n4 -     TAU*n5 + (TAU+2)*n6)/2.0*const
   return m1,m2,m3,m4,m5,m6

"""    
def projection_perp(n1,n2,n3,n4,n5,n6):
    #Direct lattice vector
    #This returns 6D indeces of a projection of 6D vector (n1,n2,n3,n4,n5,n6)
    #
    n=np.array([n1,n2,n3,n4,n5,n6])
    n.shape=(1,6)
    #c1=apar/np.sqrt(2.0+TAU)
    #c2=aperp/np.sqrt(2.0+TAU)
    c1=1.0/np.sqrt(2.0+TAU)
    c2=1.0/np.sqrt(2.0+TAU)
    
    # lattice spaceing with a_{par}
    m1=c1*np.array([1.0,  TAU,  0.0,\
                    TAU,  0.0,  1.0,\
                    TAU,  0.0, -1.0,\
                    0.0,  1.0, -TAU,\
                   -1.0,  TAU,  0.0,\
                    0.0,  1.0,  TAU])
    m1.shape=(6,3)        
    
    # lattice spaceing with a_{perp}      
    m2=c2*np.array([ TAU, -1.0,  0.0,\
                    -1.0,  0.0,  TAU,\
                    -1.0,  0.0, -TAU,\
                     0.0,  TAU,  1.0,\
                    -TAU, -1.0,  0.0,\
                     0.0,  TAU, -1.0])
    m2.shape=(6,3)
    
    # Join a sequence of arrays         
    m1=np.concatenate((m1,m2),axis=1)
    
    # Inverse matrix of m1
    m2=np.linalg.inv(m1)
    
    # Projection matrix onto perp space    
    m3=np.array([0., 0., 0., 0., 0., 0.,\
                0., 0., 0., 0., 0., 0.,\
                0., 0., 0., 0., 0., 0.,\
                0., 0., 0., 1., 0., 0.,\
                0., 0., 0., 0., 1., 0.,\
                0., 0., 0., 0., 0., 1.])
    m3.shape=(6,6)

    tmp=np.dot(n,m1)
    tmp=np.dot(tmp,m3)
    val=np.dot(tmp,m2)
    #print val
    
    b=val[0:]
    c=b[0]
    
    return c[0],c[1],c[2],c[3],c[4],c[5]
"""

if __name__ == '__main__':

   nmax=5
   plot_type='5fold'
   #plot_type='3fold'
   #plot_type='2fold_v'
   #plot_type='2fold_c'

   if plot_type=='5fold':
      #### Fivefold section ###
      #horizontal
      h1,h2,h3,h4,h5,h6=0,1,1,1,1,1
      # vertical
      v1,v2,v3,v4,v5,v6=1,0,0,0,0,0
      #
      #tmp=np.array([TAU**(2),0,0,0,0,0])/2.0*TAU**(-2)
      tmp=np.array([0.5,0,0,0,0,0])
      #tmp=projection_perp(tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],tmp[5])
      #
      # length of strt along a direction parallel to fivefold axis
      #length=tmp[7]
      print 'length of strt along a direction parallel to fivefold axis'
      #print '%8.6f'%(length)

   elif plot_type=='3fold':
      #### threefold section ###
      #horizontal
      h1,h2,h3,h4,h5,h6=0,1,0,-1,0,1
      # vertical
      v1,v2,v3,v4,v5,v6=1,0,-1,0,-1,0
      #
      tmp=np.array([1,-1,-1, 1,-1,-1])/2.0*TAU**(-2)
      #tmp=projection(tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],tmp[5])
      #
      # length of strt along a direction parallel to threefold axis
      #length=tmp[7]
      print 'length of strt along a direction parallel to threefold axis'
      #print '%8.6f'%(length)
   
   elif plot_type=='2fold_v':
      #### twofold section ###
      #horizontal
      h1,h2,h3,h4,h5,h6=0,1,1,0,0,0
      # vertical
      v1,v2,v3,v4,v5,v6=1,0,0,0,-1,0
      #
      tmp=np.array([1,-1,-1,0,-1,0])/2.0*TAU**(-2)
      #tmp=projection(tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],tmp[5])
      #
      # length of strt along a direction parallel to twofold axis
      #length=tmp[7]
      print 'length of strt along a direction parallel to twofold axis'
      #print '%8.6f'%(length)
   
   elif plot_type=='2fold_c':
      #### twofold section ###
      #horizontal
      h1,h2,h3,h4,h5,h6=0,1,1,0,0,0
      # vertical
      v1,v2,v3,v4,v5,v6=1,0,0,0,-1,0
      #
      tmp=np.array([1,-1,-1,0,-1,0])/2.0*TAU**(-2)
      #tmp=projection(tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],tmp[5])
      #
      # length of strt along a direction parallel to twofold axis
      #length=tmp[7]
      print 'length of strt along a direction parallel to twofold axis'
      #print '%8.6f'%(length)
   
   #horizontal
   h0=projection(h1,h2,h3,h4,h5,h6)
   hpar,hperp=h0[6],-h0[7]
   # vertical
   v0=projection(v1,v2,v3,v4,v5,v6)
   vpar,vperp=v0[6],v0[7]
   # length of strt on the section
   tmp=projection(tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],tmp[5])
   length=tmp[7]
   print '%8.6f'%(length)
   
   # Initialization
   xv,yv,sizev=[],[],[]    # vertcies
   xc,yc,sizec=[],[],[]    # centres
   xe,ye,sizee=[],[],[]    # edge centres
   xc1,yc1,sizec1=[],[],[]    # centres 1
   xe1,ye1,sizee1=[],[],[]    # edge centres 1
   xv1,yv1,sizev1=[],[],[]    # vertcies 1
   
   # cluster center  5f
   pos_6d=[0,0,0,0,0,0]
   pos_i=[0,0,0,0,0,0]
   tmp=projection_perp(pos_i[0],pos_i[1],pos_i[2],pos_i[3],pos_i[4],pos_i[5])   
   p=projection(pos_6d[0]+tmp[0],pos_6d[1]+tmp[1],pos_6d[2]+tmp[2],pos_6d[3]+tmp[3],pos_6d[4]+tmp[4],pos_6d[5]+tmp[5])   
   # position of strt
   pos_c0=p[7]

   # icosahedron  5f
   pos_6d=[0,0,0,0,0,0]
   pos_i=[1,0,0,0,0,0]
   tmp=projection_perp(pos_i[0],pos_i[1],pos_i[2],pos_i[3],pos_i[4],pos_i[5])   
   p=projection(pos_6d[0]+tmp[0],pos_6d[1]+tmp[1],pos_6d[2]+tmp[2],pos_6d[3]+tmp[3],pos_6d[4]+tmp[4],pos_6d[5]+tmp[5])   
   # position of strt
   pos_c1=p[7]
   print('pos_c1 = %8.6f'%(pos_c1))
   
   # RTH  5f
   pos_6d=[0,0,0,0,0,0]
   #pos_i=[-0.5,-0.5,-0.5,-0.5,-0.5,-0.5]
   pos_i=[0.5,0.5,0.5,0.5,0.5,0.5]
   tmp=projection_perp(pos_i[0],pos_i[1],pos_i[2],pos_i[3],pos_i[4],pos_i[5])
   p=projection(pos_6d[0]+tmp[0],pos_6d[1]+tmp[1],pos_6d[2]+tmp[2],pos_6d[3]+tmp[3],pos_6d[4]+tmp[4],pos_6d[5]+tmp[5])   
   # position of strt
   pos_v1=p[7]
   print('pos_v1 = %8.6f'%(pos_v1))

   # EC
   pos_6d=[0.5,0,0,0,0,0]
   pos_i=[0,0,0,0,0,0]
   tmp=projection_perp(pos_i[0],pos_i[1],pos_i[2],pos_i[3],pos_i[4],pos_i[5])
   p=projection(pos_6d[0]+tmp[0],pos_6d[1]+tmp[1],pos_6d[2]+tmp[2],pos_6d[3]+tmp[3],pos_6d[4]+tmp[4],pos_6d[5]+tmp[5])   
   length2=p[7]
   # position of strt
   pos_e=p[7]
   print('pos_e = %8.6f'%(pos_e))
   
   #print 0.5*(hpar+vpar)
   #print 0.5*(hperp+vperp)
   
   fig,ax=plt.subplots()
   for n1 in range(-nmax,nmax+1):
      for n2 in range(-nmax,nmax+1):
         #valx=n1*hlen
         #valy=n2*vlen
         valx=n1*hpar+n2*vpar
         valy=n1*hperp+n2*vperp
         #
         ############
         # vertcies
         ############
         xv.append(valx)
         yv.append(valy)
         sizev.append(1.0)
         #
         ax.plot([valx,valx],[valy+pos_v1+length,valy+pos_v1-length],lw=1.0,c='b',alpha=0.75)
         ax.plot([valx,valx],[valy-pos_v1+length,valy-pos_v1-length],lw=1.0,c='b',alpha=0.75)
         xv1.append(valx)
         xv1.append(valx)
         yv1.append(valy+pos_v1)
         yv1.append(valy-pos_v1)
         sizev1.append(2.0)
         sizev1.append(2.0)
         #
         ############
         # centres
         ############
         calx=valx+0.5*(hpar+vpar)
         caly=valy+0.5*(hperp+vperp)
         xc.append(calx)
         yc.append(caly)
         sizec.append(1.0)
         #
         # cluster center
         ax.plot([calx,calx],[caly+pos_c0+length,caly+pos_c0-length],lw=1.0,c='#cfcfcf',alpha=0.75) # grey
         #
         ax.plot([calx,calx],[caly+pos_c1+length,caly+pos_c1-length],lw=1.0,c='r',alpha=0.75)
         ax.plot([calx,calx],[caly-pos_c1+length,caly-pos_c1-length],lw=1.0,c='r',alpha=0.75)
         xc1.append(calx)
         xc1.append(calx)
         yc1.append(caly+pos_c1)
         yc1.append(caly-pos_c1)
         sizec1.append(2.0)
         sizec1.append(2.0)
         #
         #print 'Xx %8.6f %8.6f 0.0'%(valx,valy)
         
         
         ############
         # edge centre
         ############
         calx=valx+0.5*(vpar)
         caly=valy+0.5*(vperp)

         #
         ax.plot([calx,calx],[caly+length2,caly],lw=1.0,c='b',alpha=0.75)
         ax.plot([calx,calx],[caly-length2,caly],lw=1.0,c='b',alpha=0.75)
         xe1.append(calx)
         xe1.append(calx)
         ye1.append(caly)
         ye1.append(caly)
         sizee1.append(2.0)
         sizee1.append(2.0)

   # vertcies
   plt.scatter(xv,yv,s=sizev,color='k')
   plt.scatter(xv1,yv1,s=sizev1,color='b')
   # centres
   plt.scatter(xc,yc,s=sizec,color='k')
   plt.scatter(xc1,yc1,s=sizec1,color='r')
   # edge centres
   plt.scatter(xe1,ye1,s=sizee1,color='b')
   ax.set_xlim(-5,5)
   ax.set_ylim(-5,5)
   ax.set_aspect('equal')
   #plt.show()
   plt.savefig('section_5fold.png', format="png", dpi=300)
   
   """
   for i in range(len(symop)):
      op=symop[i]
      v1=np.dot(op,v)
    
      n1,n2,n3,n4=v1[0],v1[1],v1[2],v1[3]
      v2=projection_vec(n1,n2,n3,n4)
    
      # par
      print 'Xx %8.6f %8.6f 0.0 # %2d %2d %2d %2d %8.6f %8.6f'%(v2[0],v2[1],n1,n2,n3,n4,v2[4],v2[5])
    
      # perp
      print 'Xx %8.6f %8.6f 0.0 # %2d %2d %2d %2d %8.6f %8.6f'%(v2[2],v2[3],n1,n2,n3,n4,v2[4],v2[5])
   """
