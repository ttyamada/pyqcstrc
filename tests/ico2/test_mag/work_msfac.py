#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>
import sys
import numpy as np
import math
import cmath
try:
    from pyqcstrc.ico2.strc import (strc,
                                    )
    import pyqcstrc.ico2.occupation_domain as od
    from pyqcstrc.ico2.numericalc import (projection_numerical,
                                         )
    import pyqcdiff.common.atom as atm
except ImportError:
    print('import error\n')
import matplotlib.pyplot as plt

# Predefined 6D coordinates in TAU-style
POS_V  = np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.int64)
POS_C  = np.array([[1,0,2],[1,0,2],[1,0,2],[1,0,2],[1,0,2],[1,0,2]],dtype=np.int64)
POS_EC = np.array([[1,0,2],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]],dtype=np.int64)
TAU=(1+np.sqrt(5))/2.0
CONST1=1/np.sqrt(2.0+TAU)
CONST2=1/2/np.sqrt(TAU+2)
PI=np.pi
TWOPI=2.0*PI

def z2atom(z):
    elements_dic={} 
    elements_dic[63] = 'Eu2+'
    return elements_dic[z]
    
def read_file(file):
    try:
        f=open(file,'r')
    except IOError as e:
        print(e)
        sys.exit(0)
    line=[]
    while 1:
        a=f.readline()
        if not a:
            break
        line.append(a[:-1])
    return line

if __name__ == "__main__":
    
    wpath='./work_mag'
    basename='mag'
    #basename='T_10_m4_D10_L_mag'
    
    # roatation matrces
    th1 = -np.arctan(1/TAU)
    th2 = -PI/10
    cos1=np.cos(th1)
    sin1=np.sin(th1)
    cos2=np.cos(th2)
    sin2=np.sin(th2)
    rotx=np.array([[   1,    0,    0],\
                   [   0, cos1, sin1],\
                   [   0, -sin1, cos1]],dtype=np.float64)
    rotz=np.array([[ cos2, sin2,    0],\
                   [-sin2, cos2,    0],\
                   [    0,    0,    1]],dtype=np.float64)
    
    print('reading data')
    aico=5.68930 # in Ang. CdYb
    a=read_file('%s/%s.mld'%(wpath,basename))
    lst_data=[]
    for i1 in range(2,len(a)):
        b=a[i1].split()
        site=np.array([float(b[0]),float(b[1]),float(b[2])],dtype=np.float64)/aico
        atom=z2atom(int(b[3]))
        spnvec=np.array([float(b[4]),float(b[5]),float(b[6])],dtype=np.float64)/2.0 # Note that the spin vectors were multiplied by two in the input data. back to the roginal ones.
        # roatation, x,y,z // twofold axes
        site=rotx@rotz@site
        spnvec=rotx@rotz@spnvec
        #
        lst_data.append([site,'Eu',spnvec])
    print('done')
    
    print('calc: nuclear and magnetic structure factors')
    wvl=1.0 # in Ang.
    qrange=5.0 # in Ang.^-1
    qinterval=0.2 # in Ang.^-1
    nmax=int(qrange/qinterval)
    ofname='%s_nmax%d_step%3.2f'%(basename,nmax,qinterval)
    fnuc=open('%s/%s_nuc.out'%(wpath,ofname),'w')
    fmag=open('%s/%s_mag.out'%(wpath,ofname),'w')
    z_nuc=np.zeros([2*nmax,2*nmax])
    z_mag=np.zeros([2*nmax,2*nmax])
    intensity_nuc_dict={}
    intensity_mag_dict={}
    intensity_tot_dict={}
    for ix in range(-nmax,nmax+1):
        print(ix)
        for iy in range(-nmax,nmax+1):
            #for iz in range(-nmax,nmax+1):
            for iz in range(0,1):
                qxyz=np.array([ix,iy,iz])*qinterval
                q_par_len=np.linalg.norm(qxyz)
                
                # nuclear form factor
                conc, Coh_b, Inc_b, Coh_xs, Inc_xs, Scatt_xs, Abs_xs = atm.scattering_lengths_and_cross_section('Eu',wvl)
                scattering_length_b=Coh_b # coherent scattering length, in fm
                affac=scattering_length_b
                
                # magnetic form factor
                flag2=1
                mffac1=atm.magnetic_form_factor('Eu2+',q_par_len,flag2)
                
                val1=0.0+0.0*1j
                val2=0.0+0.0*1j
                for atom in lst_data:
                    xyz = atom[0]
                    spnvec=atom[2]
                    con1 = np.dot(qxyz,xyz)
                    tmp = cmath.exp(TWOPI*con1*1j)
                    mu = np.dot(spnvec,qxyz)/(q_par_len**2*qxyz+1E-06) - np.array(spnvec) # 3d vector
                    val1 += tmp
                    val2 += mu*tmp
                sfc_nuc=affac*val1
                int_nuc=sfc_nuc.real**2 + sfc_nuc.imag**2
                #
                sfc_mag=mffac1*val2
                tmp=sfc_mag.real**2 + sfc_mag.imag**2
                int_mag=np.linalg.norm(tmp)
                fnuc.write('%8.6f %8.6f %8.6f %8.6f'%(qxyz[0],qxyz[1],qxyz[2],int_nuc))
                fmag.write('%8.6f %8.6f %8.6f %8.6f'%(qxyz[0],qxyz[1],qxyz[2],int_mag))
                #
                #z_nuc[ix,iy]=np.log10(int_nuc)
                #z_mag[ix,iy]=np.log10(int_mag)
                z_nuc[ix,iy]=int_nuc
                z_mag[ix,iy]=int_mag
                
                """
                # for powder pattern
                xaxis=round(q_par_len, 3)
                try:
                    intensity_nuc_dict[xaxis]+=int_nuc
                except:
                    intensity_nuc_dict[xaxis]=int_nuc
                try:
                    intensity_mag_dict[xaxis]+=int_mag
                except:
                    intensity_mag_dict[xaxis]=int_mag
                try:
                    intensity_tot_dict[xaxis]+=int_nuc+int_mag
                except:
                    intensity_tot_dict[xaxis]=int_nuc+int_mag
                """
    fnuc.close
    fmag.close
    
    plt.subplot(121)
    plt.title('magnetic sfc')
    plt.imshow(z_mag)
    plt.colorbar ()
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.subplot(122)
    plt.title('nuclear sfc')
    plt.imshow(z_nuc)
    plt.colorbar ()
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.savefig('%s/%s.png'%(wpath,ofname), format="png", dpi=300)
    
    """
    int_array = np.array([0.]*qrange*np.sqrt(3))
    intensity_max = max(intensity_tot_dict.values())
    for x in intensity_tot_dict:
        intensity = intensity_tot_dict[x]
        int_array += intensity
    # Intensity normalization
    int_array = int_array - np.min(int_array)
    int_array = int_array/np.max(int_array)
    """
    