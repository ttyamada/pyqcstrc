#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PyQCstrc - Python library for Quasi-Crystal structure
# Copyright (c) 2021 Tsunetomo Yamada <tsunetomo.yamada@rs.tus.ac.jp>

import os
import sys
import numpy as np

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

def merge(path,num0,name_list,filename,position):
    
    fatm=open('%s/%s.atm'%(path,filename),'w', encoding="utf-8", errors="ignore")
    fpod=open('%s/%s.pod'%(path,filename),'w', encoding="utf-8", errors="ignore")
    
    # pod, header
    fpod.write('nsymo=2 icent=1 brv=\'p\'\n')
    fpod.write('symmetry operator\n')
    fpod.write('y,z,t,-x+z,u,v          12\n')
    fpod.write('t,z,y,x,u,v              m\n')
    
    counter=num0
    for od_name in name_list: 
        atm=read_file('%s/%s.atm'%(path,od_name))
        pod=read_file('%s/%s.pod'%(path,od_name))
        
        num_of_v=[]
        for i2 in range(len(pod)):
            b=pod[i2].split()
            if len(b)>2:
                if b[3]=="'comment'":
                    num_of_v.append(int(b[1]))
                else:
                    pass
            else:
                pass
        
        num=int(len(atm)/7)
        counter0=1
        counter1=0
        
        for i2 in range(num):
            b=atm[7*i2].split()
            if num==1:
                fatm.write('%d %s %d %d %d %2.1f %2.1f %2.1f %2.1f %2.1f %2.1f %2.1f (%s@%s)\n'%\
                (counter,b[1],int(b[2]),counter,\
                float(b[4]),float(b[5]),float(b[6]),float(b[7]),float(b[8]),float(b[9]),float(b[10]),float(b[11]),\
                od_name,position))
                print('%d %s@%s'%\
                (counter,od_name,position))
            else:
                fatm.write('%d %s %d %d %d %2.1f %2.1f %2.1f %2.1f %2.1f %2.1f %2.1f (%s_p%d@%s)\n'%\
                (counter,b[1],int(b[2]),counter,\
                float(b[4]),float(b[5]),float(b[6]),float(b[7]),float(b[8]),float(b[9]),float(b[10]),float(b[11]),\
                od_name,counter0,position))
                print('%d %s_p%d@%s'%\
                (counter,od_name,counter0,position))
            for i3 in range(1,7):
                fatm.write('%s\n'%(atm[i3+7*i2]))
            
            #pod_v_index = 6 + counter1 + i2*4
            #pod_v_index = counter1 + i2*4
            pod_v_index = counter1 + i2*3
            #print(pod_v_index)
            #print(pod[pod_v_index])
            c=pod[pod_v_index].split()
            if num==1:
                fpod.write('%d %d %d (%s@%s)\n'%\
                (counter,int(c[1]),int(c[2]),od_name,position))
            else:
                fpod.write('%d %d %d (%s_p%d@%s)\n'%\
                (counter,int(c[1]),int(c[2]),od_name,counter0,position))
            for i3 in range(1,num_of_v[i2]+3):
                fpod.write('%s\n'%(pod[i3+pod_v_index]))
            
            counter+=1
            counter0+=1
            counter1+=num_of_v[i2]
    
    fatm.close()
    fpod.close()
    return 0

if __name__ == '__main__':
    
    f1='od_a_1_asym'
    f2='od_a_2_asym'
    f3='od_a_3_asym'
    f4='od_a_4_asym'
    f5='od_a_5_asym'
    f6='od_a_6_asym'
    f7='od_c_1_asym'
    f8='od_c_2_asym'
    f9='od_c_3_asym'
    f10='od_c_4_asym'
    
    fname=[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10]
    
    num0=1
    path='work7_podatm'
    filename='dd_tate_asym'
    pos= '' #'V'
    merge(path,num0,fname,filename,pos)