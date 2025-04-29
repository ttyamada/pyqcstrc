import numpy as np

def crsys_init(isys_:np.int64):
    global isys,n,N,ne,ni,scly
    isys=isys_
    if isys==2: # icosahedral
        n=6
        N=5 # for sqrt(5) in qnnum
        ne=3
        ni=3
        scly=1
    elif isys==3: # decagonal
        n=5
        N=5
        ne=3
        ni=2
        scly=np.sin(2*np.pi/5) #s1
    elif isys==4: # octabonal
        n=5
        N=2 # for sqrt(2) in qnnmum
        ne=3
        ni=2
        scly=1
    elif isys==5: # dodecagonal
        n=5
        N=3 # for sqrt(3) in qnnum
        ne=3
        ni=2
        scly=1
    else:
        print("isys should be 2,3,4 or r")
        exit()
        