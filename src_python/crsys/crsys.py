import numpy as np

def crsys_init(isys_:np.int64):
    global isys,n,N
    isys=isys_
    if isys==2: # icosahedral
        n=6
        N=5 # for sqrt(5) in qnnum
    elif isys==3: # decagonal
        n=5
        N=5
    elif isys==4: # octabonal
        n=5
        N=2 # for sqrt(2) in qnnmum
    elif isys==5: # dodecagonal
        n=5
        N=3 # for sqrt(3) in qnnum
        