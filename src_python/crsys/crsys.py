import numpy as np

def crsys_init(isys_:np.int64):
    global isys,n,N,ne,ni
    isys=isys_
    if isys==2: # icosahedral
        n=6
        N=5 # for sqrt(5) in qnnum
        ne=3
        ni=3
    elif isys==3: # decagonal
        n=5
        N=5
        ne=3
        ni=2
    elif isys==4: # octabonal
        n=5
        N=2 # for sqrt(2) in qnnmum
        ne=3
        ni=2
    elif isys==5: # dodecagonal
        n=5
        N=3 # for sqrt(3) in qnnum
        ne=3
        ni=2
        