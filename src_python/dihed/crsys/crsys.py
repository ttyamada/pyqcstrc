import numpy as np

class Crsys:
    def __init__(self, isys:np.int64):
        self.isys=isys
        if self.isys==2: # icosahedral
            self.n=6
            self.N=5 # for sqrt(5) in qnnum
            self.ne=3
            self.ni=3
            self.scly=1
            self.scly2=1
        elif self.isys==3: # decagonal
            self.n=5
            self.N=5
            self.ne=3
            self.ni=2
            self.scly=np.sin(2*np.pi/5)  #s1
            self.scly2=self.scly*self.scly
        elif self.isys==4: # octabonal
            self.n=5
            self.N=2 # for sqrt(2) in qnnmum
            self.ne=3
            self.ni=2
            self.scly=1
            self.scly2=1
        elif self.isys==5: # dodecagonal
            self.n=5
            self.N=3 # for sqrt(3) in qnnum
            self.ne=3
            self.ni=2
            self.scly=1
            self.scly2=1
        else:
            print("isys should be 2,3,4 or r")
            exit()


def crsys_init(isys: np.int64):
    return Crsys(isys)
