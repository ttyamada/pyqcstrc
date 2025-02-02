import numpy as np
import pyqcstrc.qnnum.qnnum as qnn
import pyqcstrc.qnmat.qnmat as qnm

class QnNdarray(np.ndarray): # only for ndim=2
    def __new__(cls, n:np.int64, N:np.int64):
        shape=(n,n)
        return super().__new__(cls,shape,dtype=qnn.Qnnum)

    def __init__(self,n:np.int64, N:np.int64): # only for ndim=2
        qn0=qnn.Qnnum([0,0,1],N)
        #self=np.full((n,n),qn0) # 2D array
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self[i][j]=qn0
        #print("self.shape",self.shape)  # for test
        #print("self.ndim",self.ndim)    # for test
        #print("self.dtype",self.dtype)  # for test
        #qnm.printqnm("Qnmat self",self) # for test
        
def printqndm(str:str, qnm:QnNdarray):
    ndim=qnm.ndim
    print("qnm.ndim",qnm.ndim)
    print("qnm.shape",qnm.shape)
    print(str)
    if ndim==1:
        for i in range(qnm.shape[0]):
            print("[",end=" ")
            print(qnn.qn2npa(qnm[i]),end=" ")
        print("]")
    elif ndim==2:
        for i in range(qnm.shape[0]):
            print("[",end=" ")
            for j in range(qnm.shape[1]):
                print(qnn.qn2npa(qnm[i][j]),end=" ")
            print("]")
        print("]")
    else:
        print("ord in printqnm should be 1 or2 but",ord); exit()
        
if __name__ == '__main__':
    # test
    
    N=2
    n=5
    print("n=",n)
    qndm=QnNdarray(n,N) # nxn qmnum zero matrix
    print("qndm.ndim",qndm.ndim)
    print("qndm.shape",qndm.shape)
    printqndm("zero qnmat",qndm)
