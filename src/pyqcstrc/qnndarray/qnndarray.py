import numpy as np
import pyqcstrc.qnnum.qnnum as qnn

class QnNdarray(np.ndarray): # only for ndim=2
    def __new__(cls, shape, N:np.int64):
        return super().__new__(cls,shape,dtype=qnn.Qnnum)

    def __init__(self,shape, N:np.int64): # only for ndim=2
        qn0=qnn.Qnnum([0,0,1],N)
        qnn.printqnn("qn0",qn0)
        print("self.shape",self.shape)  # for test
        print("self.ndim",self.ndim)    # for test
        print("self.dtype",self.dtype)  # for test
        
        it = np.nditer(self, flags=['multi_index','refs_ok'], op_flags=['readwrite'])
        while not it.finished:  # loop up to last index
            it[0] = qn0
            idx = it.multi_index
            #print('idx=', idx ,', self[idx]=', self[idx], ', it[0]=', it[0]) # for test
            it.iternext()   #it : next index

        printqndm("Qnmat self",self) # for test
        
# only ndim=1,2,3
def printqndm(str:str, qnm:QnNdarray):
    ndim=qnm.ndim
    print("qnm.ndim",qnm.ndim)
    print("qnm.shape",qnm.shape)
    print(str)
    if ndim==1: # for a vector
        for i in range(qnm.shape[0]):
            print("[",end=" ")
            print(qnn.qn2npa(qnm[i]),end=" ")
        print("]")
    elif ndim==2: # for a matrix
        for i in range(qnm.shape[0]):
            print("[",end=" ")
            for j in range(qnm.shape[1]):
                print(qnn.qn2npa(qnm[i][j]),end=" ")
            print("]")
        print("")
    elif ndim==3: # for a matrix array
        for i in range(qnm.shape[0]):
            print("")
            for j in range(qnm.shape[1]):
                print("[",end=" ")
                for k in range(qnm.shape[2]):
                    print(qnn.qn2npa(qnm[i][j][k]),end=" ")
                print("]")
        print("")
    else:
        print("ord in printqnm should be 1, 2 or 3 but",ord); exit()
        
if __name__ == '__main__':
    # test
    
    N=2
    n=5
    print("n=",n)
    shape=(n,n)
    qndm=QnNdarray(shape,N) # nxn qmnum zero matrix
    print("qndm.ndim",qndm.ndim)
    print("qndm.shape",qndm.shape)
    printqndm("zero qnmat",qndm)
