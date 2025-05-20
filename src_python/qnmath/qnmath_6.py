
def det_matrix_2d(mtx: qnm.Qnmat) -> qnn.Qnnum:
    """Determinant of 3x3 matrix, mtx, in qnnumber
    
    Parameters
    ----------
    mtx: array
        2x2 matrix in qnnumer

    Returns
    -------
    determinant in qnnumber
    """
    #N=mtx.N
    shape=mtx.shape
    if shape[0]!=2:
        print("shape of mtx in det_matrix_2d should be (3,3) but",shape); exit(0)
    det=qnn.Qnnum([0,0,1]) # zero qnnumber
    det=det+mtx[0][0]*mtx[1][1]
    det=det-mtx[0][1]*mtx[1][0]
    return det


# this should be a function
def matrixtr(mtx: qnm.Qnmat) -> qnm.Qnmat:
    """ return transposed matrix of mtx """
    #N=mtx[0][0].N
    shape=mtx.shape
    #n_=mtx.shape[0]
    mtxt=qnm.Qnmat(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            mtxt[i][j]=qnn.copy(mtx[j][i])
    return mtxt



def matrixtr_i(mtx: NDArray[np.int64]) -> NDArray[np.int64]:
    """ return transposed matrix of mtx """
    shape=mtx.shape
    #n_=mtx.shape[0]
    mtxt=qnm.Qnmat(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            mtxt[i][j]=qnn.copy(mtx[j][i])
    return mtxt
