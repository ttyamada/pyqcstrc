import qnnum as qnn
import qnndarray as qna
import qnvec as qnv
import cython

#       for decagonal QCs
def tr6to5e(ei:qna.QnNdarray) -> qna.QnNdarray: # for direct space vector
  sd =[[4,-1,-1,-1,-1, 0],
      [-1, 4,-1,-1,-1, 0],
      [-1,-1, 4,-1,-1, 0],
      [-1,-1,-1, 4,-1, 0],
      [ 0, 0, 0, 0, 0, 5]]
  
  shape=ei.shape
  nv=shape[0]
  fi = qna.zeros((nv,5))
  qn0=qnn.zero()
  for i in range(nv):
    for  j in range(5):
      fi[i][j]=qn0
      for  k in range(6):
        fi[i][j]+=ei[i][k]*sd[j][k]/5
  return fi

def tr6to5i(ei:qna.QnNdarray) -> qna.QnNdarray:  # for reciprocal space vector
  sd = [[1, 0, 0, 0,-1, 0],
        [0, 1, 0, 0,-1, 0],
        [0, 0, 1, 0,-1, 0],
        [0, 0, 0, 1,-1, 0],
        [0, 0, 0, 0, 0, 1]]

  shape=ei.shape
  nv=shape[0]
  fi = qna.zeros((nv,5))
  qn0=qnn.zero()
  for i in range(nv):
    for j in range(5):
      fi[i][j]=qn0
      for k in range(6):
        fi[i][j]+=ei[i][k]*sd[k][j]
  return fi

  