import numpy as np
import qnnum as qnn
import qnndarray as qna
import qnvec as qnv
import cython

def tr7to5e(ei:qna.QnNdarray) -> qna.QnNdarray:  # for direct space vector
  #      for dodecagonal QCs
  sd =[[0, 1, 0, 0, 0,-1, 0],
       [1, 0, 1, 0, 0, 0, 0],
       [0, 1, 0, 1, 0, 0, 0],
       [0, 0, 1, 0, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 1]]
  shape=ei.shape
  nv=shape[0]
  fi = qna.zeros((nv,5))
  qn0=qnn.zero()
  for i in range(nv):
    for  j in range(5):
      fi[i][j]=qn0
      for k in range(7):
        fi[i][j]+=ei[i][k]*sd[j][k]
  return fi

def tr7to5i(ei:qna.QnNdarray) -> qna.QnNdarray:  # for reciprocal vector
  #      for dodecagonal QCs
  sd = [[1, 0, 0, 0,-1, 0, 0],
        [0, 1, 0, 0, 0,-1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1]]
  
  shape=ei.shape
  nv=shape[0]
  qn0=qnn.zero()
  fi = qna.zeros((nv,5))
  for i in range(nv):
    for  j in range(5):
      fi[i][j]=qn0
      for  k in range(7):
        fi[i][j]+=ei[i][k]*sd[j][k]
  return fi

