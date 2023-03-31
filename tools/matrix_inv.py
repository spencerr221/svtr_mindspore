import mindspore
from mindspore import Tensor, ops, context
import mindspore.numpy as np
import numpy
from numpy.linalg import det

context.set_context(mode=mindspore.PYNATIVE_MODE, device_target="GPU",device_id=7)
matrix_inverse = ops.MatrixInverse(adjoint=False)

def cof1(M,index):
    zs = M[:index[0]-1,:index[1]-1]
    ys = M[:index[0]-1,index[1]:]
    zx = M[index[0]:,:index[1]-1]
    yx = M[index[0]:,index[1]:]
    s = numpy.concatenate((zs,ys),axis=1)
    x = numpy.concatenate((zx,yx),axis=1)
    matrix = numpy.concatenate((s,x),axis=0)
    ans = det(matrix)
    return ans
 
def alcof(M,index):
    return pow(-1,index[0]+index[1])*cof1(M,index)
 
def adj(M):
    result = numpy.zeros((M.shape[0],M.shape[1]))
    for i in range(1,M.shape[0]+1):
        for j in range(1,M.shape[1]+1):
            result[j-1][i-1] = alcof(M,[i,j])
    return result
 
def invmat(M):
    # det_m = M.matrix_determinant()
    return 1.0/det(M)*adj(M)
 
numpy.random.seed(2022)
x=numpy.random.randn(23,23).astype(numpy.float64)
input_x=Tensor(x)

import time
start = time.time()
ans = invmat(x)
print("the time of self developing function is:",time.time()-start)

print("======================================================")
start_ms = time.time()
ans_ms = matrix_inverse(input_x)
print("the time of mindspore operator is:",time.time()-start_ms)
print("the results of Mindspore:",ans_ms[0][0].dtype)
print("the results of self_developing:",ans[0][0].dtype)
diff=numpy.max(numpy.abs(ans_ms.asnumpy()-ans))
print("the diff is:", diff)