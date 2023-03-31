from rec_svtr import SVTRNet
import mindspore as ms
from mindspore import context, Tensor
import numpy as np
context.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU",device_id=7)
# import pdb;pdb.set_trace()
model = SVTRNet().set_train(False)
np.random.seed(2022)
x=np.random.randn(64,3,32,100).astype(np.float32)
b=Tensor(x)
# b = ms.ops.ones((64,3,32,100), ms.float32)
output=model(b)
np_out=output.asnumpy()

print("the output of ms is ",output.shape,output)