from mindspore import Tensor, context
import mindspore
import numpy as np
import mindspore.nn as nn
from mindspore.ops.operations import CTCLossV2
from mindspore.ops import composite as C

context.set_context(mode=mindspore.PYNATIVE_MODE, device_target="GPU")
predicts = np.load("predicts.npy")
labels = np.load("labels.npy")
preds_lengths = np.load("preds_lengths.npy")
label_lengths = np.load("label_lengths.npy")

class test_CTC(nn.Cell):
    def __init__(self):
        super(test_CTC, self).__init__()
        self.loss_func = CTCLossV2(blank=0, reduction='none')
        self.grad = C.GradOperation(get_all=True, sens_param=False)

    def construct(self, predicts, labels, preds_lengths, label_lengths):
        loss, _ = self.loss_func(predicts, labels, preds_lengths, label_lengths)
        grads = self.grad(self.loss_func)(predicts, labels, preds_lengths, label_lengths)
        return loss.mean(), grads

test_net = test_CTC()
loss, grad = test_net(Tensor(predicts), Tensor(labels), Tensor(preds_lengths), Tensor(label_lengths))
print(loss)
print("--------------------------")
print(grad)
