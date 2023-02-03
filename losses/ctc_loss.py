from __future__ import division
from __future__ import print_function

import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore import ops
import mindspore.numpy as np

class CTCLoss(nn.Cell):
    def __init__(self, use_focal_loss=False, **kwargs):
        super(CTCLoss, self).__init__()
        # self.loss_func = nn.CTCLoss(blank=0, reduction='none')
        self.loss_func = ops.CTCLoss()
        self.use_focal_loss = use_focal_loss

    def construct(self, predicts, batch):
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]
        predicts = predicts.transpose((1, 0, 2))
        N, B, _ = predicts.shape
        preds_lengths = Tensor([N] * B, dtype=ms.int64)
        labels = batch[1].astype("int32")
        label_lengths = batch[2].astype('int64')
        loss = self.loss_func(predicts, labels, preds_lengths, label_lengths)
        if self.use_focal_loss:
            weight = ops.exp(-loss)
            weight = np.subtract(Tensor([1.0]), weight)
            weight = ops.Square(weight)
            loss = np.multiply(loss, weight)
        loss = loss.mean()
        return {'loss': loss}