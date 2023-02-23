from __future__ import division
from __future__ import print_function

import mindspore
import mindspore as ms
from mindspore import nn
from mindspore import Tensor,Parameter
from mindspore.common import dtype as mstype
from mindspore import ops
from mindspore.ops.operations import CTCLossV2
# from mindspore.nn.loss import CTCLoss
import numpy as np
from mindspore.nn.loss.loss import LossBase

class CTCLoss(LossBase):
    def __init__(self, use_focal_loss=False, **kwargs):
        super(CTCLoss, self).__init__()
        # self.loss_func = nn.CTCLoss(blank=0, reduction='none')
        self.loss_func = CTCLossV2(blank=36, reduction='none')
        self.use_focal_loss = use_focal_loss


    def construct(self, predicts, batch):
        predicts = ops.log_softmax(predicts)
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]
        predicts = predicts.transpose((1, 0, 2))
        # import pdb;pdb.set_trace()
        N, B, _ = predicts.shape
        preds_lengths = Tensor([N] * B, dtype=ms.int32)
        # import pdb;pdb.set_trace()
        # label_lengths = np.count_nonzero(batch != 36, axis=(-1)).astype('int32')
        label_lengths = ops.count_nonzero((batch != 36) * 1, axis=(-1)).astype('int32')
        # label_lengths = Tensor(label_lengths, mindspore.int32)
        labels = batch.astype("int64")
        predicts=predicts.astype("float32")
        # label_lengths = lens.astype('int32')
        loss, _ = self.loss_func(predicts, labels, preds_lengths, label_lengths)
        if self.use_focal_loss:
            print("under develop")
            # weight = ops.exp(-loss)
            # weight = np.subtract(Tensor([1.0]), weight)
            # weight = ops.Square(weight)
            # loss = np.multiply(loss, weight)
        loss = loss.mean()
        # print("loss:",loss)

        return loss

# class CTCLoss(LossBase):
#     """
#      CTCLoss definition
#
#      Args:
#         max_sequence_length(int): max number of sequence length. For text images, the value is equal to image
#         width
#         max_label_length(int): max number of label length for each input.
#         batch_size(int): batch size of input logits
#      """
#
#     def __init__(self, max_sequence_length, max_label_length, batch_size):
#         super(CTCLoss, self).__init__()
#
#         self.max_sequence_length = max_sequence_length
#         self.max_label_length = max_label_length
#         self.batch_size = batch_size
#         self.labels_indices, self.sequence_length = self.get_input_args()
#
#         self.ctc_loss = ops.CTCLoss(ctc_merge_repeated=True)
#
#     def get_input_args(self):
#         labels_indices = []
#         for i in range(self.batch_size):
#             for j in range(self.max_label_length):
#                 labels_indices.append([i, j])
#         # labels_indices = Parameter(
#         #     Tensor(np.array(labels_indices), mstype.int64), requires_grad=False)
#         # sequence_length = Parameter(Tensor(np.array(
#         #     [self.max_sequence_length] * self.batch_size), mstype.int32), requires_grad=False)
#         labels_indices = Tensor(np.array(labels_indices), mstype.int64)
#         # print("lllllllllllllllllllllabel", labels_indices)
#         sequence_length = Tensor(np.array(
#             [self.max_sequence_length] * self.batch_size), mstype.int32)
#         return labels_indices, sequence_length
#
#     def construct(self, logit, label):
#
#         if (label > 37).any():
#             print("Input label indices of Loss function!!!")
#             print(label)
#             print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
#         logit = logit.transpose((1,0,2))
#         labels_values = label.flatten()
#         labels_values=ops.cast(labels_values,mstype.int32)
#         sequence_length=ops.cast(self.sequence_length,mstype.int32)
#         # print("logit, self.labels_indices, labels_values, sequence_length",logit, self.labels_indices, labels_values.shape, sequence_length.shape)
#
#         loss, _ = self.ctc_loss(
#             logit, self.labels_indices, labels_values, sequence_length)
#         loss = loss.mean()
#         # print("loss:",loss)
#         return loss

# class CTCLoss(LossBase):
#     """
#      CTCLoss definition
#
#      Args:
#         max_sequence_length(int): max number of sequence length. For text images, the value is equal to image
#         width
#         max_label_length(int): max number of label length for each input.
#         batch_size(int): batch size of input logits
#      """
#
#     def __init__(self, max_sequence_length, max_label_length):
#         super(CTCLoss, self).__init__()
#
#         self.max_sequence_length = max_sequence_length
#         self.max_label_length = max_label_length
#         self.labels_indices = None
#         self.sequence_length = None
#
#         self.ctc_loss = ops.CTCLoss(ctc_merge_repeated=True)
#
#     def construct(self, logit, label):
#         if self.labels_indices is None:
#             batch_size = logit.shape[1]
#             labels_indices = []
#             for i in range(batch_size):
#                 for j in range(self.max_label_length):
#                     labels_indices.append([i, j])
#
#             self.labels_indices = Tensor(
#                 np.array(labels_indices), mstype.int64)
#             self.sequence_length = Tensor(
#                 np.array([self.max_sequence_length] * batch_size), mstype.int32)
#
#         if (label.asnumpy() > 37).any():
#             print("Input label indices of Loss function!!!")
#             print(label)
#             print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
#             return Tensor(0.0, mstype.float16)
#
#         labels_values = label.flatten()
#         labels_values = ops.cast(labels_values, mstype.int32)
#         loss, _ = self.ctc_loss(
#             logit, self.labels_indices, labels_values, self.sequence_length)
#         loss = loss.mean()
#         return loss