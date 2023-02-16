from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import math

import mindspore
from mindspore import Parameter,nn, Tensor
import mindspore.ops as ops
import mindspore.common.initializer as init


# def get_para_bias_attr(l2_decay, k):
#     regularizer = paddle.regularizer.L2Decay(l2_decay)  #TODO: ms version do not support l2 coefficient in paddle this is a class
#     stdv = 1.0 / math.sqrt(k * 1.0)
#     initializer = nn.initializer.Uniform(-stdv, stdv)    #TODO: ms version needs shape param
#     weight_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
#     bias_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
#     return [weight_attr, bias_attr]
def get_para_bias_attr(k):
    stdv = 1.0 / math.sqrt(k * 1.0)
    initializer = init.Uniform(stdv)
    weight_attr = initializer
    bias_attr = initializer
    return [weight_attr, bias_attr]

class CTCHead(nn.Cell):
    def __init__(self,
                 in_channels,    #TODO forward
                 out_channels,    #TODO forward
                 fc_decay=0.0004,
                 mid_channels=None,
                 return_feats=False,
                 **kwargs):
        super(CTCHead, self).__init__()
        if mid_channels is None:
            weight_attr, bias_attr = get_para_bias_attr(k=in_channels)  #TODO fix get_para_bias_attr
            self.fc = nn.Dense(
                in_channels,
                out_channels,
                bias_init=bias_attr)
        else:
            # weight_attr1, bias_attr1 = get_para_bias_attr(
            #     l2_decay=fc_decay, k=in_channels)
            weight_attr1, bias_attr1 = get_para_bias_attr(k=in_channels)  # TODO fix get_para_bias_attr
            self.fc1 = nn.Dense(
                in_channels,
                mid_channels,
                bias_init=bias_attr1)

            # weight_attr2, bias_attr2 = get_para_bias_attr(
            #     l2_decay=fc_decay, k=mid_channels)
            weight_attr2, bias_attr2 = get_para_bias_attr(k=in_channels)  # TODO fix get_para_bias_attr
            self.fc2 = nn.Dense(
                mid_channels,
                out_channels,
                bias_init=bias_attr2)
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats


    def construct(self, x, targets=None):

        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)

        if self.return_feats:
            result = (x, predicts)
        else:
            result = predicts
        if not self.training:
            softmax=nn.Softmax(axis=2)
            predicts = softmax(predicts)
            result = predicts

        return result