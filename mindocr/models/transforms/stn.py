"""
This code is refer from:
https://github.com/ayumiymk/aster.pytorch/blob/master/lib/models/stn_head.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import mindspore
from mindspore import nn, Parameter, Tensor
# from mindspore.nn import functional as F
import mindspore.numpy as np
import mindspore.ops as ops
from mindspore.common.initializer import initializer, Normal

from .tps_spatial_transformer import TPSSpatialTransformer
sigmoid = nn.Sigmoid()

def conv3x3_block(in_channels, out_channels, stride=1):
    n = 3 * 3 * out_channels
    w = math.sqrt(2. / n)
    conv_layer = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        pad_mode="pad",
        padding=1,
        weight_init=Normal(mean=0.0, sigma=w),
        has_bias=True,
        bias_init='zeros')
    block = nn.SequentialCell(conv_layer, nn.BatchNorm2d(out_channels), nn.ReLU())
    return block


class STN(nn.Cell):
    def __init__(self, in_channels, num_ctrlpoints, activation='none'):
        super(STN, self).__init__()
        self.in_channels = in_channels
        self.num_ctrlpoints = num_ctrlpoints
        self.activation = activation
        self.stn_convnet = nn.SequentialCell(
            conv3x3_block(in_channels, 32),  #32x64
            nn.MaxPool2d(
                kernel_size=2, stride=2),
            conv3x3_block(32, 64),  #16x32
            nn.MaxPool2d(
                kernel_size=2, stride=2),
            conv3x3_block(64, 128),  # 8*16
            nn.MaxPool2d(
                kernel_size=2, stride=2),
            conv3x3_block(128, 256),  # 4*8
            nn.MaxPool2d(
                kernel_size=2, stride=2),
            conv3x3_block(256, 256),  # 2*4,
            nn.MaxPool2d(
                kernel_size=2, stride=2),
            conv3x3_block(256, 256))  # 1*2
        self.stn_fc1 = nn.SequentialCell(
            nn.Dense(
                2 * 256,
                512,
                weight_init=Normal(mean=0, sigma=0.001),
                bias_init='zeros'),
            nn.BatchNorm1d(512),
            nn.ReLU())
        fc2_bias = self.init_stn()
        self.stn_fc2 = nn.Dense(
            512,
            num_ctrlpoints * 2,
            weight_init='zeros',
            bias_init=fc2_bias)    #TODO input is one dimision matrix

    def init_stn(self):
        margin = 0.01
        sampling_num_per_side = int(self.num_ctrlpoints / 2)
        ctrl_pts_x = np.linspace(margin, 1. - margin, sampling_num_per_side)
        ctrl_pts_y_top = np.ones(sampling_num_per_side) * margin
        ctrl_pts_y_bottom = np.ones(sampling_num_per_side) * (1 - margin)
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        ctrl_points = np.concatenate(
            [ctrl_pts_top, ctrl_pts_bottom], axis=0).astype(np.float32)
        if self.activation == 'none':
            pass
        elif self.activation == 'sigmoid':
            ctrl_points = -np.log(1. / ctrl_points - 1.)
        ctrl_points = Tensor(ctrl_points)
        shape=ctrl_points.shape[0] * ctrl_points.shape[1]
        fc2_bias = np.reshape(
            ctrl_points, [ctrl_points.shape[0] * ctrl_points.shape[1]])
        return fc2_bias

    def construct(self, x: Tensor) -> Tensor:
        x = self.stn_convnet(x)
        batch_size, c, h, w = x.shape
        x = ops.reshape(x, (batch_size, c*h*w))
        img_feat = self.stn_fc1(x)
        x = self.stn_fc2(0.1 * img_feat)
        if self.activation == 'sigmoid':
            x = sigmoid(x)
        x = ops.reshape(x, (-1, self.num_ctrlpoints, 2))
        return img_feat, x


class STN_ON(nn.Cell):
    def __init__(self, in_channels, tps_inputsize, tps_outputsize,
                 num_control_points, tps_margins, stn_activation):
        super(STN_ON, self).__init__()
        self.tps = TPSSpatialTransformer(
            output_image_size=tuple(tps_outputsize),
            num_control_points=num_control_points,
            margins=tuple(tps_margins))
        self.stn_head = STN(in_channels=in_channels,
                            num_ctrlpoints=num_control_points,
                            activation=stn_activation)
        self.tps_inputsize = tps_inputsize
        self.out_channels = in_channels

    def construct(self, image):
        stn_input = ops.interpolate(
            image, sizes=tuple(self.tps_inputsize), mode="bilinear")
        stn_img_feat, ctrl_points = self.stn_head(stn_input)   # check pass

        # import pdb;pdb.set_trace()
        x, _ = self.tps(image, ctrl_points)
        return x