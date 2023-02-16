from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import mindspore
from mindspore import nn, Parameter, Tensor
# from mindspore.nn import functional as F
import numpy as np
import mindspore.numpy as ms_np
import mindspore.ops as ops
from mindspore.common.initializer import initializer, Normal
import itertools

matrix_inverse = ops.MatrixInverse(adjoint=False)

def build_output_control_points(num_control_points, margins):
    margin_x, margin_y = margins
    num_ctrl_pts_per_side = num_control_points // 2
    ctrl_pts_x = np.linspace(margin_x, 1.0 - margin_x, num_ctrl_pts_per_side)
    ctrl_pts_y_top = np.ones(num_ctrl_pts_per_side) * margin_y
    ctrl_pts_y_bottom = np.ones(num_ctrl_pts_per_side) * (1.0 - margin_y)
    ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
    ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
    output_ctrl_pts_arr = np.concatenate(
        [ctrl_pts_top, ctrl_pts_bottom], axis=0)
    output_ctrl_pts = Tensor(output_ctrl_pts_arr)
    return output_ctrl_pts

# phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
def compute_partial_repr(input_points, control_points):
    N = input_points.shape[0]
    M = control_points.shape[0]
    pairwise_diff = ops.reshape(
        input_points,(N, 1, 2)) - ops.reshape(
            control_points, (1, M, 2))
    # original implementation, very slow
    # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    pairwise_diff_square = pairwise_diff * pairwise_diff
    cast = ops.Cast()
    pairwise_diff_square=cast(pairwise_diff_square,mindspore.float32)
    test_1=pairwise_diff_square[:, :, 0]
    test_2=pairwise_diff_square[:, :,1]
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :,1]
    repr_matrix = 0.5 * pairwise_dist * ms_np.log(pairwise_dist)
    # fix numerical error for 0 * log(0), substitute all nan with 0
    mask = np.array(repr_matrix != repr_matrix)
    # repr_matrix[mask] = 0
    return repr_matrix

def grid_sample(input, grid, canvas=None):
    # input.stop_gradient = False     #TODO may not work
    output = ops.grid_sample(input.astype(mindspore.float32), grid.astype(mindspore.float32))
    if canvas is None:
        return output
    else:
        input_mask = ops.ones_like(input)
        output_mask = ops.grid_sample(input_mask, grid)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output

class TPSSpatialTransformer(nn.Cell):
    def __init__(self,
                 output_image_size=None,
                 num_control_points=None,
                 margins=None):
        super(TPSSpatialTransformer, self).__init__()
        self.output_image_size = output_image_size
        self.num_control_points = num_control_points
        self.margins = margins

        self.target_height, self.target_width = output_image_size
        target_control_points = build_output_control_points(num_control_points,
                                                            margins)
        N = num_control_points

        # create padded kernel matrix
        forward_kernel = np.zeros((N + 3, N + 3))

        target_control_partial_repr = compute_partial_repr(
            target_control_points, target_control_points)
        target_control_partial_repr = ops.cast(target_control_partial_repr,mindspore.float32)
        forward_kernel[:N, :N] = target_control_partial_repr
        forward_kernel[:N, -3] = 1
        forward_kernel[-3, :N] = 1
        target_control_points = ops.cast(target_control_points,mindspore.float32)
        forward_kernel[:N, -2:] = target_control_points
        forward_kernel[-2:, :N] = ops.transpose(
            target_control_points, input_perm=(1, 0))
        # compute inverse matrix
        forward_kernel=Tensor(forward_kernel)

        inverse_kernel = matrix_inverse(forward_kernel)

        # create target cordinate matrix
        HW = self.target_height * self.target_width
        target_coordinate = list(
            itertools.product(
                range(self.target_height), range(self.target_width)))
        target_coordinate = Tensor(target_coordinate)  # HW x 2
        Y, X = ops.split(target_coordinate, output_num=target_coordinate.shape[1], axis=1)
        #----------------for Ascend----------------
        # Y=ops.cast(Y,mindspore.float16)
        # X = ops.cast(X, mindspore.float16)
        # height_others=self.target_height-1
        # width_others=self.target_width-1
        # height_others=ops.cast(height_others,mindspore.float16)
        # width_others = ops.cast(width_others, mindspore.float16)
        # Y=np.divide(Y,height_others)
        # X=np.divide(X,width_others)

        Y=ops.cast(Y,mindspore.int16)
        X = ops.cast(X, mindspore.int16)
        height_others=self.target_height-1
        width_others=self.target_width-1
        height_others=ops.cast(height_others,mindspore.int16)
        width_others = ops.cast(width_others, mindspore.int16)
        Y = Y / (height_others - 1)
        X = X / (width_others - 1)
        target_coordinate = ops.concat(
            [X, Y], axis=1)  # convert from (y, x) to (x, y)
        target_coordinate_partial_repr = compute_partial_repr(
            target_coordinate, target_control_points)
        target_coordinate_repr = ops.concat(
            [
                target_coordinate_partial_repr, ops.ones((HW, 1), mindspore.float32),
                ops.cast(target_coordinate,mindspore.float32)
            ],
            axis=1)

        # register precomputed matrices
        self.inverse_kernel = inverse_kernel
        self.padding_matrix = ms_np.zeros((3, 2),mindspore.float32)
        self.target_coordinate_repr = target_coordinate_repr
        self.target_control_points = target_control_points

    def construct(self, input, source_control_points):
        assert source_control_points.ndim == 3
        assert source_control_points.shape[1] == self.num_control_points
        assert source_control_points.shape[2] == 2
        batch_size = ops.shape(source_control_points)[0]
        expand_as=ops.ones((batch_size,3,2),mindspore.float32)

        padding_matrix = self.padding_matrix.expand_as(expand_as)
        Y = ops.concat([source_control_points, padding_matrix], 1)
        mapping_matrix = ops.matmul(self.inverse_kernel, Y)
        mapping_matrix = ops.cast(mapping_matrix, mindspore.float16)
        source_coordinate = ops.matmul(self.target_coordinate_repr,mapping_matrix)   #float32 and float32

        grid = ops.reshape(
            source_coordinate,
            (source_coordinate.shape[0], self.target_height, self.target_width, 2))           #TODO may not work
        grid = grid.clip(0,1)  # the source_control_points may be out of [0, 1].
        # the input to grid_sample is normalized [-1, 1], but what we get is [0, 1]
        grid = 2.0 * grid - 1.0
        input = ops.cast(input,mindspore.float32)
        output_maps = grid_sample(input, grid, canvas=None)
        return output_maps, source_coordinate