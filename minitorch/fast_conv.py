from typing import Tuple

import numpy as np
from numba import njit, prange

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Index,
    Shape,
    Strides,
    Storage,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right
    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = input_strides
    s2 = weight_strides

    r = int(reverse)
    for out_i in prange(out.size):
        o_idx = np.zeros_like(out_shape)
        to_index(out_i, out_shape, o_idx)
        o_pos = index_to_position(o_idx, out_strides)

        b, o_channel = o_idx[0], o_idx[1]
        init_s = (1 - r) * o_idx[2] + r * (o_idx[2] - kw + 1)

        for i_channel in prange(in_channels):
            base_i = b * s1[0] + i_channel * s1[1]
            base_w = o_channel * s2[0] + i_channel * s2[1]
            for k in prange(kw):
                input_offset = init_s + k
                if input_offset < width and input_offset >= 0:
                    i_pos = base_i + input_offset * s1[2]
                    w_pos = base_w + ((1 - r) * k + r * (kw - k - 1)) * s2[2]
                    out[o_pos] = out[o_pos] + input[i_pos] * weight[w_pos]


tensor_conv1d = njit(parallel=True)(_tensor_conv1d)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 1D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x w
            weight : out_channel x in_channel x kw

        Returns:
            batch x out_channel x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right
    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides
    # inners
    s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
    s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]

    r = int(reverse)
    for out_i in prange(out.size):
        o_idx = np.zeros_like(out_shape)
        to_index(out_i, out_shape, o_idx)
        o_pos = index_to_position(o_idx, out_strides)

        b, o_channel = o_idx[0], o_idx[1]
        init_h = (1 - r) * o_idx[2] + r * (o_idx[2] - kh + 1)
        init_w = (1 - r) * o_idx[3] + r * (o_idx[3] - kw + 1)

        for i_channel in prange(in_channels):
            base_i = b * s10 + i_channel * s11
            base_w = o_channel * s20 + i_channel * s21
            for h in prange(kh): # rows
                for w in prange(kw): # cols
                    if (init_h + h) < height and (init_h + h) >= 0 and (init_w + w) < weight and (init_w + w) >= 0:
                        i_pos = base_i + (init_h + h) * s12 + (init_w + w) * s13
                        w_pos = base_w + ((1 - r) * h + r * (kh - h - 1)) * s22 + ((1 - r) * w + r * (kw - w - 1)) * s23
                        # w_pos = base_w + ((1 - r) * k + r * (kw - k - 1)) * s2[2]
                        out[o_pos] = out[o_pos] + input[i_pos] * weight[w_pos]


tensor_conv2d = njit(parallel=True, fastmath=True)(_tensor_conv2d)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 2D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
