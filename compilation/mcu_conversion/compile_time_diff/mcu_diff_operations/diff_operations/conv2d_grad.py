import tvm
from tvm import relay
from tvm.relay.op.reduce import sum as _sum
from tvm.relay.op import nn as _nn
from tvm.relay.op.transform import (
    reshape,
    strided_slice,
    transpose,
    tile
)
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.utils import get_const_tuple


from ...operation_gradient_mapping import register_gradient

@register_gradient("nn.conv2d")
def conv2d_grad(orig, grad):
    """Gradient of conv2d"""
    
    # Get parameters
    attrs = orig.attrs
    data, weight = orig.args
    data_shape = get_const_tuple(data.checked_type.shape)
    weight_shape = get_const_tuple(weight.checked_type.shape)
    _, _, grad_h, grad_w = get_const_tuple(orig.checked_type.shape)
    batch, in_channel, in_h, in_w = data_shape
    out_channel, _, filter_h, filter_w = weight_shape

    # infer output_padding
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(
        get_const_tuple(attrs.padding), (filter_h, filter_w)
    )
    stride_h, stride_w = get_const_tuple(attrs.strides)
    dilation_h, dilation_w = get_const_tuple(attrs.dilation)
    out_h = (grad_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (grad_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    output_padding = (in_h - out_h, in_w - out_w)

    # Some checks to ensure the data is in the correct layout
    assert attrs.data_layout == "NCHW", "only support NCHW data layout"
    assert attrs.kernel_layout == "OIHW", "only support OIHW kernel layout"
    assert attrs.out_layout in ["", "NCHW"], "only support NCHW output layout"

    # gradient wrt input tensor
    backward_data = _nn.conv2d_transpose(
        grad,
        weight,
        strides=attrs.strides,
        padding=attrs.padding,
        dilation=attrs.dilation,
        groups=attrs.groups,
        output_padding=output_padding,
        kernel_size=(filter_h, filter_w),
        channels=in_channel,
    )
    grad = tvm.relay.tile(grad, [1, in_channel // attrs.groups, 1, 1])
    grad = reshape(grad, [-1, 1, 0, 0])  # batch * oc * ic // groups, 1, oh, ow
    data = reshape(data, [1, -1, 0, 0])  # 1, batch * ic, ih, iw

    # calculate gradient of weights
    backward_weight = _nn.conv2d(
        data,
        grad,
        strides=attrs.dilation,
        padding=attrs.padding,
        dilation=attrs.strides,
        groups=in_channel * batch,
        kernel_size=(grad_h, grad_w),
        channels=batch * out_channel * in_channel // attrs.groups,
    )

    # infer shape of backward_weight
    padded_weight_grad_h = (
        in_h - (grad_h - 1) * stride_h - 1 + fpad_top + fpad_bottom
    ) // dilation_h + 1
    padded_weight_grad_w = (
        in_w - (grad_w - 1) * stride_w - 1 + fpad_left + fpad_right
    ) // dilation_w + 1
    backward_weight = reshape(
        backward_weight,
        [
            batch,
            in_channel // attrs.groups,
            out_channel,
            padded_weight_grad_h,
            padded_weight_grad_w,
        ],
    )
    backward_weight = _sum(backward_weight, axis=0)
    backward_weight = transpose(backward_weight, [1, 0, 2, 3])

    assert padded_weight_grad_h >= filter_h
    assert padded_weight_grad_w >= filter_w
    if padded_weight_grad_h > filter_h or padded_weight_grad_w > filter_w:
        backward_weight = strided_slice(
            backward_weight,
            begin=[0, 0, 0, 0],
            end=[out_channel, in_channel // attrs.groups, filter_h, filter_w],
        )

    return [backward_data, backward_weight]


@register_gradient("nn.mcuconv2d")
def mcunetconv2d_grad(orig, grad):
    '''
    Gradient of the conv2d layer for MCU net
    '''
    # x, y = orig.args
    o_data, o_weight, o_bias, o_zx, o_zy, o_scale = orig.args
    data_shape = get_const_tuple(o_data.checked_type.shape)
    weight_shape = get_const_tuple(o_weight.checked_type.shape)

    # cast to int32 during backward computation
    ograd = grad
    new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    grad = relay.cast(grad, "float32")
    data, weight, bias, zx, zy, scale = new_inputs

    scale = relay.reshape(scale, newshape=[1, -1, 1, 1])

    backward_zero_y = relay.sum(grad, axis=1, exclude=True)
    grad = grad * scale
    backward_bias = relay.sum(grad, axis=1, exclude=True)
    """Gradient of conv2d"""
    attrs = orig.attrs

    _, _, grad_h, grad_w = get_const_tuple(orig.checked_type.shape)
    batch, in_channel, in_h, in_w = data_shape
    out_channel, _, filter_h, filter_w = weight_shape

    # infer output_padding
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(
        get_const_tuple(attrs.padding), (filter_h, filter_w)
    )
    stride_h, stride_w = get_const_tuple(attrs.strides)
    dilation_h, dilation_w = get_const_tuple(attrs.dilation)
    out_h = (grad_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (grad_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    output_padding = (in_h - out_h, in_w - out_w)

    assert attrs.data_layout == "NCHW", "only support NCHW data layout"
    assert attrs.kernel_layout == "OIHW", "only support OIHW kernel layout"
    assert attrs.out_layout in ["", "NCHW"], "only support NCHW output layout"

    backward_data = _nn.conv2d_transpose(
        grad,
        weight,
        strides=attrs.strides,
        padding=attrs.padding,
        dilation=attrs.dilation,
        groups=attrs.groups,
        output_padding=output_padding,
        kernel_size=(filter_h, filter_w),
        channels=in_channel,
    )
    grad = tile(grad, [1, in_channel // attrs.groups, 1, 1])
    grad = reshape(grad, [-1, 1, 0, 0])  # batch * oc * ic // groups, 1, oh, ow
    data = reshape(data, [1, -1, 0, 0])  # 1, batch * ic, ih, iw

    backward_weight = _nn.conv2d(
        data,
        grad,
        strides=attrs.dilation,
        padding=attrs.padding,
        dilation=attrs.strides,
        groups=in_channel * batch,
    )
    # infer shape of backward_weight
    padded_weight_grad_h = (
        in_h - (grad_h - 1) * stride_h - 1 + fpad_top + fpad_bottom
    ) // dilation_h + 1
    padded_weight_grad_w = (
        in_w - (grad_w - 1) * stride_w - 1 + fpad_left + fpad_right
    ) // dilation_w + 1
    backward_weight = reshape(
        backward_weight,
        [
            batch,
            in_channel // attrs.groups,
            out_channel,
            padded_weight_grad_h,
            padded_weight_grad_w,
        ],
    )
    backward_weight = _sum(backward_weight, axis=0)
    backward_weight = transpose(backward_weight, [1, 0, 2, 3])

    assert padded_weight_grad_h >= filter_h
    assert padded_weight_grad_w >= filter_w
    if padded_weight_grad_h > filter_h or padded_weight_grad_w > filter_w:
        backward_weight = strided_slice(
            backward_weight,
            begin=[0, 0, 0, 0],
            end=[out_channel, in_channel // attrs.groups, filter_h, filter_w],
        )

    return [
        backward_data,
        backward_weight,
        backward_bias,
        relay.zeros_like(o_zx),
        relay.zeros_like(o_zy),
        relay.zeros_like(o_scale),
    ]

@register_gradient("nn.mcuadd")
def mcunetconv2d_add_grad(orig, grad):
    '''
    Gradient of the elementwise addition layer for deployment on MCUs
    '''

    # cast to 32bits for backward computation
    new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    x1, x2, zero_x1, zero_x2, scale_x1, scale_x2, zero_y, scale_y = new_inputs
    grad = relay.cast(grad, "float32")

    grad_zero_y = relay.sum(grad)
    new_scale_y = relay.reshape(scale_y, newshape=[1, -1, 1, 1])
    grad_sum = grad / new_scale_y

    new_scale_x1 = relay.reshape(scale_x1, newshape=[1, -1, 1, 1])
    grad_x1 = grad_sum * new_scale_x1

    new_scale_x2 = relay.reshape(scale_x2, newshape=[1, -1, 1, 1])
    grad_x2 = grad_sum * new_scale_x2

    grad_zero_x1 = -relay.sum(grad_x1)
    grad_zero_x2 = -relay.sum(grad_x2)
    return [
        grad_x1,
        grad_x2,
        grad_zero_x1,
        grad_zero_x2,
        relay.zeros_like(scale_x1),
        relay.zeros_like(scale_x2),
        grad_zero_y,
        relay.zeros_like(scale_y),
    ]

def sparse_depth_wise_mcunetconv2d_grad(orig, grad, topk=None):
    '''
    Sparse gradient of a depthwise MCUNet Conv2d layer
    '''
    # x, y = orig.args
    o_data, o_weight, o_bias, o_zx, o_zy, o_scale = orig.args
    data_shape = get_const_tuple(o_data.checked_type.shape)
    weight_shape = get_const_tuple(o_weight.checked_type.shape)

    # cast to int32 during backward computation
    new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    grad = relay.cast(grad, "float32")
    data, weight, bias, zx, zy, scale = new_inputs

    scale = relay.reshape(scale, newshape=[1, -1, 1, 1])

    grad = grad * scale
    backward_bias = relay.sum(grad, axis=1, exclude=True)

    attrs = orig.attrs
    grad_n, grad_c, grad_h, grad_w = get_const_tuple(orig.checked_type.shape)
    batch, in_channel, in_h, in_w = data_shape
    out_channel, _, filter_h, filter_w = weight_shape

    # infer output_padding
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(
        get_const_tuple(attrs.padding), (filter_h, filter_w)
    )
    stride_h, stride_w = get_const_tuple(attrs.strides)
    dilation_h, dilation_w = get_const_tuple(attrs.dilation)
    out_h = (grad_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (grad_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    output_padding = (in_h - out_h, in_w - out_w)

    assert attrs.data_layout == "NCHW", "only support NCHW data layout"
    assert attrs.kernel_layout == "OIHW", "only support OIHW kernel layout"
    assert attrs.out_layout in ["", "NCHW"], "only support NCHW output layout"

    # Gradient of input wrt loss
    backward_data = _nn.conv2d_transpose(
        grad,
        weight,
        strides=attrs.strides,
        padding=attrs.padding,
        dilation=attrs.dilation,
        groups=attrs.groups,
        output_padding=output_padding,
        kernel_size=(filter_h, filter_w),
        channels=in_channel,
    )

    tmp_inc = in_channel
    tmp_ouc = out_channel
    groups = attrs.groups
    if topk is not None:
        tmp_inc = round(topk * in_channel)
        tmp_ouc = round(topk * out_channel)
        data = relay.strided_slice(
            data,
            begin=relay.const([0, 0, 0, 0]),
            end=relay.const([batch, tmp_inc, in_h, in_w]),
        )
        grad = relay.strided_slice(
            grad,
            begin=relay.const([0, 0, 0, 0]),
            end=relay.const([grad_n, tmp_ouc, grad_h, grad_w]),
        )
        groups = tmp_inc

    grad = tvm.relay.tile(grad, [1, tmp_inc // groups, 1, 1])
    grad = reshape(grad, [-1, 1, 0, 0])  # batch * oc * ic // groups, 1, oh, ow
    data = reshape(data, [1, -1, 0, 0])  # 1, batch * ic, ih, iw

    # Gradient of weights
    backward_weight = _nn.conv2d(
        data,
        grad,
        strides=attrs.dilation,
        padding=attrs.padding,
        dilation=attrs.strides,
        groups=tmp_inc * batch,
    )

    # infer shape of backward_weight
    padded_weight_grad_h = (
        in_h - (grad_h - 1) * stride_h - 1 + fpad_top + fpad_bottom
    ) // dilation_h + 1
    padded_weight_grad_w = (
        in_w - (grad_w - 1) * stride_w - 1 + fpad_left + fpad_right
    ) // dilation_w + 1
    backward_weight = reshape(
        backward_weight,
        [
            batch,
            tmp_inc // groups,
            tmp_ouc,
            padded_weight_grad_h,
            padded_weight_grad_w,
        ],
    )

    backward_weight = _sum(backward_weight, axis=0)
    backward_weight = transpose(backward_weight, [1, 0, 2, 3])

    assert padded_weight_grad_h >= filter_h
    assert padded_weight_grad_w >= filter_w
    if padded_weight_grad_h > filter_h or padded_weight_grad_w > filter_w:
        backward_weight = strided_slice(
            backward_weight,
            begin=[0, 0, 0, 0],
            end=[tmp_ouc, tmp_inc // groups, filter_h, filter_w],
        )

    return [
        backward_data,
        backward_weight,
        backward_bias,
        relay.zeros_like(o_zx),
        relay.zeros_like(o_zy),
        relay.zeros_like(o_scale),
    ]

def sparse_in_channel_mcunetconv2d_grad(orig, grad, topk=None):
    o_data, o_weight, o_bias, o_zx, o_zy, o_scale = orig.args
    data_shape = get_const_tuple(o_data.checked_type.shape)
    weight_shape = get_const_tuple(o_weight.checked_type.shape)

    # cast to int32 during backward computation
    ograd = grad
    new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    grad = relay.cast(grad, "float32")
    data, weight, bias, zx, zy, scale = new_inputs

    scale = relay.reshape(scale, newshape=[1, -1, 1, 1])

    backward_zero_y = relay.sum(grad, axis=1, exclude=True)
    grad = grad * scale
    backward_bias = relay.sum(grad, axis=1, exclude=True)
    """Gradient of conv2d"""
    attrs = orig.attrs
    grad_n, grad_c, grad_h, grad_w = get_const_tuple(orig.checked_type.shape)
    batch, in_channel, in_h, in_w = data_shape
    out_channel, _, filter_h, filter_w = weight_shape

    # infer output_padding
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(
        get_const_tuple(attrs.padding), (filter_h, filter_w)
    )
    stride_h, stride_w = get_const_tuple(attrs.strides)
    dilation_h, dilation_w = get_const_tuple(attrs.dilation)
    out_h = (grad_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (grad_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    output_padding = (in_h - out_h, in_w - out_w)

    assert attrs.data_layout == "NCHW", "only support NCHW data layout"
    assert attrs.kernel_layout == "OIHW", "only support OIHW kernel layout"
    assert attrs.out_layout in ["", "NCHW"], "only support NCHW output layout"

    backward_data = _nn.conv2d_transpose(
        grad,
        weight,
        strides=attrs.strides,
        padding=attrs.padding,
        dilation=attrs.dilation,
        groups=attrs.groups,
        output_padding=output_padding,
        kernel_size=(filter_h, filter_w),
        channels=in_channel,
    )

    # o_data = data
    # o_grad = grad
    tmp_inc = in_channel
    tmp_ouc = out_channel
    if topk is not None:
        tmp_inc = round(topk * in_channel)
        assert attrs.groups == 1
        data = relay.strided_slice(
            data,
            begin=relay.const([0, 0, 0, 0]),
            end=relay.const([batch, tmp_inc, in_h, in_w]),
        )

    grad = relay.tile(grad, [1, tmp_inc // attrs.groups, 1, 1])
    grad = reshape(grad, [-1, 1, 0, 0])  # batch * oc * ic // groups, 1, oh, ow
    data = reshape(data, [1, -1, 0, 0])  # 1, batch * ic, ih, iw

    backward_weight = _nn.conv2d(
        data,
        grad,
        strides=attrs.dilation,
        padding=attrs.padding,
        dilation=attrs.strides,
        groups=tmp_inc * batch,
    )

    # infer shape of backward_weight
    padded_weight_grad_h = (
        in_h - (grad_h - 1) * stride_h - 1 + fpad_top + fpad_bottom
    ) // dilation_h + 1
    padded_weight_grad_w = (
        in_w - (grad_w - 1) * stride_w - 1 + fpad_left + fpad_right
    ) // dilation_w + 1
    backward_weight = reshape(
        backward_weight,
        [
            batch,
            tmp_inc // attrs.groups,
            tmp_ouc,
            padded_weight_grad_h,
            padded_weight_grad_w,
        ],
    )

    backward_weight = _sum(backward_weight, axis=0)
    backward_weight = transpose(backward_weight, [1, 0, 2, 3])

    assert padded_weight_grad_h >= filter_h
    assert padded_weight_grad_w >= filter_w
    if padded_weight_grad_h > filter_h or padded_weight_grad_w > filter_w:
        backward_weight = strided_slice(
            backward_weight,
            begin=[0, 0, 0, 0],
            end=[tmp_ouc, tmp_inc // attrs.groups, filter_h, filter_w],
        )

    backward_zero_x = -relay.sum(backward_data, axis=1, exclude=True)

    return [
        backward_data,
        backward_weight,
        backward_bias,
        relay.zeros_like(o_zx),
        relay.zeros_like(o_zy),
        relay.zeros_like(o_scale),
    ]