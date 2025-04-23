import tvm
from tvm import relay
from tvm.relay.op.reduce import sum as _sum
from tvm.relay.op import nn as _nn
from tvm.relay.op.transform import (
    reshape,
    strided_slice,
    transpose,
    tile,
)
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.utils import get_const_tuple
from ..helper_functions import get_weight_scales, get_call_dtype
from ...operation_gradient_mapping import register_gradient

def post_process_gradients(backward_data, backward_weight, eps=None):
    '''
    post processing of gradients mainly - rescaling and conversion to int8
    '''
    w_scales = get_weight_scales(backward_weight, n_bits=8)
    x_scales = get_weight_scales(backward_data, n_bits=8)
    if eps is None:
        backward_data = backward_data / x_scales
        backward_weight = backward_weight / w_scales
    else:
        backward_data = relay.cast(backward_data, "float32")
        backward_weight = relay.cast(backward_weight, "float32")
        x_scales = relay.cast(x_scales, "float32")
        w_scales = relay.cast(w_scales, "float32")
        backward_data = backward_data / (x_scales + relay.const(1e-12, dtype="float32"))
        backward_weight = backward_weight / (
            w_scales + relay.const(1e-12, dtype="float32")
        )
    backward_data = relay.cast(backward_data, dtype="int8")
    backward_weight = relay.cast(backward_weight, dtype="int8")
    return backward_data, backward_weight


#@register_gradient("nn.mcuconv2d")
def mcunetconv2d_int8_grad(orig, grad):
    '''
    Gradient computation for MCUNet Conv2D layer in int8
    '''
    # x, y = orig.args
    o_data, o_weight, o_bias, o_zx, o_zy, o_scale = orig.args
    data_shape = get_const_tuple(o_data.checked_type.shape)
    weight_shape = get_const_tuple(o_weight.checked_type.shape)
    data_dtype = o_data.checked_type.dtype
    weight_dtype = o_weight.checked_type.dtype

    # cast to int32 during backward computation
    ograd = grad
    # new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    # grad = relay.cast(grad, "float32")
    new_inputs = orig.args
    data, weight, bias, zx, zy, scale = orig.args

    # scale = relay.reshape(scale, newshape=[1, -1, 1, 1])
    backward_zero_y = relay.sum(grad, axis=1, exclude=True)
    # grad = grad * scale
    dtype = "float32"
    out_dtype = "float32"
    if "int" in str(weight_dtype) and "int" in str(data_dtype):
        out_dtype = "int32"
    # print(data_dtype, weight_dtype )
    tmp_grad = relay.cast(grad, dtype=out_dtype)
    backward_bias = relay.sum(tmp_grad, axis=1, exclude=True)
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

    grad_dtype = get_call_dtype(grad)
    conv_out_dtype = "float32"
    temp_weight = weight
    if grad_dtype != weight_dtype:
        temp_weight = relay.cast(weight, grad_dtype)
    temp_weight_dtype = get_call_dtype(temp_weight)
    if "int" in str(grad_dtype) and "int" in str(temp_weight_dtype):
        conv_out_dtype = "int32"
    backward_data = _nn.conv2d_transpose(
        grad,
        temp_weight,
        strides=attrs.strides,
        padding=attrs.padding,
        dilation=attrs.dilation,
        groups=attrs.groups,
        output_padding=output_padding,
        kernel_size=(filter_h, filter_w),
        channels=in_channel,
        out_dtype=conv_out_dtype,
    )
    grad = tile(grad, [1, in_channel // attrs.groups, 1, 1])
    grad = reshape(grad, [-1, 1, 0, 0])  # batch * oc * ic // groups, 1, oh, ow
    data = reshape(data, [1, -1, 0, 0])  # 1, batch * ic, ih, iw

    conv_out_dtype = "float32"
    temp_data = data
    if data_dtype != grad_dtype:
        temp_data = relay.cast(data, grad_dtype)
    temp_data_dtype = get_call_dtype(temp_data)
    if "int" in str(temp_data_dtype) and "int" in str(grad_dtype):
        conv_out_dtype = "int32"
    backward_weight = _nn.conv2d(
        temp_data,
        grad,
        strides=attrs.dilation,
        padding=attrs.padding,
        dilation=attrs.strides,
        groups=in_channel * batch,
        out_dtype=conv_out_dtype,
    )
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

    backward_zero_x = -relay.sum(backward_data, axis=1, exclude=True)
    backward_data, backward_weight = post_process_gradients(
        backward_data, backward_weight
    )

    return [
        backward_data,
        backward_weight,
        backward_bias,
        relay.zeros_like(o_zx),
        relay.zeros_like(o_zy),
        relay.zeros_like(o_scale),
    ]


def sparse_in_channel_mcunetconv2d_int8grad(orig, grad, topk=None):
    '''
    Gradient computation of a sparse in channel Conv2D layer in MCUNet
    '''
    # x, y = orig.args
    o_data, o_weight, o_bias, o_zx, o_zy, o_scale = orig.args
    data_shape = get_const_tuple(o_data.checked_type.shape)
    weight_shape = get_const_tuple(o_weight.checked_type.shape)
    data_dtype = o_data.checked_type.dtype
    weight_dtype = o_weight.checked_type.dtype

    # cast to int32 during backward computation
    ograd = grad
    new_inputs = orig.args
    data, weight, bias, zx, zy, scale = orig.args

    backward_zero_y = relay.sum(grad, axis=1, exclude=True)
    # grad = grad * scale
    dtype = "float32"
    out_dtype = "float32"
    if "int" in str(weight_dtype) and "int" in str(data_dtype):
        out_dtype = "int32"
    # print(data_dtype, weight_dtype )
    tmp_grad = relay.cast(grad, dtype=out_dtype)
    backward_bias = relay.sum(tmp_grad, axis=1, exclude=True)
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

    grad_dtype = get_call_dtype(grad)
    conv_out_dtype = "float32"

    temp_weight = weight
    if grad_dtype != weight_dtype:
        temp_weight = relay.cast(weight, grad_dtype)
    temp_weight_type = get_call_dtype(temp_weight)
    if "int" in str(grad_dtype) and "int" in str(temp_weight_type):
        conv_out_dtype = "int32"
    backward_data = _nn.conv2d_transpose(
        grad,
        temp_weight,
        strides=attrs.strides,
        padding=attrs.padding,
        dilation=attrs.dilation,
        groups=attrs.groups,
        output_padding=output_padding,
        kernel_size=(filter_h, filter_w),
        channels=in_channel,
        out_dtype=conv_out_dtype,
    )

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

    grad = tile(grad, [1, tmp_inc // attrs.groups, 1, 1])
    grad = reshape(grad, [-1, 1, 0, 0])  # batch * oc * ic // groups, 1, oh, ow
    data = reshape(data, [1, -1, 0, 0])  # 1, batch * ic, ih, iw

    conv_out_dtype = "float32"
    if "int" in str(data_dtype) and "int" in str(grad_dtype):
        conv_out_dtype = "int32"
    temp_data = data
    if data_dtype != grad_dtype:
        temp_data = relay.cast(data, grad_dtype)
    backward_weight = _nn.conv2d(
        temp_data,
        grad,
        strides=attrs.dilation,
        padding=attrs.padding,
        dilation=attrs.strides,
        groups=tmp_inc * batch,
        out_dtype=conv_out_dtype,
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
    #
    backward_data, backward_weight = post_process_gradients(
        backward_data, backward_weight
    )

    return [
        backward_data,
        backward_weight,
        backward_bias,
        relay.zeros_like(o_zx),
        relay.zeros_like(o_zy),
        relay.zeros_like(o_scale),
    ]


def sparse_depth_wise_mcunetconv2d_int8grad(orig, grad, topk=None):
    '''
    Gradient computation for a sparse depthwise conv2d layer in MCUNets
    '''
    # x, y = orig.args
    o_data, o_weight, o_bias, o_zx, o_zy, o_scale = orig.args
    data_shape = get_const_tuple(o_data.checked_type.shape)
    weight_shape = get_const_tuple(o_weight.checked_type.shape)
    data_dtype = o_data.checked_type.dtype
    weight_dtype = o_weight.checked_type.dtype

    # cast to int32 during backward computation
    ograd = grad
    # new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    # grad = relay.cast(grad, "float32")
    new_inputs = orig.args
    data, weight, bias, zx, zy, scale = orig.args

    # scale = relay.reshape(scale, newshape=[1, -1, 1, 1])

    backward_zero_y = relay.sum(grad, axis=1, exclude=True)
    # grad = grad * scale
    dtype = "float32"
    out_dtype = "float32"
    if "int" in str(weight_dtype) and "int" in str(data_dtype):
        out_dtype = "int32"
    # print(data_dtype, weight_dtype )
    tmp_grad = relay.cast(grad, dtype=out_dtype)
    backward_bias = relay.sum(tmp_grad, axis=1, exclude=True)
    """Gradient of conv2d"""
    attrs = orig.attrs

    # _, _, grad_h, grad_w = get_const_tuple(orig.checked_type.shape)
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

    grad_dtype = get_call_dtype(grad)
    conv_out_dtype = "float32"
    if "int" in str(grad_dtype) and "int" in str(weight_dtype):
        conv_out_dtype = "int32"
    temp_weight = weight
    if grad_dtype != weight_dtype:
        temp_weight = relay.cast(weight, grad_dtype)
    backward_data = _nn.conv2d_transpose(
        grad,
        temp_weight,
        strides=attrs.strides,
        padding=attrs.padding,
        dilation=attrs.dilation,
        groups=attrs.groups,
        output_padding=output_padding,
        kernel_size=(filter_h, filter_w),
        channels=in_channel,
        out_dtype=conv_out_dtype,
    )

    # o_data = data
    # o_grad = grad
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

    grad = tile(grad, [1, tmp_inc // groups, 1, 1])
    grad = reshape(grad, [-1, 1, 0, 0])  # batch * oc * ic // groups, 1, oh, ow
    data = reshape(data, [1, -1, 0, 0])  # 1, batch * ic, ih, iw

    conv_out_dtype = "float32"
    if "int" in str(data_dtype) and "int" in str(grad_dtype):
        conv_out_dtype = "int32"
    temp_data = data
    if data_dtype != grad_dtype:
        temp_data = relay.cast(data, grad_dtype)
    backward_weight = _nn.conv2d(
        temp_data,
        grad,
        strides=attrs.dilation,
        padding=attrs.padding,
        dilation=attrs.strides,
        groups=tmp_inc * batch,
        out_dtype=conv_out_dtype,
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

    backward_zero_x = -relay.sum(backward_data, axis=1, exclude=True)
    #
    backward_data, backward_weight = post_process_gradients(
        backward_data, backward_weight
    )

    return [
        backward_data,
        backward_weight,
        backward_bias,
        relay.zeros_like(o_zx),
        relay.zeros_like(o_zy),
        relay.zeros_like(o_scale),
    ]
