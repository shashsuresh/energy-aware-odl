from tvm import relay
from tvm.relay.op.reduce import sum as _sum
from tvm.relay.op.tensor import exp

from ...operation_gradient_mapping import register_gradient, PROJECT_LEVEL
from .mean_grad import _get_reduce_axis, _unreduce_expand

@register_gradient("nn.cross_entropy_with_logits")
def cross_entropy_with_logits_grad(orig, grad):
    '''Computes the gradient of the cross_entropy_with_logits operation'''

    # Get relevant parameters
    x, y = orig.args

    # Compute batch size
    batch_size = relay.const(int(x.checked_type.shape[0]))

    # Compute gradient
    grad  = grad / batch_size.astype(x.checked_type.dtype)
    return [-grad * y, -grad * x]

@register_gradient("nn.log_softmax", level=PROJECT_LEVEL)
def log_softmax_grad(orig, grad):
    '''Computes the gradient of the `log_softmax` operation'''

    # Compute gradient
    output = grad - _sum(grad, axis=orig.attrs.axis, keepdims=True) * exp(orig)

    return [output]

@register_gradient("cast", level=30)
def cast_grad(orig, grad):
    '''Computes the gradient of the `cast` operation'''
    return [grad]

@register_gradient("reshape", level=PROJECT_LEVEL)
def reshape_grad(orig, grad):
    """Gradient of reshape"""
    return [relay.reshape_like(grad, orig.args[0])]


##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
###################### THE DEFINITIONS BELOW ARE NOT USED FOR MOBILENETV2 AND HENCE HAVE NOT BEEN TESTED #####################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################

# Using the back-up version for now, need to check whether value is correct tho...
@register_gradient("nn.dense")
def dense_grad(orig, grad):
    x, w = orig.args

    return [
        # Gradient of loss wrt. input
        relay.collapse_sum_like(
            relay.nn.dense(grad, relay.transpose(w), units=w.checked_type.shape[1]), x
        ),

        # Gradient of loss wrt. weights
        relay.collapse_sum_like(
            relay.nn.dense(
                relay.transpose(grad), relay.transpose(x), units=x.checked_type.shape[1]
            ),
            w,
        ),
    ]


@register_gradient("nn.bias_add")
def bias_add_grad(orig, grad):
    '''
    Computes the gradient of the bias_add operation
    '''
    return [
        grad,
        _sum(grad, orig.attrs.axis, keepdims=False, exclude=True)
    ]

@register_gradient("clip")
def clip_grad(orig, grad):
    '''
    Computes the gradient of the clip operation
    '''
    x = orig.args[0]
    a_min = orig.attrs.get_int("a_min")
    a_max = orig.attrs.get_int("a_max")
    zeros = relay.zeros_like(x)
    ones = relay.ones_like(x)
    a_mins = relay.zeros(x.checked_type.shape, dtype=x.checked_type.dtype) * relay.const(a_min)
    a_maxs = relay.ones(x.checked_type.shape, dtype=x.checked_type.dtype) * relay.const(a_max)
    return [relay.where(relay.less(x, a_mins), zeros, relay.where(relay.less(a_maxs, x), zeros, ones * grad))]

@register_gradient("mean")
def mean_grad(orig, grad):
    """Returns grad broadcasted to data dims"""
    data, axis = orig.args[0], _get_reduce_axis(orig)
    shape = data.checked_type.concrete_shape
    if axis is None:
        axis = list(range(len(data.checked_type.concrete_shape)))
    if not orig.attrs.keepdims:
        grad = _unreduce_expand(grad, axis)
    mult = 1.0
    for a in axis:
        mult /= shape[a]
    # return [broadcast_to_like(grad * const(mult, dtype=data.checked_type.dtype), data)]
    return [grad * relay.const(mult, dtype=data.checked_type.dtype)]


@register_gradient("nn.relu")
def relu_grad(orig, grad):
    """Returns grad * (select(x < 0, 0, 1))."""
    x = orig.args[0]
    zeros = relay.zeros_like(x)
    return [
        relay.op.transform.where(relay.less(x, zeros), zeros, grad),
    ]

@register_gradient("add", level=PROJECT_LEVEL)
def add_grad(orig, grad):
    """Returns [grad, grad]"""
    return [
        relay.collapse_sum_like(grad, orig.args[0]),
        relay.collapse_sum_like(grad, orig.args[1]),
    ]