from ...operation_gradient_mapping import register_gradient
import tvm.relay as relay
from tvm.relay.op.transform import expand_dims


@register_gradient("mcumean")
def mcumean_grad(orig, grad):
    """Returns grad broadcasted to data dims"""
    
    data, axis = orig.args[0], _get_reduce_axis(orig)
    shape = data.checked_type.concrete_shape
    dtype = "float32"
    grad, data = [relay.cast(_, dtype) for _ in (grad, data)]

    if axis is None:
        axis = list(range(len(shape)))

    if not orig.attrs.keepdims:
        grad = _unreduce_expand(grad, axis)
    mult = 1.0
    for a in axis:
        mult /= shape[a]

    return [
        grad
        * relay.const(mult, dtype=dtype)
        * relay.ones_like(data)
    ]

def _get_reduce_axis(call):
    """Helper function that returns the reduce axis of the call as plain python ints."""
    x, axis = call.args[0], call.attrs.axis
    shape = x.checked_type.concrete_shape

    # should never exclude when axis is None
    assert not (axis is None and call.attrs.exclude)

    if axis is None:
        return None

    # convert to non-negative integers and sort
    axis = sorted([ax if ax >= 0 else len(shape) + ax for ax in map(int, axis)])
    if call.attrs.exclude:
        axis = [ax for ax in range(len(shape)) if ax not in axis]
    return axis


def _unreduce_expand(x, axis):
    """Helper function that returns x expanded on the reduced dimensions in axis."""
    # assume axis is sorted nonnegative ints
    for ax in axis:
        x = expand_dims(x, ax)
    return x