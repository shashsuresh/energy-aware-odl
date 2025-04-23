import tvm.relay as relay
from ..helper_functions import get_call_shape
from ...operation_gradient_mapping import register_gradient

#@register_gradient("nn.mcuadd")
def mcuadd_int8_grad(orig, grad):
    '''
    Int 8 gradient computation for an addition layer for deployment on MCUs
    '''
    # cast to 32bits for backward computation
    # new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    new_inputs = orig.args
    x1, x2, zero_x1, zero_x2, scale_x1, scale_x2, zero_y, scale_y = new_inputs
    ograd = grad
    grad_dtype = get_call_shape(grad)

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

    # print(grad_dtype, check_call_shape(grad_x1), check_call_shape(grad_x2))
    # input()
    return [
        relay.cast(grad_x1, grad_dtype),
        relay.cast(grad_x2, grad_dtype),
        # ograd, ograd,
        grad_zero_x1,
        grad_zero_x2,
        relay.zeros_like(scale_x1),
        relay.zeros_like(scale_x2),
        grad_zero_y,
        relay.zeros_like(scale_y),
    ]