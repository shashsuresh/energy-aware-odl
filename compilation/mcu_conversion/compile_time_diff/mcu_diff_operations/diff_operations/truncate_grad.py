from ...operation_gradient_mapping import register_gradient
import tvm.relay as relay

@register_gradient("nn.mcutruncate")
def mcutruncate_grad(orig, grad):
    '''
    Gradient calculation for the truncate operation
    '''
    new_inputs = [relay.cast(_, "float32") for _ in orig.args]
    x = new_inputs[0]
    dtype = "float32"
    # min = orig.attrs.min
    # max = orig.attrs.max
    min = relay.const(orig.attrs.min, dtype=dtype)
    max = relay.const(orig.attrs.max, dtype=dtype)

    mask1 = relay.greater_equal(x, min)
    mask2 = relay.less_equal(x, max)

    # mask = relay.logical_and(mask1, mask2)
    mask = mask1 * mask2
    zeros = relay.zeros_like(grad)
    # mask = relay.cast(mask, "float32")
    return [
        relay.where(mask, grad, zeros),
    ]