from tvm import relay
from tvm.relay.op import reg

@reg.register_legalize("mcumean", level=30)
def mcu_mean_calc(attrs, inputs, types):
    '''
    Calculates the mean of a provided input
    '''
    # cast to float for calculation
    new_x = relay.cast(inputs[0], "float32")
    # calculate mean
    out = relay.mean(new_x, axis=attrs.axis, keepdims=attrs.keepdims)
    # Post processing
    out = relay.round(out)
    return relay.cast(out, types[-1].dtype)