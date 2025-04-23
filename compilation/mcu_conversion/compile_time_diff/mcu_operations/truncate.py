import tvm
from tvm import relay
from tvm.relay.op import reg
from tvm.relay.op.nn import _make

def truncate_wrt_max(x: relay.expr.Call, threshold_val=127, dtype="int8"):
    '''
    Truncates values above provided threshold, to the threshold
    '''
    
    # Make threshold tensor
    threshold = relay.ones_like(x) * relay.const(threshold_val, dtype=dtype)

    # Any value over the threshold is set to the threshold
    return relay.where(relay.greater(x, threshold), threshold, x)

def truncate_wrt_min(x: relay.expr.Call, threshold_val=127, dtype="int8"):
    '''
    Truncates values above provided threshold, to the threshold
    '''

    # Make threshold tensor
    threshold = relay.ones_like(x) * relay.const(threshold_val, dtype=dtype)
    
    # Any value under the threshold is set to the threshold
    return relay.where(relay.less(x, threshold), threshold, x)

@reg.register_legalize("nn.mcutruncate", level=30)
def mcu_truncate(inputs, types):
    x = inputs[0]
    dtype = types[0].dtype
    int8_result = truncate_wrt_max(x, threshold_val=127, dtype=dtype)
    int8_result = truncate_wrt_max(int8_result, threshold_val=-128, dtype=dtype)
    return relay.cast(int8_result, types[1].dtype)