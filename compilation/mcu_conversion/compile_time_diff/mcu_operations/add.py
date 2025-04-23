import tvm
import tvm.relay as relay
from tvm.relay.op import reg

@reg.register_legalize("nn.mcuadd", level=30)
def mcu_nn_add(attrs, inputs, types):

    # Cast to float for operations
    new_inputs = [relay.cast(_, "float32") for _ in inputs]
    x1, x2, zero_x1, zero_x2, scale_x1, scale_x2, zero_y, scale_y = new_inputs

    # Operations with scales
    scale_x1 = relay.reshape(scale_x1, newshape=[1, -1, 1, 1])
    scale_x2 = relay.reshape(scale_x2, newshape=[1, -1, 1, 1])
    scale_y = relay.reshape(scale_y, newshape=[1, -1, 1, 1])

    # Perform addition
    x1 = (x1 - zero_x1) * scale_x1
    x2 = (x2 - zero_x2) * scale_x2
    out = x1 + x2

    # Post processing
    int32_res = relay.round(out / scale_y)
    int32_res = int32_res + zero_y
    return relay.cast(int32_res, types[-1].dtype)


def make_mcu_add(num1, num2, out_channels=3, 
                 prefix="", param_dtype="int8"):
    '''Created an element wise add layer for deployment on the target'''

    zero_x1 = relay.var(
    f"{prefix}zero_x1",
    shape=[
        1,
    ],
    dtype=param_dtype,
    )
    zero_x2 = relay.var(
        f"{prefix}zero_x2",
        shape=[
            1,
        ],
        dtype=param_dtype,
    )
    zero_y = relay.var(
        f"{prefix}zero_y",
        shape=[
            1,
        ],
        dtype=param_dtype,
    )

    scale_x1 = relay.var(
        f"{prefix}scale_x1",
        shape=[
            1,
        ],
        dtype="float32",
    )
    scale_x2 = relay.var(
        f"{prefix}scale_x2",
        shape=[
            1,
        ],
        dtype="float32",
    )
    scale_y = relay.var(
        f"{prefix}scale_y",
        shape=[
            1,
        ],
        dtype="float32",
    )

    out = relay.nn.mcuadd(
        num1, num2, zero_x1, zero_x2, scale_x1, scale_x2, zero_y, scale_y
    )
    return (
        relay.nn.mcutruncate(out),
        (zero_x1, zero_x2, scale_x1, scale_x2, zero_y, scale_y),
    )

def extract_mcuadd_params(module, args, param_dtype="int8"):
    '''
    Extracts all parameters from an MCU add module
    '''
    params = {}

    vname = args[0].name_hint
    vtensor = module.zero_x1.detach().view(1).numpy().astype(param_dtype)
    params[vname] = tvm.nd.array(vtensor)

    vname = args[1].name_hint
    vtensor = module.zero_x2.detach().view(1).numpy().astype(param_dtype)
    params[vname] = tvm.nd.array(vtensor)

    vname = args[2].name_hint
    vtensor = module.scale_x1.detach().numpy().astype("float32")
    params[vname] = tvm.nd.array(vtensor)

    vname = args[3].name_hint
    vtensor = module.scale_x2.detach().numpy().astype("float32")
    params[vname] = tvm.nd.array(vtensor)

    vname = args[4].name_hint
    vtensor = module.zero_y.detach().view(1).numpy().astype(param_dtype)
    params[vname] = tvm.nd.array(vtensor)

    vname = args[5].name_hint
    vtensor = module.scale_y.detach().numpy().astype("float32")
    params[vname] = tvm.nd.array(vtensor)

    return params