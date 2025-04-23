import tvm
from tvm import relay
from tvm.relay.op import reg
import numpy as np

# Doing this signals to the IR compiler to convert this into something the target can implement
@reg.register_legalize("nn.mcuconv2d", level=20)
def mcu_conv2d(attrs, inputs, types):
    '''
    The convolution 2d function that will be deployed on the MCU

    The IR compiler will convert this into a function that can deploy it on a target device
    '''

    # Cast function inputs
    x, weight, bias, zx, zy, scale = [relay.cast(_, "float32") for _ in inputs]

    # Perform convolution
    conv_result = relay.nn.conv2d(
        x - zx, weight, attrs.strides, attrs.padding, attrs.dilation, attrs.groups
    )
    # Add bias
    conv_result = relay.nn.bias_add(conv_result, bias)

    # Update the scale
    scale = relay.reshape(scale, newshape=[1, -1, 1, 1])

    # Apply the scale
    conv_result = conv_result * scale

    # Round and add zero bias to quantized value
    int32_layer_output = relay.round(conv_result) + relay.cast(zy, "float32")

    # Cast and return
    return relay.cast(int32_layer_output, types[-1].dtype)


def make_mcuconv(
    features,
    prefix="",
    in_channels=3,
    out_channels=3,
    kernel_size=3,
    strides=1,
    padding=1,
    groups=1,
    param_dtype="int8"):
    '''
    This function converts a QuantizedConv2D into a representation
    that can be interpreted by MCUs using apache TVM
    '''

    # Set kernel size
    if isinstance(kernel_size, (list, tuple)):
        kernel_size_new = kernel_size[0]
    else:
        kernel_size_new = kernel_size
    
    # Define weights
    weight = relay.var(f"{prefix}weight", shape=[out_channels, in_channels // groups, kernel_size_new, kernel_size_new],dtype=param_dtype)

    # Define bias
    bias = relay.var(f"{prefix}bias", shape=[out_channels,], dtype="int32")

    # Define zero points
    zero_x = relay.var(f"{prefix}zero_x", shape=[1,], dtype=param_dtype)

    zero_y = relay.var(f"{prefix}zero_y", shape=[1,], dtype=param_dtype)

    # Define scale
    scale = relay.var(f"{prefix}scale", shape=[out_channels,], dtype="float32")

    # Create relay conv2d
    output = relay.nn.mcuconv2d(features, weight, bias,
                             zero_x, zero_y, scale,
                             strides=strides, padding=padding,
                             groups=groups)

    # Combine params
    params = (weight, bias, zero_x, zero_y, scale)

    return (relay.nn.mcutruncate(output), params)

def extract_mcuconv2d_params(module, args, param_dtype="int8"):
    '''
    Extract conv2d layer parameters
    '''
    params = {}
    # weight
    vname = args[0].name_hint
    vtensor = module.weight.detach().numpy().astype(param_dtype)
    params[vname] = tvm.nd.array(vtensor)

    # bias
    vname = args[1].name_hint
    vtensor = module.bias.detach().numpy()
    vtensor_clipped = np.clip(vtensor, -2147483648, 2147483647)
    vtensor = vtensor_clipped.astype("int32")
    params[vname] = tvm.nd.array(vtensor)

    # 0_zero_x
    vname = args[2].name_hint
    vtensor = module.zero_x.detach().view(1).numpy().astype(param_dtype)
    params[vname] = tvm.nd.array(vtensor)

    # 0_zero_y
    vname = args[3].name_hint
    vtensor = module.zero_y.detach().view(1).numpy().astype(param_dtype)
    params[vname] = tvm.nd.array(vtensor)

    # effective_scale
    vname = args[4].name_hint
    vtensor = module.effective_scale.detach().numpy().astype("float32")
    params[vname] = tvm.nd.array(vtensor)

    vs = vname.split("_")[:-1]
    if hasattr(module, "x_scale"):
        vname = "_".join(vs + ["x_scale"])
        vtensor = np.array(module.x_scale).astype("float32")
        params[vname] = tvm.nd.array(vtensor)

    if hasattr(module, "y_scale"):
        vname = "_".join(vs + ["y_scale"])
        vtensor = np.array(module.y_scale).astype("float32")
        params[vname] = tvm.nd.array(vtensor)

    return params
