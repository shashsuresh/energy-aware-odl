import torch.nn as nn
import tvm
from tvm import relay
from quantization.quantized_operations import QuantizedConv2D, QuantizedMBBlock
from mcu_conversion.compile_time_diff.mcu_operations.conv import make_mcuconv, extract_mcuconv2d_params
from mcu_conversion.compile_time_diff.mcu_operations.add import make_mcu_add, extract_mcuadd_params

def nn_to_ir(model, input_res=[1, 3, 80, 80]):
    '''
    Converts a PyTorch nn module to an TVM IR
    '''
    data = relay.var("input", shape=input_res, dtype="int8")
    tot_args = [data]
    tot_params = {}
    out = data

    if isinstance(model[0], QuantizedConv2D):
        out = convert_QuantizedConv2d(model[0], data, tot_args, tot_params)

    op_idx = 1
    for sub_n in model[1]:
        assert isinstance(sub_n, QuantizedMBBlock), f"sub_n instance found: {type(sub_n)} | Expected {QuantizedMBBlock}"
        out = convert_QuantizedMBBlock(sub_n, out, tot_args, tot_params, op_idx)
        op_idx+=1
    
    n = model[2]
    if isinstance(n, QuantizedConv2D):
        convert_QuantizedConv2d(n, out, tot_args, tot_params, str(op_idx))
        op_idx+=1

    out = relay.mcumean(out, axis=[2,3], keepdims=True)
    assert isinstance(model[-1], QuantizedConv2D),  f"sub_n instance found: {type(model[-1])} | Expected {QuantizedConv2D}"
    n = model[-1]

    out = convert_QuantizedConv2d(n, out, tot_args, tot_params, op_idx)

    return out, tot_args, tot_params, op_idx



def convert_QuantizedConv2d(layer, data, tot_args, tot_params, op_idx="0"):
    '''
    Converts a `QuantizedConv2d` instance (in PyTorch) to 
    the corresponding IR representation and updates the relevant 
    parameters
    '''
    out, args = make_mcuconv(data, prefix=f"{op_idx}_", in_channels=layer.in_channels,
                             out_channels=layer.out_channels, padding=layer.padding,
                             strides=layer.stride, groups=layer.groups,
                             kernel_size=layer.kernel_size)

    # Update args
    tot_args += list(args)

    # Update params
    params = extract_mcuconv2d_params(layer, args)
    tot_params.update(params)

    return out

def convert_QuantizedMBBlock(sub_n, out, tot_args, tot_params, op_idx):
    '''
    Converts a `QuantizedMBBlock` instance
    to the corresponding PyTorch representation
    and updates relevant parameters 
    '''
    assert isinstance(sub_n.conv, nn.Sequential)
    og_out = out
    for idx2, n in enumerate(sub_n.conv):
        assert isinstance(n, QuantizedConv2D)
        out = convert_QuantizedConv2d(n, data=out, tot_args=tot_args, tot_params=tot_params, op_idx=str(op_idx)+f"_conv_{idx2}")

    # residual handling
    if sub_n.q_add is not None:
        out,args = make_mcu_add(og_out, out, out_channels=n.out_channels, prefix=f"{op_idx}_qadd_")
        if op_idx == 11:
            pass
        tot_args += list(args)

        tot_params.update(extract_mcuadd_params(sub_n.q_add, args))

    return out