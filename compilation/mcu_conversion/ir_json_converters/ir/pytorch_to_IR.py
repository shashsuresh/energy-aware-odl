import tvm
from tvm import relay
from mcu_conversion.ir_json_converters.ir.nn_to_ir import nn_to_ir
from mcu_conversion.compile_time_diff.appending_loss import appending_loss
from mcu_conversion.ir_utils.change_name import ChangeName

def pytorch_to_ir(model, input_res=(1,3,80,80), num_classes=0):
    '''
    Converts a pytorch model into IR to prepare the model for MCU
    deployment
    '''
    # Convert convolution layers first
    out, tot_args, export_params, op_idx = nn_to_ir(model, input_res)
    
    real_params = {}
    scale_params = {}

    for k, v in export_params.items():
        # If the key begins with a digit, append v
        if k[0].isdigit():
            k = "v" + k
        # Scales in `scale_params`, other values in `real_params`
        if k.endswith("x_scale") or k.endswith("y_scale"):
            scale_params[k] = float(v.numpy())
        else:
            real_params[k] = v

    expression = relay.Function(tot_args, out)

    # Perform the modifications necessary
    new_expression = ChangeName().visit(expression)
    
    # Create a forward module after the modifications have been applied
    forward_model = tvm.IRModule.from_expr(new_expression)

    # Add the softmax and loss functions if required

    if num_classes <= 0:
        forward_model = relay.transform.InferType()(forward_model)
        return forward_model, real_params, scale_params, op_idx
    else:

        out = relay.reshape(out, newshape=[0,0])
        out = relay.cast(out, dtype="float32")
        expression = relay.Function(tot_args, out)

        new_expression = ChangeName().visit(expression)
        model = tvm.IRModule.from_expr(new_expression)
        mod, _, _ = appending_loss(model, real_params, "label", label_shape=[1, num_classes])
        forward_model_with_loss = relay.transform.InferType()(mod)

        return forward_model_with_loss, real_params, scale_params, op_idx

