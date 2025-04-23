import tvm
from tvm import relay
from mcu_conversion.compile_time_diff.autodiff import AutoDiff

def compute_autodiff(mod, keep_prediction=True, filter_fn=lambda x, 
                     g: True, sparse_op_idx={}, return_sparse_meta_info=False,
                     return_gradient_tensors=False, int8_bp=False):
    '''
    Perform the auto diff operation at compile time, to generate the backward graph
    which will be deployed on the MCU
    '''
    if isinstance(mod, relay.Function):
        mod = tvm.IRModule.from_expr(mod)
    assert isinstance(mod, tvm.IRModule)
    mod = relay.transform.InferType()(mod)

    prediction = mod["main"].body

    ad = AutoDiff(debug=False, sparse_op_idx=sparse_op_idx)
    ad.int8_gradients = int8_bp
    ad.compute_gradients(mod["main"].body)

    names, gradients = ad.obtain_grads(filter_fn=filter_fn)
    expression  = relay.Function(ad.vars, relay.Tuple(gradients))

    if keep_prediction:
        expression = relay.Function(ad.vars, relay.Tuple([prediction] + gradients))
        names = ["fwd@output"] + names

    # Make an IR module for the backward pass
    backward_mod = tvm.IRModule.from_expr(expression)

    # Some post processing
    backward_mod = relay.transform.InferType()(backward_mod)

    backward_mod = tvm.transform.Sequential(
        [
            relay.transform.DeadCodeElimination(),
            relay.transform.ToGraphNormalForm(),
            relay.transform.FoldConstant(),
            relay.transform.SimplifyExpr()
        ]
    )(backward_mod)

    if return_gradient_tensors:
        return backward_mod, names, gradients
    
    if return_sparse_meta_info:
        return backward_mod, names, ad.sparse_update_meta_info
    else:
        return backward_mod, names
