import tvm
from tvm import relay

def appending_loss(module, params, name="label", label_shape=[1, 10]):
    '''
    Appends the label and loss function to the DAG of an IR model
    '''
    data_inputs = []
    func_args = []
    for v in relay.analysis.all_vars(module["main"]):
        if v.name_hint in params:
            func_args.append(v)
        else:
            data_inputs.append(v)

    ret = module["main"].body

    # add label and loss to the DAG
    label = relay.var(name, shape=label_shape)
    zz = relay.nn.log_softmax(ret)
    loss = relay.nn.cross_entropy_with_logits(zz, label)

    # Update arguments
    new_args = data_inputs + [label] + func_args
    # make function
    func = relay.Function(new_args, loss)
    # Update module
    module = tvm.IRModule.from_expr(func)
    # Extract names
    names = [arg.name_hint for arg in new_args]

    return module, params, names