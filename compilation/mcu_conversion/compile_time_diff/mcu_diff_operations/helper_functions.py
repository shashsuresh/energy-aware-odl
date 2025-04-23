import tvm
from tvm import relay

def get_call_dtype(call):
    '''
    Return the dtype of the passed function to be called
    '''

    #Get all variables needed for FN call
    variables = relay.analysis.all_vars(call)

    # Make a Relay function
    expression = relay.Function(variables, call)

    #Make the expression an IR module and infer type
    mod = tvm.IRModule.from_expr(expression)
    mod = relay.transform.InferType()(mod)
    
    # Extract the main function from the module
    expression = mod["main"]
    
    # Return the type of the return value of extracted function
    return expression.body.checked_type.dtype

def get_call_shape(call):
    '''
    Return the shape of the passed function to be called
    '''

    #Get all variables needed for FN call
    variables = relay.analysis.all_vars(call)

    # Make a Relay function
    expression = relay.Function(variables, call)

    #Make the expression an IR module and infer type
    mod = tvm.IRModule.from_expr(expression)
    mod = relay.transform.InferType()(mod)
    
    # Extract the main function from the module
    expression = mod["main"]

    # Return the shape of the return value of the extracted function
    return expression.body.checked_type.shape

def get_call_info(call):
    '''
    Return all information on the passed function to be called
    '''

    # Create expression from call and relevant variables
    expression = relay.Function(relay.analysis.all_vars(call), call)

    # Create IR module from expression
    mod = tvm.IRModule.from_expr(expression)

    # Infer the type
    mod = relay.transform.InferType()(mod)

    # Extract information from the constructed module
    return mod["main"].body.checked_type

def get_weight_scales(w, n_bits=8, axis=1):
    '''
    Return the scales of the weights (part of QAS)

    `TODO` This is probably where we need to manually cast to float32
    to make our models compatible with `codegen`
    '''
    wmax = relay.max(relay.abs(w), axis=axis, keepdims=True)
    dtype = get_call_dtype(wmax)
    return wmax / relay.const(2 ** (n_bits - 1) - 1, dtype=dtype)