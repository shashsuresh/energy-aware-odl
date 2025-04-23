import tvm

def ir_var_to_list(temp):
    '''
    Converts IR vars into a list
    '''
    if isinstance(temp, tvm.ir.container.Array):
        return list([ir_var_to_list(_) for _ in temp])
    elif isinstance(temp, tvm.tir.expr.IntImm):
        return int(temp)
    elif isinstance(temp, tvm.runtime.container.String):
        return str(temp)
    elif isinstance(temp, (float, int, str)):
        return temp
    elif temp is None:
        return temp
    if isinstance(temp, (tuple, list)):
        return list(ir_var_to_list(_) for _ in temp)
    else:
        raise NotImplementedError(type(temp), temp, isinstance(temp, int))