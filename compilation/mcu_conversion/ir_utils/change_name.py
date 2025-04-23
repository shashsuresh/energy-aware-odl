import tvm

from tvm import relay
from tvm.relay import ExprMutator

class ChangeName(ExprMutator):
    '''
    A class to help change names of IR model elements
    '''
    def visit_var(self, var):
        '''
        Rename vars
        '''
        vname = var.name_hint
        shape = var.type_annotation.shape
        dtype = var.type_annotation.dtype
        if vname[0].isdigit():
            return relay.var("v" + vname, shape=shape, dtype=dtype)
        return var

    def visit_call(self, call):
        '''
        Return modified version of `call`
        '''
        new_fn = self.visit(call.op)
        args = []
        for arg in call.args:
            args.append(self.visit(arg))
        return relay.Call(new_fn, args, call.attrs)