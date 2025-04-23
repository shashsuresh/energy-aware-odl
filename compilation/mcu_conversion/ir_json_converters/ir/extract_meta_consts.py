from tvm import relay
from tvm.relay.expr_functor import ExprMutator

class ExtractMetaConstants(ExprMutator):
    '''
    Class to facilitate extraction of Meta constants
    '''
    def __init__(self):
        super().__init__()
        self.constants = []

    def visit_constant(self, const: relay.expr.Constant):
        '''
        Overload of the visit constant function, to extract meta constants
        '''
        new_const = relay.const(const.data.numpy())
        np_data = const.data.numpy()
        if np_data.size == 1:
            value = np_data.item()
            new_const = relay.const(value, dtype=str(np_data.dtype))
        if "meta" in str(const):
            self.constants.append(np_data)
        return new_const

    def extract_constants(self, func):
        '''
        Extracts meta constants from the relay IR
        '''
        expr = self.visit(func)
        return expr, self.constants