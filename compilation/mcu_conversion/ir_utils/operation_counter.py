from tvm import relay
from tvm.relay import ExprVisitor
from collections import Counter

class OperationCounter(ExprVisitor):
    def __init__(self, expresion):
        super().__init__()
        self.counter = Counter()
        self.visit(expr=expresion)

    def visit_var(self, var):
        return super().visit_var(var)
    
    def visit_call(self, call):
        self.visit(call.op)
        self.counter[str(call.op)] += 1
        for arg in call.args:
            self.visit(arg)

def ir_scan_ops(expr):
    op = OperationCounter(expr)
    return op.counter