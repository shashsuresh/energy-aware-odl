'''
Generation of backward graph at compile time
'''
import tvm
from tvm import relay
from tvm.relay import ExprVisitor
from tvm.topi.utils import get_const_tuple
from tvm.relay.expr import Tuple, TupleGetItem

from mcu_conversion.compile_time_diff.operation_gradient_mapping import GRADIENT_OPERATION_MAP
from mcu_conversion.compile_time_diff.mcu_diff_operations.int8_diff_operations.conv2d_int_grad import sparse_in_channel_mcunetconv2d_int8grad, sparse_depth_wise_mcunetconv2d_int8grad
from mcu_conversion.compile_time_diff.mcu_diff_operations.diff_operations.conv2d_grad import sparse_depth_wise_mcunetconv2d_grad, sparse_in_channel_mcunetconv2d_grad
from mcu_conversion.compile_time_diff.mcu_diff_operations.diff_operations.truncate_grad import mcutruncate_grad
from mcu_conversion.compile_time_diff.mcu_diff_operations.diff_operations.misc_fn_grads import cross_entropy_with_logits_grad, log_softmax_grad

class AutoDiff(ExprVisitor):
    '''
    The AutoDiff class - performs autodiff at run time and gives us a backward graph
    that can be used for code generation later on. 

    Inherits from Relay's `ExprVisitor`
    
    Moves autodiff from runtime (as is the case in PyTorch or TF) to compile time
    '''

    def __init__(self, debug=True, sparse_op_idx={}):
        '''
        Initialize an auto diff class
        '''
        super().__init__()
        self.names = set()
        self.grads = dict()
        self.var_grads = dict()
        self.last_call = None
        self.debug_mode = debug
        self.op_idx=1
        self.sparse_op_idx = sparse_op_idx
        self.sparse_update_meta_info = []
        self.int8_gradients = False

    def compute_gradients(self, expression):
        '''
        Compute gradients by visiting the provided expression
        '''

        self.vars = relay.analysis.all_vars(expression)
        self.names = set()
        self.grads = dict()
        self.var_grads = dict()
        self.last_call = None
        self.visit(expression)

    def visit_tuple(self, tup):
        '''
        An implementation for relay's `ExprVisitor's` `visit_tuple` function
        '''
        try:
            tuple_gradients = self.grads[hex(tup.handle.value)][0]
        except KeyError:
            if len(self.grads.keys()) != 0:
                raise
            gs = []
            for idx, field in enumerate(tup.fields):
                # dtype = check_call_dtype(field)
                # shape = check_call_shape(field)
                gs.append(relay.ones_like(field))
            tuple_gradients = relay.Tuple(gs)
            self.grads[hex(tup.handle.value)] = tuple_gradients

        for idx, field in enumerate(tup.fields):
            grad = relay.TupleGetItem(tuple_gradients, idx)
            self.grads[hex(field.handle.value)] = (grad, "TupleVisit")
            if isinstance(field, relay.expr.Var):
                self.var_grads[field.name_hint] = (grad, "TupleVisit")
        return Tuple([self.visit(field) for field in tup.fields], tup.span)

    def visit_tuple_getitem(self, op):
        """
        An implementation for relay's `ExprVisitor's` `visit_tuple_getitem` function
        """
        item_gradients = self.grads[hex(op.handle.value)][0]
        arg = op.tuple_value
        addr = hex(arg.handle.value)
        if addr not in self.grads:
            self.grads[addr] = ({op.index: item_gradients}, "TupleGetItem")
        else:
            self.grads[addr][0][op.index] = item_gradients
        if isinstance(arg, relay.expr.Var):
            self.var_grads[arg.name_hint] = (item_gradients, "PlaceHolder")

        # recursively parse the AST
        tuple_value = self.visit(op.tuple_value)
        if not tuple_value.same_as(op.tuple_value):
            return TupleGetItem(tuple_value, op.index)
        return op

    def visit_call(self, call: relay.expr.Call):
        '''
        Get backward function for each call and store this
        '''

        call_op = str(call.op)
        assert (
            call_op != "nn.batch_norm"
        ), "batch norm is not supported yet, please fuse BN"

        address = hex(call.handle.value)

        if address not in self.grads:
            grad_output = relay.ones_like(call)
        else:
            grad_output, _ = self.grads[address]

        if call_op not in GRADIENT_OPERATION_MAP:
            raise NotImplementedError(f"[AutoDiff Error] - |{call.op}| not registered in GRADIENT_OPERATION_MAP")
        else:
            grad_fn = GRADIENT_OPERATION_MAP[call_op]
            is_sparse_update = False

            # Conv2d needs special handling as it can be depth-wise or point-wise and supports partial updates
            if call_op != "nn.mcuconv2d":
                gs = grad_fn(call, grad_output)
            else:
                if self.op_idx in self.sparse_op_idx:
                    is_sparse_update = True
                    gs = self.get_sparse_update_gs(call, grad_output, grad_fn)
                else:
                    gs = grad_fn(call, grad_output)

                #print("OP ", self.op_idx, call.args[0].checked_type.shape, "=>", call.checked_type.shape)
                self.op_idx += 1

            # Assign gradients to each input argument
            assert len(call.args) == len(gs), f"{call.op} |args: {len(call.args)}, gradients: {len(gs)}|"

            for arg, grad in zip(call.args, gs):
                self.grads[hex(arg.handle.value)] = (grad, str(call_op))
                if isinstance(arg, relay.expr.Var):
                    self.var_grads[arg.name_hint] = (grad, is_sparse_update)

        # Recursively past the AST
        for a in list(call.args)[::-1]:
            self.visit(a)
        return call
    
    def obtain_grads(self, filter_fn= lambda x: True):
        '''
        Obtain gradients using name hint and provided filter
        '''
        names = []
        needed_gradients = []
        print("Obtaining gradients")
        for v in self.vars:
            if v.name_hint in self.var_grads and filter_fn(v, self.var_grads[v.name_hint]):
                names.append(v.name_hint)
                needed_gradients.append(self.var_grads[v.name_hint][0])
        return names[::-1], needed_gradients[::-1]
    
    def get_sparse_update_gs(self, call: relay.expr.Call, grad_output, grad_fn):
        '''
        Special handling for sparse BP of nn.mcuconv2d layers
        '''

        attrs = call.attrs
        kernel_size = call.args[1].checked_type.shape[-1]
        data, weight, *_ = call.args
        data_shape = get_const_tuple(data.checked_type.shape)
        weight_shape = get_const_tuple(weight.checked_type.shape)

        # Check which version of in channel and depth wise BP functions we need to use
        if self.int8_gradients:
            in_chanel_sparse_bp = sparse_in_channel_mcunetconv2d_int8grad
            depth_wise_sparse_bp = sparse_depth_wise_mcunetconv2d_int8grad
        else:
            in_chanel_sparse_bp = sparse_in_channel_mcunetconv2d_grad
            depth_wise_sparse_bp = sparse_depth_wise_mcunetconv2d_grad 
            
        base_string =  f"[int8: {self.int8_gradients}] Special handling for sparse bp nn.mcuconv2d: "+f"{str(self.op_idx)} - "+f"shape: {str(call.args[1].checked_type.shape)} "+f"ratio: {str(self.sparse_op_idx[self.op_idx])}"
            
        # Only a subsection of weights to be trained
        if self.sparse_op_idx[self.op_idx] < 1:
            if kernel_size == 1:
                gs = in_chanel_sparse_bp(call, grad_output, topk=self.sparse_op_idx[self.op_idx])
                print(f"[point wise] " + base_string)
            elif attrs.groups == data_shape[1] and data_shape[1] == weight_shape[0]:
                gs = depth_wise_sparse_bp(call, grad_output, topk=self.sparse_op_idx[self.op_idx])
                print(f"[depth wise] " + base_string)
            else:
                raise NotImplementedError(f"kernel size={kernel_size}, {attrs.groups}, {data_shape[1]}, {weight_shape[0]}")
        else:
            from mcu_conversion.ir_utils.ir_var_to_list import ir_var_to_list
            print(f"[full update {kernel_size} * {kernel_size}] " + base_string)
            gs = grad_fn(call, grad_output)
            self.sparse_update_meta_info.append(
                        {
                            "op_idx(revser order)": self.op_idx,
                            "sparse ratio": self.sparse_op_idx[self.op_idx],
                            "gradient shape": ir_var_to_list(
                                call.args[1].checked_type.shape
                            ),
                        }
                    )
        return gs