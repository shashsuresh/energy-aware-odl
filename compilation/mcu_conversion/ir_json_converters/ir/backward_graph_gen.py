'''
The backward graph generation module
'''

from mcu_conversion.compile_time_diff.compute_autodiff import compute_autodiff

class BackwardGraphGenerator():
    '''
    Backward Graph Generator
    Interface to backward graph generation using the custom offline (compile time)
    `autodiff` framework
    '''
    def __init__(self, mod, op_idx, method, sparse_bp_config=None, int8_bp=True):
        '''
        Initialize the backward graph generator
        '''
        self.mod = mod
        self.op_idx = op_idx
        self.sparse_bp_config = sparse_bp_config
        self.int8_bp = int8_bp
        
        assert method in ["bias_only", "last_only", "sparse_bp", "full_bp"], "Chosen update method is not supported"

        self.method = method

    def bp_method_selector(self):

        def full_bp(v):
            '''
            Full BP graph filter function
            All weights and biases will be selected for update
            '''
            vname = v.name_hint
            if not ("_weight" in vname or "_bias" in vname):
                return False
            if "input" in vname:
                return False
            if "label" in vname:
                return False
            if "_weight" in vname:
                return True
            if "_bias" in vname:
                return True

            return False

        def last_only(v):
            '''
            Last only BP graph filter function
            Only the last layer weights and biases will be selected for update
            '''
            vname = v.name_hint
            if "input" in vname:
                return False
            if "label" in vname:
                return False
            if not ("_weight" in vname or "_bias" in vname):
                return False
            idx = int(vname.split("_")[0].replace("v", ""))
            if idx in [
                self.op_idx,
            ]:
                return True
            return False

        def bias_only(v):
            '''
            Bias only BP graph filter function
            Only the biases will be selected for update
            '''
            vname = v.name_hint
            if "input" in vname:
                return False
            if "label" in vname:
                return False
            if not ("_weight" in vname or "_bias" in vname):
                return False
            idx = int(vname.split("_")[0].replace("v", ""))
            if idx in [
                self.op_idx,
            ]:
                return True
            if "_bias" in vname:
                return True

            return False

        if self.method == "bias_only":
            return bias_only
        elif self.method == "last_only":
            return last_only
        elif self.method == "full_bp":
            return full_bp
        else:
            raise Exception("Only bias_only, last_only, full_bp are supported by the selector. Sparse bp has its own selector!")
        

    def sparse_bp_init(self):
        '''
        Initialize a few parameters for correct generation of backward graph

        **IMPORTANT: MUST BE CALLED PRIOR TO `compute_autodiff`**
        '''
        assert self.sparse_bp_config # Need to know what to update and what not to!

        from mcu_conversion.ir_utils.operation_counter import ir_scan_ops
        total_convs = ir_scan_ops(self.mod["main"])["nn.mcuconv2d"]

        sparse_op_idx = {}

        for idx, r in zip(self.sparse_bp_config["layer_idx"],self.sparse_bp_config["w_ratio"]):
                
            # We only update when w_ratio is non zero
            if r <= 0:
                continue
                
            sparse_op_idx[total_convs - idx] = r
    
        return sparse_op_idx, total_convs

    def get_sparse_bp_fn(self, total_convs):
        '''
        Returns the filter function for sparse bp update parameters selection
        '''
        tot_bias = self.sparse_bp_config["n_bias_update"]
        bias_count = 0
        tot_modules = total_convs

        def sparse_bp(v,g):
            vname = v.name_hint
            grad, is_sparse = g
            if "input" in vname:
                return False
            if "label" in vname:
                return False
            if not ("_weight" in vname or "_bias" in vname):
                return False
            idx = int(vname.split("_")[0].replace("v", ""))
            nonlocal tot_bias, bias_count
            if "_bias" in vname:
                bias_count += 1
                if (tot_modules - bias_count) <= tot_bias:
                    return True
                else:
                    return False
            if is_sparse and "_weight" in vname:
                return True
            if f"{self.op_idx}_" in vname:
                return True
            return False
    
        return sparse_bp
    
    def total_update_gen(self, bwd_names):
        '''
        Generates an overview of weights and biases to updated
        and prints this out
        '''
        from collections import Counter

        update_counter = Counter()
        for name in bwd_names:
            if "_bias" in name:
                update_counter["bias"] += 1
            if "_weight" in name:
                update_counter["weight"] += 1
        print("total update ", update_counter.total())
        print("bias ", update_counter['bias'])
        print("weights ", update_counter['weight'])

    def generate_backward_graph(self):
        '''
        Generates the backward graph for a given config
        '''
        if self.method == "sparse_bp":
            sparse_op_idx, tot_convs = self.sparse_bp_init()
            bwd_mod, bwd_names, sparse_meta_info = compute_autodiff(self.mod, filter_fn=self.get_sparse_bp_fn(tot_convs), sparse_op_idx=sparse_op_idx, return_sparse_meta_info=True, int8_bp=self.int8_bp)

            self.total_update_gen(bwd_names)

            return bwd_mod, bwd_names, sparse_meta_info

        else:
            return compute_autodiff(self.mod, filter_fn=self.bp_method_selector(), int8_bp=self.int8_bp)
        