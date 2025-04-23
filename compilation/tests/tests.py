from mcu_conversion.ir_json_converters.model_builder import build_full_quantized_model
from mcu_conversion.ir_json_converters.ir.pytorch_to_IR import pytorch_to_ir
from mcu_conversion.ir_json_converters.ir.backward_graph_gen import BackwardGraphGenerator
from mcu_conversion.ir_json_converters.ir.extract_meta_consts import ExtractMetaConstants
from mcu_conversion.ir_json_converters.json.ir_json_translator import IRJSONTranslator
import tvm.relay
import tvm.ir
import unittest

class TestIRGeneration(unittest.TestCase):

    def setUp(self):
        self.model, _ = build_full_quantized_model(10)

        self.translator_bwd = IRJSONTranslator(path="tests/test_refs/sparse_bp-50kb-1x3x128x128.ir", out_folder="model/sample")
        self.translator_fwd = IRJSONTranslator(path="tests/test_refs/fwd-1x3x128x128.ir", out_folder="model/sample")

        self.fwd_mod, self.real_params, self.scale_params, self.op_idx = pytorch_to_ir(self.model, input_res=[1, 3, 128, 128], num_classes=10)

    def test_forward_gen(self):
        self.assertEqual(True, tvm.ir.structural_equal(self.fwd_mod,self.translator_fwd.model))
    
    def test_backward_gen(self):
        sparse_update_config = {
            "50kb": {
                "n_bias_update": 20,"w_ratio": [0, 0.25, 0.5, 0.5, 0, 0], "layer_idx": [23, 24, 27, 30, 33, 39],
            },
        }

        bwd_graph_gen = BackwardGraphGenerator(self.fwd_mod, self.op_idx, method="sparse_bp", sparse_bp_config=sparse_update_config['50kb'], int8_bp=False)
        bwd_mod, bwd_names, sparse_meta_info = bwd_graph_gen.generate_backward_graph()

        self.assertEqual(True, tvm.ir.structural_equal(bwd_mod,self.translator_bwd.model))

if __name__ == "main":
    unittest.main()