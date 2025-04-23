'''
The IR generation entry point
'''

# TODO - this should be the main entry point to the generation of IRs

## TODO this needs to be clean and we need to smoke test the different steps to ensure things work correctly

# Will be called from the parent runner, that will allow generation of IR, visualization etc.

import os
import json

from ir_json_converters.model_builder import build_full_quantized_model
from ir_json_converters.model_utils import save_model
from mcu_conversion.ir_json_converters.ir.pytorch_to_IR import pytorch_to_ir
from mcu_conversion.ir_json_converters.ir.backward_graph_gen import BackwardGraphGenerator
from mcu_conversion.ir_json_converters.ir.extract_meta_consts import ExtractMetaConstants

model_path = "irs/mcunet_quantized"

model, _ = build_full_quantized_model(10)

sparse_update_config = {
    ## sample on device
    "49kb": {
        "n_bias_update": 20,"w_ratio": [0, 0.25, 0.5, 0.5, 0, 0], "layer_idx": [23, 24, 27, 30, 33, 39],
    },
}

fwd_mod, real_params, scale_params, op_idx = pytorch_to_ir(model, input_res=[1, 3, 128, 128], num_classes=10)


os.makedirs(model_path, exist_ok=True)
with open(f"{model_path}/scale.json", "w") as fp:
    json.dump(scale_params, fp, indent=2)

save_model(fwd_mod, params=real_params, meta=None, path=model_path, model_name="forward.ir", param_name="weights.params")

for mem, cfg in sparse_update_config.items():
    print(mem)
    bwd_graph_gen = BackwardGraphGenerator(fwd_mod, op_idx, method="sparse_bp", sparse_bp_config=cfg, int8_bp=False)
    bwd_mod, bwd_names, sparse_meta_info = bwd_graph_gen.generate_backward_graph()
    meta_info = {
        "output_info" : bwd_names,
        "sparse_update_info": sparse_meta_info,
    }

    _, consts = ExtractMetaConstants().extract_constants(bwd_mod['main'])

    save_model(
        bwd_mod,
        None,
        path=f"{model_path}",
        model_name=f"sparse_bp_{mem}.ir",
        meta=consts,
        param_name=f"{mem}.params"
    )
    with open(os.path.join(model_path, f"sparse_bp_{mem}.meta"), "w") as fp:
        json.dump(
            meta_info,
            fp,
            indent=2,
        )
    print(bwd_names)