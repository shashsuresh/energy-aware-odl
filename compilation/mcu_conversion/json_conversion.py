'''
Convert IRs to JSONs so the model can be deployed on MCUs
'''

import sys
import os
from ir_json_converters.json.ir_json_translator import IRJSONTranslator

model_path="irs/mcunet_quantized/sparse_bp_to_deploy.ir"

if len (sys.argv) >= 2:
    model_path = sys.argv[-1]

assert os.path.exists(model_path), f"{model_path} does not exists!"

translator = IRJSONTranslator(path=model_path, out_folder="to_deploy")
translator.translate()