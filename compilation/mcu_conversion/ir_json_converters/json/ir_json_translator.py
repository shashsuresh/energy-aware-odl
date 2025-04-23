from mcu_conversion.ir_json_converters.model_utils import load_model
import os
import json
import pickle
from tvm import relay
from mcu_conversion.ir_json_converters.json.serialization import SerializeVisitor

class IRJSONTranslator():
    '''
    A translator class for IRs\n
    Allows transformation from IR to JSON
    '''
    def __init__(self, path, out_folder):
        '''
        Load the model and parameters for later use
        '''
        self.file=os.path.basename(path)
        self.folder=os.path.dirname(path)

        self.out_folder=out_folder
        
        # Meta consts present
        if os.path.exists(path.replace(".ir", ".pkl")):
            self.model, self.params = load_model(self.folder, model_file=self.file, meta_consts=self.file.replace(".ir", ".pkl"))
        # No meta consts
        else:
            self.model, self.params = load_model(self.folder, model_file=self.file)

        # Read metadata in the .meta file
        meta_data_path = os.path.join(path.replace(".ir", ".meta"))
        self.meta_info = None

        if os.path.exists(meta_data_path):
            self.meta_info = json.load(open(meta_data_path, "r"))
        
        self.model = relay.transform.InferType()(self.model)

        if self.params is None:
            self.new_params = None
        else:
            self.new_params = {}
            for k, v in self.params.items():
                n = k 
                if k[0].isdigit():
                    n = "v" + k
                self.new_params[n] = self.params[k]

    def translate(self):
        '''
        Main translation function
        '''
        print(self.file)
        print(self.folder)
        ir_serializer = SerializeVisitor(params=self.new_params, meta=self.meta_info)
        ir_serializer.visit(self.model["main"].body)

        os.makedirs(os.path.dirname(os.path.join(self.out_folder, f"{self.file}.ir")), exist_ok=True)
        
        file_name = self.file.split(".")[0]

        # Store BP graph as json
        with open(os.path.join(self.out_folder, f"{file_name}-graph.json"), "w") as file:
            json.dump(ir_serializer.graph, file, indent=2)

        # Store params in .pkl
        with open(os.path.join(self.out_folder, f"{file_name}-params.pkl"), "wb") as file:
            pickle.dump(ir_serializer.params, file)

        # Store final model as an ir
        with open(os.path.join(self.out_folder, f"{file_name}.ir"), "w") as file:
            file.write(str(self.model["main"]))