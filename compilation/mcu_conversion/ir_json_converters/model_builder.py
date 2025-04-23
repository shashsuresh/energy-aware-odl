import torch as tch
from quantization.quantized_network_builder import construct_quantized_torch_nn
from quantization.utils import append_qconv2d_head
from quantization.quantized_operations import QuantizedConv2D

def build_model_for_mcu(base_model="mbv2-w0.35.pkl"):
    '''
    Converts the model stored in the .pkl file to a
    Quantized NN model
    '''

    # Load pkl model
    model_pkl = tch.load("./models/"+base_model)

    # construct NN
    model = construct_quantized_torch_nn(model_pkl, n_bit=8)

    # Attach head to the model
    model = append_qconv2d_head(model)

    return model

def build_full_quantized_model(classes=10):
    '''
    Build a full NN model that the training representations
    will be derived from
    '''
    subnet = build_model_for_mcu("mcunet-5fps.pkl")

    # Take the first 5 blocks
    subnet = tch.nn.Sequential(*subnet[:5])

    # update last layer
    last_layer = subnet[-1]
    subnet[-1] = QuantizedConv2D(
        last_layer.in_channels,
        classes,
        kernel_size=last_layer.kernel_size,
        stride=last_layer.stride,
        zero_x=last_layer.zero_x,
        zero_y=last_layer.zero_y,
        effective_scale=last_layer.effective_scale[:classes]
    )
    subnet[-1].x_scale = last_layer.x_scale
    subnet[-1].y_scale = last_layer.y_scale
    subnet[-1].weight.data = last_layer.weight.data[:classes, :, :, :]
    return subnet, 128