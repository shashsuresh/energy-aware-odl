use serde::{Deserialize, Serialize};

use super::{
    channel_ratio::ChannelRatio,
    params::{
        convolution_params::ConvolutionParameters, fully_connected_params::FullyConnectedParams,
    },
};

static WEIGHT_BITS: usize = 8;
static BIAS_BITS: usize = 32;
static FC_BITS: usize = 0;
static ACTIVATION_BITS: usize = 8;

#[derive(Deserialize, Serialize, Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
/// Wrapper enum where each variant represents a layer and its parameters
pub enum LayerDescriptor {
    GroupConv(ConvolutionParameters), // Group convolution layer with typical convolution parameters
    PointwiseConv(ConvolutionParameters), // Pointwise convolution layer with typical convolution parameters
    DepthwiseConv(ConvolutionParameters), // Depthwise convolution layer with typical convolution parameters
    FC(FullyConnectedParams),             // Fully connected layer with typical FC parameters
}

impl LayerDescriptor {
    /// Returns the computation cost of updating just the biases of a given layer
    /// in terms of ops
    pub fn get_bias_op_count(&self) -> usize {
        match self {
            LayerDescriptor::GroupConv(params) => params.output_shape.iter().product(),
            LayerDescriptor::PointwiseConv(params) => params.output_shape.iter().product(),
            LayerDescriptor::DepthwiseConv(params) => params.output_shape.iter().product(),
            LayerDescriptor::FC(params) => params.out_features, //TODO
        }
    }

    /// Returns the memory cost of updating just the biases of the given layer in bits
    pub fn get_bias_only_memory_prediction(&self) -> usize {
        match self {
            LayerDescriptor::FC(params) => params.bias_count * FC_BITS,
            LayerDescriptor::GroupConv(params)
            | LayerDescriptor::PointwiseConv(params)
            | LayerDescriptor::DepthwiseConv(params) => params.bias_count * BIAS_BITS,
        }
    }

    /// Returns the memory cost of updating the weights
    /// (or subset of weights, based on the `ratio_divider` value)
    /// of the given layer in bits
    pub fn get_weight_update_memory_prediction(&self, ratio_divider: ChannelRatio) -> usize {
        match self {
            LayerDescriptor::FC(params) => params.weight_count * FC_BITS,
            LayerDescriptor::GroupConv(params) => {
                let channel: usize = params.weight_shape.iter().product();
                channel * WEIGHT_BITS
            }
            LayerDescriptor::PointwiseConv(params) => {
                (params.weight_shape[0]
                    * ratio_divider.map_ratio_to_channels(params)
                    * params.weight_shape[2]
                    * params.weight_shape[3])
                    * WEIGHT_BITS
            }
            LayerDescriptor::DepthwiseConv(params) => {
                (ratio_divider.map_ratio_to_channels(params)
                    * params.weight_shape[2]
                    * params.weight_shape[3])
                    * WEIGHT_BITS
            }
        }
    }

    /// Returns the activation memory cost of updating just the biases
    /// of a given layer in bits
    pub fn get_bias_only_activation_memory_prediction(&self) -> usize {
        match self {
            LayerDescriptor::FC(_params) => 0,
            LayerDescriptor::GroupConv(convolution_parameters)
            | LayerDescriptor::PointwiseConv(convolution_parameters)
            | LayerDescriptor::DepthwiseConv(convolution_parameters) => {
                convolution_parameters.output_shape[1]
                    * convolution_parameters.output_shape[2]
                    * convolution_parameters.output_shape[3]
            }
        }
    }

    /// Returns the memory cost of updating the weights
    /// (or subset of weights, based on the `ratio_divider` value)
    /// of the given layer in kB
    pub fn get_weight_update_activation_memory_prediction(
        &self,
        ratio_divider: ChannelRatio,
    ) -> usize {
        match self {
            LayerDescriptor::FC(params) => params.in_features * ACTIVATION_BITS,
            LayerDescriptor::GroupConv(params)
            | LayerDescriptor::PointwiseConv(params)
            | LayerDescriptor::DepthwiseConv(params) => {
                (params.input_shape[2] * params.input_shape[3])
                    * ratio_divider.map_ratio_to_channels(params)
                    * ACTIVATION_BITS
            }
        }
    }

    /// Returns the computation cost of updating the weights
    /// (or subset of weights, based on the `ratio_divider` value)
    /// of the given layer in terms of ops
    pub fn get_weight_op_count(&self, ratio_divider: ChannelRatio) -> usize {
        match self {
            LayerDescriptor::GroupConv(params) => {
                let macs_grad_in_wrt_loss = params.output_shape[2]
                    * params.output_shape[3]
                    * params.weight_shape[2]
                    * params.weight_shape[3]
                    * params.weight_shape[1]
                    * params.output_shape[1]
                    / params.groups;
                let macs_grad_w_wrt_loss = params.input_shape[2]
                    * params.input_shape[3]
                    * params.weight_shape[2]
                    * params.weight_shape[3]
                    * params.input_shape[1]
                    * params.output_shape[1]
                    / params.groups;

                ((2 * macs_grad_in_wrt_loss + 2 * macs_grad_w_wrt_loss) as f32
                    / ratio_divider as usize as f32)
                    .round() as usize
            }
            LayerDescriptor::PointwiseConv(params) => {
                let macs_grad_in_wrt_loss: usize = params.output_shape[2]
                    * params.output_shape[3]
                    * params.weight_shape[2]
                    * params.weight_shape[3]
                    * params.weight_shape[1]
                    * params.output_shape[1];
                let macs_grad_w_wrt_loss = params.input_shape[2]
                    * params.input_shape[3]
                    * params.weight_shape[2]
                    * params.weight_shape[3]
                    * params.input_shape[1]
                    * params.output_shape[1];

                ((2 * macs_grad_in_wrt_loss + 2 * macs_grad_w_wrt_loss) as f32
                    / ratio_divider as usize as f32)
                    .round() as usize
            }
            LayerDescriptor::DepthwiseConv(params) => {
                let macs_grad_in_wrt_loss = params.output_shape[2]
                    * params.output_shape[3]
                    * params.weight_shape[2]
                    * params.weight_shape[3]
                    * params.weight_shape[1];
                let macs_grad_w_wrt_loss = params.input_shape[2]
                    * params.input_shape[3]
                    * params.weight_shape[2]
                    * params.weight_shape[3]
                    * params.input_shape[1];

                2 * macs_grad_in_wrt_loss + 2 * macs_grad_w_wrt_loss
            }
            LayerDescriptor::FC(params) => {
                // Only addition operations here, no MACs
                //TODO we need to recalculate this
                params.in_features * params.out_features
            }
        }
    }

    pub fn get_delta_acc(&self, ratio_divider: Option<ChannelRatio>) -> isize {
        match self {
            LayerDescriptor::GroupConv(convolution_parameters)
            | LayerDescriptor::PointwiseConv(convolution_parameters)
            | LayerDescriptor::DepthwiseConv(convolution_parameters) => {
                if let Some(divider) = ratio_divider {
                    match divider {
                        ChannelRatio::All => convolution_parameters.all_acc_x100,
                        ChannelRatio::Half => convolution_parameters.half_acc_x100,
                        ChannelRatio::Quarter => convolution_parameters.quarter_acc_x100,
                        ChannelRatio::OneEighth => convolution_parameters.eighth_acc_x100,
                    }
                } else {
                    0
                }
            }
            _ => 0,
        }
    }
}
