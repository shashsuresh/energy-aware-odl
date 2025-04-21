use serde::{Deserialize, Serialize};

use crate::{ChannelRatio, layer_descriptor::LayerDescriptor};

#[derive(Deserialize, Serialize, Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
/// A CNN layer described by an ID and a layer descriptor
pub struct Layer {
    pub id: String,                  // Just an identifier for the layer
    pub layer_info: LayerDescriptor, // Layer parameters - can vary based on type
}

impl Layer {
    /// Returns the computation cost for updating a given layer based
    /// on the provided channel ratio
    /// No channel - only bias ops
    pub fn get_computation_cost(&self, channel_ratio: Option<ChannelRatio>) -> usize {
        let bias_ops = self.layer_info.get_bias_op_count();
        if let Some(ratio) = channel_ratio {
            let weight_ops = self.layer_info.get_weight_op_count(ratio);
            weight_ops + bias_ops
        } else {
            bias_ops
        }
    }
    /// Returns the activation memory cost for updating a given layer based
    /// on the provided channel ratio
    /// No channel - only bias memory
    pub fn get_activation_memory(&self, channel_ratio: Option<ChannelRatio>) -> usize {
        let bias_activation_mem = self.layer_info.get_bias_only_activation_memory_prediction();
        if let Some(ratio) = channel_ratio {
            let weight_activation_mem = self
                .layer_info
                .get_weight_update_activation_memory_prediction(ratio);
            weight_activation_mem + bias_activation_mem
        } else {
            bias_activation_mem
        }
    }

    /// Returns the "weight" memory cost of updating a given layer based
    /// on the provided channel ratio
    /// No channel ratio - only bias memory
    pub fn get_weight_memory(&self, channel_ratio: Option<ChannelRatio>) -> usize {
        let bias_weight_mem = self.layer_info.get_bias_only_memory_prediction();
        if let Some(ratio) = channel_ratio {
            let weight_mem = self.layer_info.get_weight_update_memory_prediction(ratio);
            bias_weight_mem + weight_mem
        } else {
            bias_weight_mem
        }
    }
}
