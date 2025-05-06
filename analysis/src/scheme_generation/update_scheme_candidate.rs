use serde::{Deserialize, Serialize};

use crate::{
    model_representation::channel_ratio::ChannelRatio, model_representation::layer::Layer,
};

/// Represents a potential member of the sparse update config
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct UpdateSchemeCandidate {
    pub id: usize,
    pub ratio: ChannelRatio,
    pub stats: RatioStats,
}

impl UpdateSchemeCandidate {
    /// Creates a new `UpdateSchemeCandidate` instance from a `Layer` instance,
    /// the `id` of the layer and a `ChannelRatio`
    pub fn new(layer: &Layer, id: usize, channel_ratio: ChannelRatio) -> Self {
        UpdateSchemeCandidate {
            id,
            stats: RatioStats::new(layer, Some(channel_ratio)),
            ratio: channel_ratio,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Copy)]
/// Cost and effect of each variant of a layer, derived by "Contribution Analysis"
pub struct RatioStats {
    pub delta_acc: isize,
    pub bp_ops: usize,
    pub bp_memory: usize,
}

impl RatioStats {
    /// Create a new `RadioStats` instance from a `Layer` instance and a `ChannelRatio`
    /// If ratio is `None`, this means that only bias is updated.
    pub fn new(layer: &Layer, channel_ratio: Option<ChannelRatio>) -> Self {
        RatioStats {
            delta_acc: layer.layer_info.get_delta_acc(channel_ratio),
            bp_ops: if channel_ratio.is_none() {
                // Only bias computation cost
                layer.get_computation_cost(channel_ratio)
            } else {
                // Only weight computation cost
                layer.get_computation_cost(channel_ratio) - layer.get_computation_cost(None)
            },
            bp_memory: if channel_ratio.is_none() {
                // Only take bias memory
                layer.get_activation_memory(channel_ratio) + layer.get_weight_memory(channel_ratio)
            } else {
                // Only take weight memory
                (layer.get_activation_memory(channel_ratio)
                    + layer.get_weight_memory(channel_ratio))
                    - (layer.get_activation_memory(None) + layer.get_weight_memory(None))
            },
        }
    }
}
