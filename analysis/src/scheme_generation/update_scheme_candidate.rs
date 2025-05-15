use serde::{Deserialize, Serialize};

use crate::{
    model_representation::channel_ratio::ChannelRatio, model_representation::layer::Layer,
};

/// Represents a potential member of the sparse update config
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct UpdateSchemeCandidate {
    pub id: usize,
    pub ratio: Option<ChannelRatio>,
    pub stats: RatioStats,
}

impl UpdateSchemeCandidate {
    /// Creates a new `UpdateSchemeCandidate` instance from a `Layer` instance,
    /// the `id` of the layer and a `ChannelRatio`
    pub fn new(layer: &Layer, id: usize, channel_ratio: Option<ChannelRatio>) -> Self {
        UpdateSchemeCandidate {
            id,
            stats: RatioStats::new(layer, channel_ratio),
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
    /// If `channel_ratio` is `None`, this means that only bias is updated.
    ///
    /// **As a rule - if the weights of a layer is updated, then the bias is updated**
    pub fn new(layer: &Layer, channel_ratio: Option<ChannelRatio>) -> Self {
        RatioStats {
            delta_acc: layer.layer_info.get_delta_acc(channel_ratio),
            bp_ops: layer.get_computation_cost(channel_ratio),
            bp_memory: layer.get_activation_memory(channel_ratio)
                + layer.get_weight_memory(channel_ratio),
        }
    }
}
