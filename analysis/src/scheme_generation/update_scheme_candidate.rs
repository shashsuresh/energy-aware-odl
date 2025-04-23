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
    // pub half: RatioStats,
    // pub quarter: RatioStats,
    // pub eighth: RatioStats,
    // pub bias: RatioStats,
}

impl UpdateSchemeCandidate {
    /// Creates a new `UpdateSchemeCandidate` instance from a `Layer` instance,
    /// the `id` of the layer and a `ChannelRatio`
    pub fn new(layer: &Layer, id: usize, channel_ratio: ChannelRatio) -> Self {
        UpdateSchemeCandidate {
            id,
            stats: RatioStats::new(layer, Some(channel_ratio)),
            ratio: channel_ratio,
            // half: RatioStats::new(layer, Some(ChannelRatio::Half)),
            // quarter: RatioStats::new(layer, Some(ChannelRatio::Quarter)),
            // eighth: RatioStats::new(layer, Some(ChannelRatio::OneEighth)),
            // bias: RatioStats::new(layer, None),
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
            bp_ops: layer.get_computation_cost(channel_ratio),
            bp_memory: layer.get_activation_memory(channel_ratio)
                + layer.get_weight_memory(channel_ratio),
        }
    }
}
