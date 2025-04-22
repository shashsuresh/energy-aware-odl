use serde::{Deserialize, Serialize};

use super::params::convolution_params::ConvolutionParameters;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Serialize, Deserialize)]
/// Channel ratios we support for sparse update configs
/// Note that for any given channel ratio, the weights are selected
/// based on magnitude.
/// Refer to MIT's `tiny-training/algorithm` for more information
/// on weight selection
pub enum ChannelRatio {
    All = 1,       // All channels (1)
    Half = 2,      // Half the channels (0.5)
    Quarter = 4,   // Quarter of the channels (0.25)
    OneEighth = 8, // One Eighth of the channels (0.125)
}

impl ChannelRatio {
    /// A selector to return the correct channel count for a given channel ratio
    /// from the parsed layers.
    /// For specifications on how the channel count is obtained, refer to
    /// MIT's `tiny-training/algorithm`
    pub fn map_ratio_to_channels(&self, layer_desc: &ConvolutionParameters) -> usize {
        match self {
            ChannelRatio::All => layer_desc.total_channels,
            ChannelRatio::Half => layer_desc.half_channels,
            ChannelRatio::Quarter => layer_desc.quarter_channels,
            ChannelRatio::OneEighth => layer_desc.eighth_channels,
        }
    }

    /// Returns the value corresponding to each enum variant
    /// can be used for evaluation or for printing
    pub fn get_value(&self) -> f64 {
        match self {
            ChannelRatio::All => 1.,
            ChannelRatio::Half => 0.5,
            ChannelRatio::Quarter => 0.25,
            ChannelRatio::OneEighth => 0.125,
        }
    }
}
