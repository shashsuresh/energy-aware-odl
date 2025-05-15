use std::fmt::Display;

use crate::{
    model_representation::channel_ratio::ChannelRatio,
    scheme_generation::{
        bias_update_candidate::BiasUpdateCandidate, update_scheme_candidate::UpdateSchemeCandidate,
    },
};

/// Represents a sparse update configuration
pub struct SparseUpdateConfig {
    pub weights: Vec<(usize, Option<ChannelRatio>)>, // Layers id, weight update ratio
    pub bias: Option<usize>,                         // last k biases to be updated
}

impl SparseUpdateConfig {
    /// Creates a new sparse update config from an update scheme (`Vec<UpdateSchemeCandidate>`)
    /// with a custom `last_k_bias`, similar to the default version of MIT's framework
    /// Primarily to switch from an update scheme to something the analysis framework can use
    pub fn from_scheme_with_k_bias(
        scheme: Vec<UpdateSchemeCandidate>,
        bias: &BiasUpdateCandidate,
    ) -> Self {
        let mut weights = Vec::new();
        for layer in scheme {
            weights.push((layer.id, layer.ratio));
        }
        SparseUpdateConfig {
            weights,
            bias: Some(bias.get_last_k()),
        }
    }

    /// Creates a new sparse update config from an update scheme (`Vec<UpdateSchemeCandidate>`)
    /// Primarily to switch from an update scheme to something the analysis framework can use
    pub fn from_scheme(scheme: Vec<UpdateSchemeCandidate>) -> Self {
        let mut weights = Vec::new();
        for layer in scheme {
            weights.push((layer.id, layer.ratio));
        }
        SparseUpdateConfig {
            weights,
            bias: None,
        }
    }

    /// Create a new sparse update config from a `Vec` of layers and ratios and
    /// the number of biases to update
    pub fn new_with_k_bias(
        layer_ratio_pairs: Vec<(usize, Option<ChannelRatio>)>,
        k_bias: usize,
    ) -> Self {
        SparseUpdateConfig {
            weights: layer_ratio_pairs,
            bias: Some(k_bias),
        }
    }

    /// Create a new sparse update config from a `Vec` of layers and ratios
    pub fn new(layer_ratio_pairs: Vec<(usize, Option<ChannelRatio>)>) -> Self {
        SparseUpdateConfig {
            weights: layer_ratio_pairs,
            bias: None,
        }
    }
}

impl Display for SparseUpdateConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut tmp = String::new();
        for layer in &self.weights {
            match layer.1 {
                Some(ratio) => {
                    tmp += &format!(
                        "Layer: {}, Weight Update Ratio: {}\n",
                        layer.0,
                        ratio.get_value()
                    );
                }
                None => {
                    tmp += &format!("Layer: {}, Bias only\n", layer.0,);
                }
            }
        }
        if let Some(last_k_bias_to_update) = self.bias {
            write!(f, "Bias of last {} layers\n{}", last_k_bias_to_update, tmp)
        } else {
            write!(f, "{}", tmp)
        }
    }
}
