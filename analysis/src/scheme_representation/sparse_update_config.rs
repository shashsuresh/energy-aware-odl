use std::fmt::Display;

use crate::{
    model_representation::channel_ratio::ChannelRatio,
    scheme_generation::{
        bias_update_candidate::BiasUpdateCandidate, update_scheme_candidate::UpdateSchemeCandidate,
    },
};

/// Represents a sparse update configuration
#[derive(Debug)]
pub struct SparseUpdateConfig {
    pub weights: Vec<(usize, ChannelRatio)>, // Layers id, weight update ratio
    pub bias: usize,                         // last k biases to be updated
    pub delta_acc_x100: isize,
    pub efficiency: f64,
}

impl SparseUpdateConfig {
    /// Creates a new sparse update config from an update scheme (`Vec<UpdateSchemeCandidate>`)
    /// Primarily to switch from an update scheme to something the analysis framework can use
    pub fn from_scheme(scheme: Vec<UpdateSchemeCandidate>, bias: &BiasUpdateCandidate) -> Self {
        let mut weights = Vec::new();
        let mut delta_acc_x100 = 0;
        let mut efficiency_total = 0.;
        for layer in scheme {
            weights.push((layer.id, layer.ratio));
            delta_acc_x100 += layer.stats.delta_acc;
            efficiency_total += (layer.stats.delta_acc as f64 / 100.) / layer.stats.bp_ops as f64
        }
        delta_acc_x100 += bias.get_delta_acc();
        SparseUpdateConfig {
            weights,
            bias: bias.get_last_k(),
            delta_acc_x100,
            efficiency: efficiency_total + bias.get_efficiency(),
        }
    }

    /// Create a new sparse update config from a `Vec` of layers and ratios and
    /// the number of biases to update
    pub fn new(layer_ratio_pairs: Vec<(usize, ChannelRatio)>, k_bias: usize) -> Self {
        SparseUpdateConfig {
            weights: layer_ratio_pairs,
            bias: k_bias,
            delta_acc_x100: 0,
            efficiency: 0.,
        }
    }
}

impl Display for SparseUpdateConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut tmp = String::new();
        for layer in &self.weights {
            tmp += &format!(
                "Layer: {}, Weight Update Ratio: {}\n",
                layer.0,
                layer.1.get_value()
            );
        }
        write!(f, "Bias of last {} layers\n{}", self.bias, tmp)
    }
}
