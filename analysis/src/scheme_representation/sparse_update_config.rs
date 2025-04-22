use crate::{
    model_representation::channel_ratio::ChannelRatio,
    scheme_generation::update_scheme_candidate::UpdateSchemeCandidate,
};

/// Represents a sparse update configuration
pub struct SparseUpdateConfig {
    pub weights: Vec<(usize, ChannelRatio)>, // Layers id, weight update ratio
    pub bias: usize,                         // last k biases to be updated
}

impl SparseUpdateConfig {
    /// Creates a new sparse update config from an update scheme (`Vec<UpdateSchemeCandidate>`)
    /// Primarily to switch from an update scheme to something the analysis framework can use
    pub fn from_scheme(scheme: Vec<UpdateSchemeCandidate>, k_biases: usize) -> Self {
        let mut weights = Vec::new();
        for layer in scheme {
            weights.push((layer.id, layer.ratio));
        }
        SparseUpdateConfig { weights, bias: k_biases }
    }

    /// Display the scheme in a reader friendly format
    pub fn display_scheme(&self) {
        print!("Bias of last {} layers\nWeights - ", self.bias);
        for layer in &self.weights {
            print!(
                "Layer: {}, Weight Update Ratio: {} ",
                layer.0,
                layer.1.get_value()
            )
        }
        println!();
    }
}
