use crate::scheme_generation::update_scheme_candidate::UpdateSchemeCandidate;

use super::channel_ratio::ChannelRatio;

pub struct SparseUpdateConfig {
    pub weights: Vec<(usize, ChannelRatio)>,
    pub bias: usize,
}

impl SparseUpdateConfig {
    pub fn from_scheme(scheme: Vec<UpdateSchemeCandidate>) -> Self {
        let mut weights = Vec::new();
        for layer in scheme {
            weights.push((layer.id, layer.ratio));
        }
        SparseUpdateConfig { weights, bias: 22 }
    }

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
