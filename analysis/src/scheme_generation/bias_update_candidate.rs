use crate::{
    config::Config, model_representation::model::Model,
    scheme_representation::sparse_update_config::SparseUpdateConfig,
};

#[derive(Clone)]
/// A struct to represent the `last_k_biases` to update part of SparseUpdateConfig
pub struct BiasUpdateCandidate {
    last_k: usize, // Last k biases to update
    delta_acc_x100: isize,
    efficiency: f64,
}

impl BiasUpdateCandidate {
    /// Create a new BiasUpdateCandidate
    pub fn new(last_k: usize, model: &Model, config: &Config) -> Self {
        let mut delta_acc_x100 = 0;
        let mut ops = 0;
        for layer in &model.layers {
            if let Some(layer_idx) = layer.id.strip_prefix("conv") {
                if let Ok(id) = layer_idx.parse::<usize>() {
                    if id - 1 >= config.model.get_last_layer_idx() - config.last_k_biases {
                        delta_acc_x100 += layer.layer_info.get_delta_acc(None);
                        ops += layer.get_computation_cost(None);
                    }
                }
            }
        }
        BiasUpdateCandidate {
            last_k,
            delta_acc_x100,
            efficiency: (delta_acc_x100 as f64 / 100.) / ops as f64,
        }
    }

    /// Get the memory cost of running this bias update scheme only
    pub fn get_memory_cost(&self, model: &Model, last_layer_idx: usize) -> usize {
        model
            .get_sparse_update_statistics(
                SparseUpdateConfig::new(Vec::new(), self.last_k),
                last_layer_idx,
            )
            .get_total_memory_usage()
    }

    /// Method to return the private member `last_k` for later use
    pub fn get_last_k(&self) -> usize {
        self.last_k
    }

    /// Method to return the bias config's delta acc
    pub fn get_delta_acc(&self) -> isize {
        self.delta_acc_x100
    }

    /// Method to return the bias config's efficiency
    pub fn get_efficiency(&self) -> f64 {
        self.efficiency
    }
}
