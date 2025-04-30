use crate::{
    model_representation::model::Model,
    scheme_representation::sparse_update_config::SparseUpdateConfig,
};

#[derive(Clone)]
/// A struct to represent the `last_k_biases` to update part of SparseUpdateConfig
pub struct BiasUpdateCandidate {
    last_k: usize,          // Last k biases to update
    _delta_acc_x100: isize, // Delta accuracy x 100 when updating the last K biases
}

impl BiasUpdateCandidate {
    /// Create a new BiasUpdateCandidate
    pub fn new(last_k: usize, delta_acc_x100: isize) -> Self {
        BiasUpdateCandidate {
            last_k,
            _delta_acc_x100: delta_acc_x100,
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
}
