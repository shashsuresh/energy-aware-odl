#[derive(Clone)]
/// A struct to represent the `last_k_biases` to update part of SparseUpdateConfig
pub struct BiasUpdateCandidate {
    last_k: usize, // Last k biases to update
}

impl BiasUpdateCandidate {
    /// Create a new BiasUpdateCandidate
    pub fn new(last_k: usize) -> Self {
        BiasUpdateCandidate { last_k }
    }

    /// Method to return the private member `last_k` for later use
    pub fn get_last_k(&self) -> usize {
        self.last_k
    }
}
