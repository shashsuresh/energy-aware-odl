use std::fmt::Display;

/// Represents the energy and computation costs
/// of a particular sparse update config
pub struct SparseUpdateStats {
    layer_wise_activation: Vec<usize>, // Activation cost of each layer
    layer_wise_weights: Vec<usize>,    // Weights cost of each layer
    layer_wise_ops: Vec<usize>,        // Computation cost of each layer
}

impl SparseUpdateStats {
    /// Create a new sparse update stats instance
    pub fn new(
        layer_wise_activation: Vec<usize>,
        layer_wise_weights: Vec<usize>,
        layer_wise_ops: Vec<usize>,
    ) -> Self {
        SparseUpdateStats {
            layer_wise_activation,
            layer_wise_weights,
            layer_wise_ops,
        }
    }

    pub fn get_total_memory_usage(&self) -> usize {
        (self.layer_wise_activation.iter().sum::<usize>() as f32 / 1024. / 8.).round() as usize
            + (self.layer_wise_weights.iter().sum::<usize>() as f32 / 1024. / 8.).round() as usize
    }
}

impl Display for SparseUpdateStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} Additional ops, Activation memory {}kB, Weights memory {}kB",
            self.layer_wise_ops.iter().sum::<usize>(),
            (self.layer_wise_activation.iter().sum::<usize>() as f32 / 1024. / 8.).round() as usize,
            (self.layer_wise_weights.iter().sum::<usize>() as f32 / 1024. / 8.).round() as usize
        )
    }
}
