use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
/// Represents all the parameters of a FC (linear) layer
/// relevant for the analysis of the layer's computation
/// and memory costs
pub struct FullyConnectedParams {
    pub in_features: usize,
    pub out_features: usize,
    pub weight_count: usize,
    pub bias_count: usize,
}
