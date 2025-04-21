use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct FullyConnectedParams {
    pub in_features: usize,
    pub out_features: usize,
    pub weight_count: usize,
    pub bias_count: usize,
}
