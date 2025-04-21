use serde::{Deserialize, Serialize};
#[derive(Deserialize, Serialize, Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct ConvolutionParameters {
    pub input_shape: [usize; 4],
    pub output_shape: [usize; 4],
    pub groups: usize,
    pub weight_shape: [usize; 4],
    pub bias_count: usize,
    pub total_channels: usize,
    pub all_acc_x100: isize,
    pub half_channels: usize,
    pub half_acc_x100: isize,
    pub quarter_channels: usize,
    pub quarter_acc_x100: isize,
    pub eighth_channels: usize,
    pub eighth_acc_x100: isize,
}
