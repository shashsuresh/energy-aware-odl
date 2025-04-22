use serde::{Deserialize, Serialize};
#[derive(Deserialize, Serialize, Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
/// Represents all the parameters of a convolution layer
/// relevant for the analysis of the layer's computation
/// and memory costs
pub struct ConvolutionParameters {
    pub input_shape: [usize; 4],
    pub output_shape: [usize; 4],
    pub groups: usize,
    pub weight_shape: [usize; 4],
    pub bias_count: usize,
    pub total_channels: usize,
    pub all_acc_x100: isize, // delta accuracy if all channels are updated
    pub half_channels: usize,
    pub half_acc_x100: isize, // delta accuracy if half the total channels are updated
    pub quarter_channels: usize,
    pub quarter_acc_x100: isize, // delta accuracy if a quarter of the total channels are updated
    pub eighth_channels: usize,
    pub eighth_acc_x100: isize, // delta accuracy if an eighth of the total channels are updated
}
