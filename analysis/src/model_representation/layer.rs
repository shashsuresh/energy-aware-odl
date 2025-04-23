use serde::{Deserialize, Serialize};

use crate::model_representation::layer_descriptor::LayerDescriptor;

use super::channel_ratio::ChannelRatio;

#[derive(Deserialize, Serialize, Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
/// A CNN layer described by an ID and a layer descriptor
pub struct Layer {
    pub id: String,                  // Just an identifier for the layer
    pub layer_info: LayerDescriptor, // Layer parameters - can vary based on type
}

impl Layer {
    /// Returns the computation cost for updating a given layer based
    /// on the provided channel ratio
    /// No channel - only bias ops
    pub fn get_computation_cost(&self, channel_ratio: Option<ChannelRatio>) -> usize {
        let bias_ops = self.layer_info.get_bias_op_count();
        if let Some(ratio) = channel_ratio {
            let weight_ops = self.layer_info.get_weight_op_count(ratio);
            weight_ops + bias_ops
        } else {
            bias_ops
        }
    }
    /// Returns the activation memory cost for updating a given layer based
    /// on the provided channel ratio
    /// No channel - only bias memory
    pub fn get_activation_memory(&self, channel_ratio: Option<ChannelRatio>) -> usize {
        let bias_activation_mem = self.layer_info.get_bias_only_activation_memory_prediction();
        if let Some(ratio) = channel_ratio {
            let weight_activation_mem = self
                .layer_info
                .get_weight_update_activation_memory_prediction(ratio);
            weight_activation_mem + bias_activation_mem
        } else {
            bias_activation_mem
        }
    }

    /// Returns the "weight" memory cost of updating a given layer based
    /// on the provided channel ratio
    /// No channel ratio - only bias memory
    pub fn get_weight_memory(&self, channel_ratio: Option<ChannelRatio>) -> usize {
        let bias_weight_mem = self.layer_info.get_bias_only_memory_prediction();
        if let Some(ratio) = channel_ratio {
            let weight_mem = self.layer_info.get_weight_update_memory_prediction(ratio);
            bias_weight_mem + weight_mem
        } else {
            bias_weight_mem
        }
    }
}

#[cfg(test)]
mod test {
    use crate::model_representation::params::convolution_params::ConvolutionParameters;

    use super::*;

    fn create_test_layer() -> Layer {
        Layer {
            id: "38".to_string(), //Zero indexed
            layer_info: LayerDescriptor::PointwiseConv(ConvolutionParameters {
                input_shape: [1, 384, 4, 4],
                output_shape: [1, 96, 4, 4],
                groups: 1,
                weight_shape: [96, 384, 1, 1],
                bias_count: 96,
                total_channels: 384,
                all_acc_x100: -19,
                half_channels: 192,
                half_acc_x100: -18,
                quarter_channels: 96,
                quarter_acc_x100: 5,
                eighth_channels: 48,
                eighth_acc_x100: 0,
            }),
        }
    }

    #[test]
    fn test_computation_cost_calculations() {
        let layer_to_test = create_test_layer();
        assert_eq!(
            layer_to_test.get_computation_cost(Some(ChannelRatio::All))
                - layer_to_test.get_computation_cost(None),
            589824 * 4
        );
        assert_eq!(
            layer_to_test.get_computation_cost(Some(ChannelRatio::Half))
                - layer_to_test.get_computation_cost(None),
            294912 * 4
        );
        assert_eq!(
            layer_to_test.get_computation_cost(Some(ChannelRatio::Quarter))
                - layer_to_test.get_computation_cost(None),
            147456 * 4
        );
        assert_eq!(
            layer_to_test.get_computation_cost(Some(ChannelRatio::OneEighth))
                - layer_to_test.get_computation_cost(None),
            73728 * 4
        );
    }

    #[test]
    fn test_memory_cost_calculations() {
        let layer_to_test = create_test_layer();
        assert_eq!(
            layer_to_test.get_activation_memory(Some(ChannelRatio::All)),
            50688
        );
        assert_eq!(
            layer_to_test.get_weight_memory(Some(ChannelRatio::All)),
            297984
        );

        assert_eq!(
            layer_to_test.get_activation_memory(Some(ChannelRatio::Half)),
            26112
        );
        assert_eq!(
            layer_to_test.get_weight_memory(Some(ChannelRatio::Half)),
            150528
        );

        assert_eq!(
            layer_to_test.get_activation_memory(Some(ChannelRatio::Quarter)),
            13824
        );
        assert_eq!(
            layer_to_test.get_weight_memory(Some(ChannelRatio::Quarter)),
            76800
        );

        assert_eq!(
            layer_to_test.get_activation_memory(Some(ChannelRatio::OneEighth)),
            7680
        );
        assert_eq!(
            layer_to_test.get_weight_memory(Some(ChannelRatio::OneEighth)),
            39936
        );

        assert_eq!(layer_to_test.get_activation_memory(None), 1536);
        assert_eq!(layer_to_test.get_weight_memory(None), 3072);
    }
}
