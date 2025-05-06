use std::{
    fs::File,
    io::{Error, Read},
};

use serde_json::{Map, Value, from_str, from_value};

use crate::{
    scheme_generation::update_scheme_candidate::UpdateSchemeCandidate,
    scheme_representation::{
        sparse_update_config::SparseUpdateConfig, sparse_update_stats::SparseUpdateStats,
    },
};

use super::{channel_ratio::ChannelRatio, layer::Layer, layer_descriptor::LayerDescriptor};

/// A CNN Model represented as a collection of layers
pub struct Model {
    pub layers: Vec<Layer>,
}

impl Model {
    /// Creates a model instance by parsing a json file into a
    /// vector of layers
    /// If there is any failure, an IO Error is returned
    /// along with a string to help pinpoint the error
    pub fn from_json(json_path: &str) -> Result<Self, Error> {
        //File reading
        let mut json_file = File::open(json_path)?;
        let mut data = String::new();
        json_file.read_to_string(&mut data)?;

        //JSON string to Map conversion
        let layers: Map<String, Value> = from_str(&data)?;

        let mut parsed_layers: Vec<Layer> = vec![];

        // Traverse through the map and convert json strings into `Layer` objects
        for (id, params) in layers {
            // Fully Connected layer
            if id == "FC" {
                parsed_layers.push(Layer {
                    id,
                    layer_info: LayerDescriptor::FC(from_value(params)?),
                });
            }
            // Convolution layer
            else {
                match params["conv_type"].as_str().ok_or(Error::new(
                    std::io::ErrorKind::NotFound,
                    "Conv type not supported",
                )) {
                    Ok("depthwise") => parsed_layers.push(Layer {
                        id,
                        layer_info: LayerDescriptor::DepthwiseConv(from_value(params)?),
                    }),
                    Ok("normal") => parsed_layers.push(Layer {
                        id,
                        layer_info: LayerDescriptor::PointwiseConv(from_value(params)?),
                    }),
                    Ok("group") => parsed_layers.push(Layer {
                        id,
                        layer_info: LayerDescriptor::GroupConv(from_value(params)?),
                    }),
                    _ => {
                        return Err(Error::new(
                            std::io::ErrorKind::NotFound,
                            "Conv type not supported",
                        ));
                    }
                };
            }
        }
        Ok(Self {
            layers: parsed_layers,
        })
    }

    /// For a given sparse update config, return some key statistics
    /// so the user can quickly see an overview of the scheme
    /// Eventually we can use a gui to display this and maybe help the
    /// user select schemes on their own, the current framework is rather difficult
    /// to use
    pub fn get_sparse_update_statistics(
        &self,
        config: SparseUpdateConfig,
        layer_iter_max: usize,
    ) -> SparseUpdateStats {
        // Op tracker
        let mut ops = Vec::new();
        let mut activation_memory = Vec::new();
        let mut weights_memory = Vec::new();

        for layer in &self.layers {
            if let Some(layer_id) = layer.id.strip_prefix("conv") {
                let layer_id_parsed: usize = layer_id.parse::<usize>().unwrap() - 1;

                //First look for the layer in the config, if yes incude weights and biases
                if let Some(pair) = config.weights.iter().find(|pair| pair.0 == layer_id_parsed) {
                    activation_memory.push(layer.get_activation_memory(Some(pair.1)));
                    weights_memory.push(layer.get_weight_memory(Some(pair.1)));
                    ops.push(layer.get_computation_cost(Some(pair.1)));
                } else if (layer_iter_max - config.bias..layer_iter_max).contains(&layer_id_parsed)
                {
                    activation_memory.push(layer.get_activation_memory(None));
                    weights_memory.push(layer.get_weight_memory(None));
                    ops.push(layer.get_computation_cost(None));
                } else {
                    activation_memory.push(0);
                    weights_memory.push(0);
                    ops.push(0);
                }
            } else {
                activation_memory.push(0);
                weights_memory.push(0);
                ops.push(0);
            }
        }
        SparseUpdateStats::new(activation_memory, weights_memory, ops)
    }

    /// Converts layers into candidates for the search
    ///
    /// Candidate: `Vec<UpdateSchemeCandidate>`, a collection of all layer variants we can choose from
    pub fn into_candidates(&self) -> Vec<UpdateSchemeCandidate> {
        let mut candidates = Vec::new();
        for parsed_layer in self.layers.clone() {
            if let Some(layer_idx) = parsed_layer.id.strip_prefix("conv") {
                let layer_idx_parsed: usize = layer_idx.parse::<usize>().unwrap() - 1;
                let tmp_layer =
                    UpdateSchemeCandidate::new(&parsed_layer, layer_idx_parsed, ChannelRatio::All);
                candidates.push(tmp_layer);
                let tmp_layer =
                    UpdateSchemeCandidate::new(&parsed_layer, layer_idx_parsed, ChannelRatio::Half);
                candidates.push(tmp_layer);
                let tmp_layer = UpdateSchemeCandidate::new(
                    &parsed_layer,
                    layer_idx_parsed,
                    ChannelRatio::Quarter,
                );
                candidates.push(tmp_layer);
            }
        }
        candidates
    }
}
