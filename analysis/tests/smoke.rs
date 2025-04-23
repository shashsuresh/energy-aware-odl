use std::io::Error;

use analysis::{
    model_representation::{channel_ratio::ChannelRatio, model::Model},
    scheme_generation::{
        params_constraints, sparse_update::SparseUpdateSchemeGenerator,
        update_scheme_candidate::UpdateSchemeCandidate,
    },
};

#[test]
fn test_smoke() -> Result<(), Error> {
    // Create a Model instance from the json generated when running a simulation of the on device training
    let model = Model::from_json("../analysis/tests/test_input.json")?;

    let mut candidates = Vec::new();

    // Create candidates from the layers
    for parsed_layer in model.layers.clone() {
        if let Some(layer_idx) = parsed_layer.id.strip_prefix("conv") {
            let layer_idx_parsed: usize = layer_idx.parse::<usize>().unwrap() - 1;
            let tmp_layer =
                UpdateSchemeCandidate::new(&parsed_layer, layer_idx_parsed, ChannelRatio::All);
            candidates.push(tmp_layer);
            let tmp_layer =
                UpdateSchemeCandidate::new(&parsed_layer, layer_idx_parsed, ChannelRatio::Half);
            candidates.push(tmp_layer);
            let tmp_layer =
                UpdateSchemeCandidate::new(&parsed_layer, layer_idx_parsed, ChannelRatio::Quarter);
            candidates.push(tmp_layer);
        }
    }
    // Create a GreedyGenerator instance
    let mut scheme_gen = SparseUpdateSchemeGenerator::new(
        params_constraints::Constraints::Memory(16), // 34kB reserved for BIAS updates
        params_constraints::OptimizationParam::Accuracy,
    );
    // Generate the best update scheme for the given constraints
    let scheme = scheme_gen.generate_schemes_greedy(candidates);

    // We know that with the data in test_input, 32 gives the highest improvement and fits in memory
    assert_eq!(scheme.len(), 1);
    assert_eq!(scheme[0].id, 32);
    assert_eq!(scheme[0].ratio, ChannelRatio::Half);

    Ok(())
}
