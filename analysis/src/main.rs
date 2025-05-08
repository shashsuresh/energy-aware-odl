use analysis::{
    config::{Config, Models},
    model_representation::{self, channel_ratio::ChannelRatio},
    scheme_generation::{self, bias_update_candidate::BiasUpdateCandidate},
    scheme_representation,
};
use model_representation::model::Model;
use scheme_generation::{params_constraints, sparse_update::SparseUpdateSchemeGenerator};
use scheme_representation::sparse_update_config::SparseUpdateConfig;
use std::io::Error;

fn main() -> Result<(), Error> {
    // Create a Model instance from the json generated when running a simulation of the on device training
    let model = Model::from_json("analysis/model_jsons/mcunet-5fps_all.json")?;

    let candidates = model.into_candidates();

    let mut best_sparse_update_config = SparseUpdateConfig::new(vec![(0, ChannelRatio::All)], 0);

    //In a loop,
    for i in 1..26 {
        let analysis_config = Config {
            last_k_biases: i,
            max_memory: 50,
            model: Models::MCUnet,
        };

        // Create a bias candidate
        let bias_candidate =
            BiasUpdateCandidate::new(analysis_config.last_k_biases, &model, &analysis_config);

        // Create a GreedyGenerator instance
        let mut scheme_gen = SparseUpdateSchemeGenerator::new(
            params_constraints::Constraints::Memory(
                analysis_config.max_memory
                    - bias_candidate
                        .get_memory_cost(&model, analysis_config.model.get_last_layer_idx()),
            ),
            params_constraints::OptimizationParam::Accuracy,
            bias_candidate.get_last_k(),
        );
        // Generate the best update scheme for the given constraints
        let scheme = scheme_gen.generate_scheme_dp(
            candidates.clone(),
            analysis_config.model.get_last_layer_idx(),
        );

        let sparse_update_config = SparseUpdateConfig::from_scheme(scheme, &bias_candidate);
        if sparse_update_config.delta_acc_x100 > best_sparse_update_config.delta_acc_x100 {
            best_sparse_update_config = sparse_update_config
        }
    }
    println!("best: {:?}", best_sparse_update_config);
    Ok(())
}
