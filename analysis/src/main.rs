use analysis::{
    config::{Config, Models},
    model_representation,
    scheme_generation::{self, bias_update_candidate::BiasUpdateCandidate},
    scheme_representation,
};
use model_representation::{channel_ratio::ChannelRatio, model::Model};
use scheme_generation::{
    params_constraints, sparse_update::SparseUpdateSchemeGenerator,
    update_scheme_candidate::UpdateSchemeCandidate, update_scheme_fitness::UpdateSchemeFitness,
};
use scheme_representation::sparse_update_config::SparseUpdateConfig;
use std::io::Error;

use evolutionary::prelude::*;

fn main() -> Result<(), Error> {
    // Create a Model instance from the json generated when running a simulation of the on device training
    let model = Model::from_json("analysis/model_jsons/mcunet-5fps_all.json")?;

    let mut candidates = Vec::new();

    let mut bias_delta_acc = 0;
    let analysis_config = Config {
        last_k_biases: 22,
        max_memory: 50,
        model: Models::MCUnet,
    };

    // Create candidates from the layers
    for parsed_layer in model.layers.clone() {
        if let Some(layer_idx) = parsed_layer.id.strip_prefix("conv") {
            let layer_idx_parsed: usize = layer_idx.parse::<usize>().unwrap() - 1;
            let tmp_layer =
                UpdateSchemeCandidate::new(&parsed_layer, layer_idx_parsed, ChannelRatio::All);
            if layer_idx_parsed
                > analysis_config.model.get_last_layer_idx() - analysis_config.last_k_biases
            {
                bias_delta_acc += parsed_layer.layer_info.get_delta_acc(None)
            }
            candidates.push(tmp_layer);
            let tmp_layer =
                UpdateSchemeCandidate::new(&parsed_layer, layer_idx_parsed, ChannelRatio::Half);
            candidates.push(tmp_layer);
            let tmp_layer =
                UpdateSchemeCandidate::new(&parsed_layer, layer_idx_parsed, ChannelRatio::Quarter);
            candidates.push(tmp_layer);
        }
    }
    // Create a bias candidate
    let bias_candidate = BiasUpdateCandidate::new(analysis_config.last_k_biases, bias_delta_acc);

    println!(
        "Delta acc by updating bias of last {} layers {}",
        bias_candidate.get_last_k(),
        bias_candidate.get_delta_acc() as f64 / 100.
    );

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
    let scheme =
        scheme_gen.generate_schemes_greedy(candidates, analysis_config.model.get_last_layer_idx());

    // Get the training statistics for the scheme
    let scheme_config = SparseUpdateConfig::from_scheme(scheme, &bias_candidate);
    println!("Our update strategy: \n{}", scheme_config);

    println!(
        "Our scheme update costs {}",
        model.get_sparse_update_statistics(
            scheme_config,
            analysis_config.model.get_last_layer_idx()
        )
    );

    // MIT's final scheme - used to compare our output with theirs
    let mit_100kb_vec = vec![
        (21, ChannelRatio::All),
        (24, ChannelRatio::All),
        (27, ChannelRatio::All),
        (30, ChannelRatio::All),
        (36, ChannelRatio::OneEighth),
        (39, ChannelRatio::Quarter),
    ];

    // Analyze and print this too
    let mit_100kb = SparseUpdateConfig::new(mit_100kb_vec, 22);
    println!(
        "MIT best scheme update costs {}",
        model.get_sparse_update_statistics(mit_100kb, analysis_config.model.get_last_layer_idx())
    );

    Ok(())
}

// TODO need to implement the evolutionary search
#[allow(unused)]
fn run_evolutionary_search(candidates: Vec<UpdateSchemeCandidate>) {
    let mut evolution = EvolutionBuilder::new(candidates.len() as u32, 5, GeneCod::Bin, ())
        .with_fitness(UpdateSchemeFitness { candidates })
        .with_selection(TournamentSelection::default())
        .with_crossover(NPointsCrossover::default())
        .with_mutation(BitFlipMutation::default())
        .with_stop_condition(move |_, iterations, _| iterations >= 1000)
        .build()
        .unwrap();

    evolution.run();
    // After the evolution is done, we can get the best individual and its fitness:
    let best = evolution.current_best();
    println!("Best individual: {:?}", best);
}
