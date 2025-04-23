mod model_representation;
mod scheme_generation;
mod scheme_representation;

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
        params_constraints::Constraints::Memory(66),
        params_constraints::OptimizationParam::Accuracy,
    );
    // Generate the best update scheme for the given constraints
    let scheme = scheme_gen.generate_schemes_greedy(candidates);

    // Get the training statistics for the scheme
    let scheme_config = SparseUpdateConfig::from_scheme(scheme, 20);
    println!("Our update strategy:");
    scheme_config.display_scheme();

    println!("Our scheme update costs:");
    model
        .get_sparse_update_statistics(scheme_config, 42)
        .display_total_stats();

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
    println!("MIT scheme update costs:");
    model
        .get_sparse_update_statistics(mit_100kb, 42)
        .display_total_stats();

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
