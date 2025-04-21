mod model_representation;
mod scheme_generation;
mod scheme_representation;

use model_representation::{channel_ratio::ChannelRatio, model::Model};
use scheme_generation::{
    scheme_generators::greedy::GreedyGenerator,
    update_scheme_candidate, update_scheme_fitness,
    update_scheme_gen::{self, SchemeGenerator},
};
use scheme_representation::sparse_update_config::SparseUpdateConfig;
use std::io::Error;
use update_scheme_candidate::UpdateSchemeCandidate;
use update_scheme_fitness::UpdateSchemeFitness;

use evolutionary::prelude::*;

fn main() -> Result<(), Error> {
    let model = Model::from_json("analysis/misc/mcunet-5fps_all.json")?;

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

    let mut scheme_gen = GreedyGenerator::new(
        update_scheme_gen::Constraints::Memory(16),
        update_scheme_gen::OptimizationParam::Efficiency,
    );
    let scheme = scheme_gen.generate_schemes(candidates);

    let scheme_config = SparseUpdateConfig::from_scheme(scheme);
    scheme_config.display_scheme();
    model
        .get_sparse_update_statistics(scheme_config, 42)
        .display_total_stats();

    Ok(())
}

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
