use evolutionary::{Fitness, Individual, prelude::Bin};

use crate::scheme_generation::update_scheme_candidate::UpdateSchemeCandidate;

#[derive(Clone, Debug)]
pub struct UpdateSchemeFitness {
    pub candidates: Vec<UpdateSchemeCandidate>,
}

impl Fitness<Bin> for UpdateSchemeFitness {
    fn calculate_fitness(&self, individual: &Bin) -> f64 {
        let memory_bp: usize = individual
            .get_chromosome()
            .iter()
            .enumerate()
            .filter_map(|cand| {
                if *cand.1 {
                    Some(self.candidates[cand.0].stats.bp_memory)
                } else {
                    None
                }
            })
            .sum();
        if memory_bp as f64 / 1024.0 / 8.0 > 50. {
            f64::MIN
        } else {
            memory_bp as f64 / 1024.0 / 8.0
        }
    }
}
