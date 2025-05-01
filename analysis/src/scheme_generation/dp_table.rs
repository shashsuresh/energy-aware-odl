use std::collections::HashMap;

use crate::scheme_generation::update_scheme_candidate::UpdateSchemeCandidate;

use super::sparse_update::SparseUpdateSchemeGenerator;

pub struct DPSearch {
    pub solution_table: Vec<Vec<f64>>, //This is just a table to store solutions
    pub mapped_options: HashMap<usize, Vec<UpdateSchemeCandidate>>, // Table to map idx to the variants
}

impl DPSearch {
    /// Create a new DPSearch instance
    pub fn new(budget: usize, available_options: Vec<UpdateSchemeCandidate>) -> Self {
        let solution_table = vec![vec![0.; budget]; (available_options.len() / 3) + 1];
        let mut mapped_options: HashMap<usize, Vec<UpdateSchemeCandidate>> = HashMap::new();
        for candidate in available_options {
            mapped_options
                .entry(candidate.id)
                .and_modify(|variants| variants.push(candidate.clone()))
                .or_insert(vec![candidate]);
        }
        Self {
            solution_table,
            mapped_options,
        }
    }

    pub fn search_optimal(
        &mut self,
        scheme_gen: &SparseUpdateSchemeGenerator,
    ) -> Vec<UpdateSchemeCandidate> {
        let mut layers_to_train = Vec::new();
        for layer_idx in 1..self.solution_table.len() {
            for budget in 1..self.solution_table[layer_idx].len() {
                let score_without_layer = self.solution_table[layer_idx - 1][budget];
                let mut score_with_layer = 0.;
                if layer_idx - 1 >= 21 {
                    // We do a -1 to 0 index the layers - as that is how they are stored in the options map
                    if let Some(variant) = (self.mapped_options.get(&(layer_idx - 1)))
                        .unwrap()
                        .iter()
                        .filter(|variant| scheme_gen.get_cost(&variant) <= budget)
                        .max_by(|x, y| {
                            (scheme_gen.get_opt_param(x))
                                .partial_cmp(&(scheme_gen.get_opt_param(y)))
                                .unwrap()
                        })
                    {
                        if (layer_idx == 35 && budget == 63)
                            || (layer_idx == 34 && budget == 55)
                            || (layer_idx == 33 && budget == 46)
                            || (layer_idx == 31 && budget == 33)
                            || (layer_idx == 29 && budget == 26)
                            || (layer_idx == 28 && budget == 19)
                            || (layer_idx == 25 && budget == 13)
                            || (layer_idx == 24 && budget == 8)
                            || (layer_idx == 22 && budget == 4)
                        {
                            layers_to_train.push(variant.clone());
                        }
                        score_with_layer += scheme_gen.get_opt_param(variant);
                        let available_memory = budget - scheme_gen.get_cost(variant);
                        score_with_layer += self.solution_table[layer_idx - 1][available_memory]
                    } else {
                        score_with_layer += 0.;
                    }
                }
                self.solution_table[layer_idx][budget] = score_with_layer.max(score_without_layer);
            }
        }
        for i in 21..43 {
            println!("Layer - {}: {:?}", i - 1, self.solution_table[i]);
        }
        for layer in &layers_to_train {
            println!("{:?}", layer);
        }
        layers_to_train
    }
}
