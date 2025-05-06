use std::collections::HashMap;

use crate::scheme_generation::update_scheme_candidate::UpdateSchemeCandidate;

use super::{searcher::Searchable, sparse_update::SparseUpdateSchemeGenerator};

pub struct DPSearch {
    pub solution_table: Vec<Vec<f64>>, //This is just a table to store solutions
    pub mapped_options: HashMap<usize, Vec<UpdateSchemeCandidate>>, // Table to map idx to the variants
}

impl DPSearch {
    /// Create a new DPSearch instance (a table for storing solutions to sub-problems and a mapping to store variants for each layer)
    /// One row for each value from 0 to last layer - `(available_options.len() / 3) + 1`
    /// One column for each value from 0 to `budget+1`
    pub fn new(budget: usize, available_options: Vec<UpdateSchemeCandidate>) -> Self {
        let solution_table = vec![vec![0.; budget + 1]; (available_options.len() / 3) + 1];
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
        last_layer_idx: usize,
    ) -> Vec<UpdateSchemeCandidate> {
        let mut layers_to_train = Vec::new();
        let mut variant_map: HashMap<usize, UpdateSchemeCandidate> = HashMap::new();
        // Account for empty top row and for 0-1 index conversion
        // Ensure we only pick layers within the specified last_k layers for bias update
        for layer_idx in
            ((last_layer_idx + 1) - scheme_gen.get_last_k() + 1)..self.solution_table.len()
        {
            //Leave first column empty as we can't have 0 cost layers
            for budget in 1..self.solution_table[layer_idx].len() {
                // Used to compare whether adding this layer (if possible) is any good or not
                let score_without_layer = self.solution_table[layer_idx - 1][budget];
                // Holds the score of the current layer
                let mut score_with_layer = 0.;
                // Keep track of which variant we are inserting
                let mut variant_tmp = None;
                // If the given layer is part of the initially created map then
                // Find the layer's variant that fits in memory and has the maximum score
                if let Some(variant) = (self.mapped_options.get(&(layer_idx - 1)))
                    .unwrap()
                    .iter()
                    .filter(|variant| scheme_gen.get_cost(variant) <= budget)
                    .max_by(|x, y| {
                        (scheme_gen.get_opt_param(x))
                            .partial_cmp(&(scheme_gen.get_opt_param(y)))
                            .unwrap()
                    })
                {
                    // The result for this sub problem is solved here
                    score_with_layer += scheme_gen.get_opt_param(variant);
                    // We get the previously obtained results, rather than recalculating and update
                    let available_memory = budget - scheme_gen.get_cost(variant);
                    score_with_layer += self.solution_table[layer_idx - 1][available_memory];
                    // We also archive the variant inserted, so that we are able to retrieve it for use later on
                    variant_tmp = Some(variant.clone());
                } else {
                    // If the layer is not inserted, then this component is just left to 0
                    score_with_layer += 0.;
                }
                // If adding the layer to the solution improves performance, then we add it and update our archive hashmap
                if score_with_layer > score_without_layer {
                    self.solution_table[layer_idx][budget] = score_with_layer;
                    variant_map.insert(layer_idx * budget, variant_tmp.unwrap());
                }
                // If not, we forward the previous result, as that still remains the most optimal solution to this
                //sub-problem too
                else {
                    self.solution_table[layer_idx][budget] = score_without_layer;
                }
            }
        }
        // This is the solution extraction part of the dp_table
        let mut prev_max =
            self.solution_table[self.solution_table.len() - 1][self.solution_table[0].len() - 1];
        let mut mem_budget_tracker = self.solution_table[0].len() - 1;
        for layer in (1..self.solution_table.len() - 1).rev() {
            if self.solution_table[layer][mem_budget_tracker] != prev_max {
                if let Some(layer_variant) = variant_map.get(&((layer + 1) * mem_budget_tracker)) {
                    layers_to_train.push(layer_variant.clone());
                    mem_budget_tracker -= scheme_gen.get_cost(layer_variant);
                    prev_max = self.solution_table[layer][mem_budget_tracker];
                }
            }
        }
        layers_to_train
    }
}
