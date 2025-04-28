use std::{cmp::max, collections::HashMap};

use crate::{
    model_representation::channel_ratio::ChannelRatio,
    scheme_generation::{
        params_constraints::{Constraints, OptimizationParam},
        update_scheme_candidate::UpdateSchemeCandidate,
    },
};

/// Structure to represent a sparse update scheme generator
/// which maximizes the provided `opt_param` while
/// ensuring the `constraint` provided is met.
pub struct SparseUpdateSchemeGenerator {
    constraints: Constraints,     // Constraint that the scheme must meet
    opt_param: OptimizationParam, // Parameter the chosen algorithm should try to maximize
}
impl SparseUpdateSchemeGenerator {
    /// Create a new greedy instance
    /// `constraint` must be of the type `Constraints`
    /// `opt_param` must be of the type `OptimizationParam`
    pub fn new(constraints: Constraints, opt_param: OptimizationParam) -> Self {
        SparseUpdateSchemeGenerator {
            constraints,
            opt_param,
        }
    }

    /// A method that allows generation of update strategies from a list of all possible layers to choose from
    /// using the greedy algorithm
    pub fn generate_schemes_greedy(
        &mut self,
        all_options: Vec<UpdateSchemeCandidate>,
    ) -> Vec<UpdateSchemeCandidate> {
        let good_solutions = self.eliminate_unreasonable(all_options);
        let good_solutions = self.sort_solutions(good_solutions);
        let mut scheme: Vec<UpdateSchemeCandidate> = Vec::new();
        let mut budget = self.get_budget();
        for candidate in good_solutions {
            if !scheme.iter().any(|to_update| to_update.id == candidate.id)
                && self.get_cost(&candidate) < budget
            {
                budget -= self.get_cost(&candidate);
                scheme.push(candidate.clone());
            }
        }
        let mut delta_acc = 0;
        for val in &scheme {
            delta_acc += val.stats.delta_acc;
        }
        println!("Delta Acc {}", delta_acc);
        scheme
    }

    // WIP function - we have something that makes sense rn
    pub fn generate_scheme_dp(&mut self, available_options: Vec<UpdateSchemeCandidate>) {
        // Create a table we can easily refer to
        let mut table: HashMap<String, Vec<usize>> = HashMap::new();
        let mut table_index_data = Vec::new();
        table.insert(0.to_string(), vec![0_usize; self.get_budget() + 1]);
        for option in &available_options {
            if option.ratio == ChannelRatio::All {
                let key = option.id.to_string() + "_" + &option.ratio.get_value().to_string();
                table.insert(key, vec![0_usize; self.get_budget() + 1]);
                table_index_data.push(option.to_owned());
            }
        }

        #[allow(clippy::needless_range_loop)]
        for layer_option in 1..=41 {
            for memory_used in
                1..=(table.get(&(layer_option.to_string() + "_1")).unwrap().len() - 1)
            {
                let exclude =
                    table.get(&((layer_option - 1).to_string() + "_1")).unwrap()[memory_used];
                let mut include = 0_usize;

                //TODO this is an issue
                let memory_cost_item = self.get_cost(&table_index_data[layer_option]);

                if memory_cost_item <= memory_used {
                    //TODO this is another issue
                    include = self.get_opt_param(&table_index_data[layer_option]) as usize;

                    let available_memory = memory_used - memory_cost_item;
                    include += table.get(&((layer_option - 1).to_string() + "_1")).unwrap()
                        [available_memory]
                }
                table
                    .entry((layer_option).to_string() + "_1")
                    .and_modify(|cols| cols[memory_used] = max(exclude, include));
            }
        }
        println!("{:?}", table.get("41_1").unwrap()[16])
    }

    /// A private method that sorts all the available layers in descending
    /// order of chosen optimization parameter
    fn sort_solutions(
        &self,
        mut good_solutions: Vec<UpdateSchemeCandidate>,
    ) -> Vec<UpdateSchemeCandidate> {
        good_solutions.sort_by(|x, y| {
            (self.get_opt_param(y))
                .partial_cmp(&self.get_opt_param(x))
                .unwrap()
        });
        good_solutions
    }

    /// Private method that returns the optimization parameter for the update strategy search, based on how the
    /// generator is configured
    fn get_opt_param(&self, instance: &UpdateSchemeCandidate) -> f64 {
        match self.opt_param {
            OptimizationParam::Accuracy => instance.stats.delta_acc as f64, // We can guarantee this as all negative delta acc. candidates have been removed!
            OptimizationParam::Efficiency => {
                instance.stats.delta_acc as f64 / instance.stats.bp_ops as f64
            }
        }
    }

    /// Private method that returns the budget available for the scheme searcher to use
    fn get_budget(&self) -> usize {
        match self.constraints {
            Constraints::Memory(available) => available,
            Constraints::MACs(_) => 0,
            Constraints::Efficiency(_) => 0,
        }
    }

    /// Private method that returns the cost of choosing a layer for update based on the constraint type
    fn get_cost(&self, instance: &UpdateSchemeCandidate) -> usize {
        match self.constraints {
            Constraints::Memory(_) => {
                (instance.stats.bp_memory as f64 / 1024. / 8.).round() as usize
            }
            Constraints::MACs(_) => 0,
            Constraints::Efficiency(_) => 0,
        }
    }

    /// Private method to eliminate all the solutions that have negative or zero delta acc from the loaded "Contribution Analysis" data
    fn eliminate_unreasonable(
        &self,
        all_options: Vec<UpdateSchemeCandidate>,
    ) -> Vec<UpdateSchemeCandidate> {
        let good_population: Vec<UpdateSchemeCandidate> = all_options
            .iter()
            .filter_map(|candidate| {
                if candidate.stats.delta_acc <= 0 {
                    None
                } else {
                    Some(candidate.to_owned())
                }
            })
            .to_owned()
            .collect();
        good_population
    }
}
