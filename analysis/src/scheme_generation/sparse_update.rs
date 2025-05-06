use crate::scheme_generation::{
    params_constraints::{Constraints, OptimizationParam},
    update_scheme_candidate::UpdateSchemeCandidate,
};

use super::{dp_table::DPSearch, searcher::Searchable};

/// Structure to represent a sparse update scheme generator
/// which maximizes the provided `opt_param` while
/// ensuring the `constraint` provided is met.
pub struct SparseUpdateSchemeGenerator {
    constraints: Constraints,     // Constraint that the scheme must meet
    opt_param: OptimizationParam, // Parameter the chosen algorithm should try to maximize
    last_k: usize, // The chosen bias update, that schemes need to be generated based on
}
impl SparseUpdateSchemeGenerator {
    /// Create a new greedy instance
    /// `constraint` must be of the type `Constraints`
    /// `opt_param` must be of the type `OptimizationParam`
    pub fn new(constraints: Constraints, opt_param: OptimizationParam, last_k: usize) -> Self {
        SparseUpdateSchemeGenerator {
            constraints,
            opt_param,
            last_k,
        }
    }

    /// A method that allows generation of update strategies from a list of all possible layers to choose from
    /// using the greedy algorithm
    pub fn generate_schemes_greedy(
        &mut self,
        all_options: Vec<UpdateSchemeCandidate>,
        last_layer_idx: usize,
    ) -> Vec<UpdateSchemeCandidate> {
        // Remove all zero / negative values
        let good_solutions = self.eliminate_unreasonable(all_options);
        // Sort the solutions in descending order of `optimization_param`
        let good_solutions = self.sort_solutions(good_solutions);
        // Placeholder for the result
        let mut scheme: Vec<UpdateSchemeCandidate> = Vec::new();
        // Total available budget
        let mut budget = self.get_budget() * 1024;
        // Iterate through the good solutions
        for candidate in good_solutions {
            // If the cost is lower than the available budget
            // and this layer is not already in the list of
            // all solutions and we update its bias too
            // then insert and update the available budget
            if !scheme.iter().any(|to_update| to_update.id == candidate.id)
                && candidate.id > last_layer_idx - self.last_k
                && self.get_cost(&candidate) < budget
            {
                budget -= self.get_cost(&candidate);
                scheme.push(candidate.clone());
            }
        }
        // Return scheme
        scheme
    }

    // WIP function - we have something that makes sense rn
    // Think of this
    // For each layer at each stage evaluate each option and pick the best that fits in
    // Move this to a separate struct, so that we can run our tests
    pub fn generate_scheme_dp(
        &mut self,
        available_options: Vec<UpdateSchemeCandidate>,
        last_layer_idx: usize,
    ) -> Vec<UpdateSchemeCandidate> {
        // Create a table we can easily refer to
        let mut dp_searcher = DPSearch::new(self.get_budget() * 1024, available_options);
        dp_searcher.search_optimal(self, last_layer_idx)
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

    /// Returns the budget available for the scheme searcher to use
    pub fn get_budget(&self) -> usize {
        match self.constraints {
            Constraints::Memory(available) => available,
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

    /// Getter function, so that `last_k` remains a private member of the `SparseUpdateSchemeGenerator` struct
    pub fn get_last_k(&self) -> usize {
        self.last_k
    }
}

impl Searchable<UpdateSchemeCandidate> for SparseUpdateSchemeGenerator {
    /// Method that returns the cost of choosing a layer for update based on the constraint type
    fn get_cost(&self, instance: &UpdateSchemeCandidate) -> usize {
        match self.constraints {
            Constraints::Memory(_) => instance.stats.bp_memory / 8,
            Constraints::MACs(_) => 0,
            Constraints::Efficiency(_) => 0,
        }
    }
    /// Method that returns the optimization parameter for the update strategy search, based on how the
    /// generator is configured
    fn get_opt_param(&self, instance: &UpdateSchemeCandidate) -> f64 {
        match self.opt_param {
            OptimizationParam::Accuracy => instance.stats.delta_acc as f64, // We can guarantee this as all negative delta acc. candidates have been removed!
            OptimizationParam::Efficiency => {
                instance.stats.delta_acc as f64 / instance.stats.bp_ops as f64
            }
        }
    }

    /// 2 candidates are duplicates if their ids are same
    /// We can only have one variant of a layer in the scheme
    fn is_duplicate(
        &self,
        instance_1: &UpdateSchemeCandidate,
        instance_2: &UpdateSchemeCandidate,
    ) -> bool {
        instance_1.id == instance_2.id
    }

    fn is_allowed(&self, instance: &UpdateSchemeCandidate) -> bool {
        //TODO - how do we want this 42?
        instance.id > 42 - self.last_k
    }
}
