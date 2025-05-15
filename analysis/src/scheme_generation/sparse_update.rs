use crate::{
    scheme_generation::{
        params_constraints::{Constraints, OptimizationParam},
        update_scheme_candidate::UpdateSchemeCandidate,
    },
    search_algorithms::{greedy::Greedy, tabular_dp::TabularDP},
};

use super::searchable::Searchable;

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
    pub fn generate_scheme_greedy(
        &mut self,
        all_options: Vec<UpdateSchemeCandidate>,
    ) -> Vec<UpdateSchemeCandidate> {
        // Remove all zero / negative values
        let good_solutions = self.eliminate_unreasonable(all_options);
        // Sort the solutions in descending order of `optimization_param`
        let good_solutions = self.sort_solutions(good_solutions);

        // Create a search instance
        let greedy_searcher: Greedy<UpdateSchemeCandidate, SparseUpdateSchemeGenerator> =
            Greedy::new(self.get_budget() * 1024);

        // Return the scheme that maximize the provided constraint
        greedy_searcher.search(good_solutions, self)
    }

    // WIP function - we have something that makes sense rn
    // Think of this
    // For each layer at each stage evaluate each option and pick the best that fits in
    // Move this to a separate struct, so that we can run our tests
    pub fn generate_scheme_dp(
        &mut self,
        available_options: Vec<UpdateSchemeCandidate>,
    ) -> Vec<UpdateSchemeCandidate> {
        // Create a table we can easily refer to
        let mut dp_searcher =
            TabularDP::new_with_variants(self.get_budget() * 1024, available_options, 4, self);
        // Search
        dp_searcher.search(&self)
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

    fn is_allowed(&self, _instance: &UpdateSchemeCandidate) -> bool {
        true
    }

    fn get_id(&self, instance: &UpdateSchemeCandidate) -> usize {
        instance.id
    }
}
