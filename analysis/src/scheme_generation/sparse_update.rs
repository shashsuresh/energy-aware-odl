use std::cmp::{max, min};

use crate::scheme_generation::{
    params_constraints::{Constraints, OptimizationParam},
    update_scheme_candidate::UpdateSchemeCandidate,
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

    fn dp_fill_scheme(
        &mut self,
        budget: usize,
        options: &Vec<UpdateSchemeCandidate>,
        current_solution: &mut Vec<UpdateSchemeCandidate>,
        element_to_pick: usize,
    ) -> usize {
        // If we are already at the end of the list or there is no memory budget left, just snap out
        if element_to_pick == 0 || budget == 0 {
            return 0;
        }

        // Extract the cost of updating this particular layer
        let cost = self.get_cost(&options[element_to_pick]);
        // New solution, if the currently analyzed layer is added to the solution
        let mut solution_with = 0;

        // If the new layer can be accommodated
        if cost <= budget {
            let contribution = options[element_to_pick].stats.delta_acc;
            current_solution.push(options[element_to_pick].clone());
            let options = options
                .iter()
                .filter(|option| option.id != options[element_to_pick].id)
                .cloned()
                .collect();
            solution_with = contribution as usize
                + self.dp_fill_scheme(
                    budget - cost,
                    &options,
                    current_solution,
                    min(options.len() - 1, element_to_pick - 1),
                )
        }
        // If we can not accommodate the analyzed layer, then we try with the next one
        let solution_without =
            self.dp_fill_scheme(budget, options, current_solution, element_to_pick - 1);

        max(solution_with, solution_without)
    }

    pub fn generate_scheme_dp(
        &mut self,
        all_options: Vec<UpdateSchemeCandidate>,
    ) -> Vec<UpdateSchemeCandidate> {
        let good_solutions = self.eliminate_unreasonable(all_options);
        let mut scheme = Vec::new();
        let tmp = self.dp_fill_scheme(
            self.get_budget(),
            &good_solutions,
            &mut scheme,
            good_solutions.len() - 1,
        );
        println!("Delta Acc {}", tmp);
        scheme
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
