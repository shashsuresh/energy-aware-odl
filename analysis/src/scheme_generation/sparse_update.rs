use crate::scheme_generation::{
    params_constraints::{Constraints, OptimizationParam},
    update_scheme_candidate::UpdateSchemeCandidate,
};

/// Structure to represent a greedy update scheme generator
/// which will use the greedy algorithm to derive an update
/// scheme, which maximizes the provided `opt_param` while
/// ensuring the `constraint` provided is met.
pub struct SparseUpdateSchemeGenerator {
    constraints: Constraints,     // Constraint that the scheme must meet
    opt_param: OptimizationParam, // Parameter the greedy algorithm should try to maximize
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

        for candidate in good_solutions {
            if !scheme.iter().any(|to_update| to_update.id == candidate.id) {
                let constraint_val = self.get_constraint(&candidate);
                if constraint_val.1 < constraint_val.0 {
                    self.constraints = Constraints::Memory(constraint_val.0 - constraint_val.1);
                    scheme.push(candidate.clone());
                }
            }
        }
        scheme
    }

    #[allow(unused)]
    pub fn generate_scheme_dp(
        &mut self,
        all_options: Vec<UpdateSchemeCandidate>,
    ) -> Vec<UpdateSchemeCandidate> {
        let good_solutions = self.eliminate_unreasonable(all_options);
        let _good_solutions = self.sort_solutions(good_solutions);

        todo!();
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

    /// Private method that returns the constraint and the cost of the instance that is being evaluated for selection
    fn get_constraint(&self, instance: &UpdateSchemeCandidate) -> (usize, usize) {
        match self.constraints {
            Constraints::Memory(available) => (
                available,
                (instance.stats.bp_memory as f64 / 1024. / 8.0).round() as usize,
            ),
            Constraints::MACs(_) => (0, 0),
            Constraints::Efficiency(_) => (0, 0),
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
