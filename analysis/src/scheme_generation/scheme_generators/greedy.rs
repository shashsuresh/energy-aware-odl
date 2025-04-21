use crate::scheme_generation::{
    update_scheme_candidate::UpdateSchemeCandidate,
    update_scheme_gen::{Constraints, OptimizationParam, SchemeGenerator},
};

/// Structure to represent a greedy update scheme generator
/// which will use the greedy algorithm to derive an update
/// scheme, which maximizes the provided `opt_param` while
/// ensuring the `constraint` provided is met.
pub struct GreedyGenerator {
    constraints: Constraints,     // Constraint that the scheme must meet
    opt_param: OptimizationParam, // Parameter the greedy algorithm should try to maximize
}
impl GreedyGenerator {
    /// Create a new greedy instance
    /// `constraint` must be of the type `Constraints`
    /// `opt_param` must be of the type `OptimizationParam`
    pub fn new(constraints: Constraints, opt_param: OptimizationParam) -> Self {
        GreedyGenerator {
            constraints,
            opt_param,
        }
    }
}

impl SchemeGenerator<UpdateSchemeCandidate> for GreedyGenerator {
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

    fn generate_schemes(
        &mut self,
        all_options: Vec<UpdateSchemeCandidate>,
    ) -> Vec<UpdateSchemeCandidate> {
        let mut good_solutions = self.eliminate_unreasonable(all_options);
        good_solutions.sort_by(|x, y| {
            (self.get_opt_param(y))
                .partial_cmp(&self.get_opt_param(x))
                .unwrap()
        });
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

    fn get_opt_param(&self, instance: &UpdateSchemeCandidate) -> f64 {
        match self.opt_param {
            OptimizationParam::Accuracy => instance.stats.delta_acc as f64, // We can guarantee this as all negative delta acc. candidates have been removed!
            OptimizationParam::Efficiency => {
                instance.stats.delta_acc as f64 / instance.stats.bp_ops as f64
            }
        }
    }

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
}
