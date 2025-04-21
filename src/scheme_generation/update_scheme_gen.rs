use crate::scheme_generation::update_scheme_candidate::UpdateSchemeCandidate;

#[allow(unused)]
pub enum OptimizationParam {
    Accuracy,
    Efficiency,
}

#[allow(unused)]
pub enum Constraints {
    Memory(usize),
    MACs(usize),
    Efficiency(usize),
}

pub trait SchemeGenerator<T> {
    fn get_opt_param(&self, instance: &T) -> usize;
    fn get_constraint(&self, instance: &T) -> (usize, usize);
    fn generate_schemes(
        &mut self,
        all_options: Vec<T>,
    ) -> Vec<T>;
    fn eliminate_unreasonable(
        &self,
        all_options: Vec<T>,
    ) -> Vec<T>;
}

pub struct GreedyMaxAccGenerator {
    constraints: Constraints,
    opt_param: OptimizationParam,
}
impl GreedyMaxAccGenerator {
    pub fn new(constraint: Constraints, opt_param: OptimizationParam) -> Self {
        GreedyMaxAccGenerator {
            constraints: constraint,
            opt_param,
        }
    }
}

impl SchemeGenerator<UpdateSchemeCandidate> for GreedyMaxAccGenerator {
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
        good_solutions.sort_by(|x, y| (self.get_opt_param(y)).cmp(&self.get_opt_param(x)));
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

    fn get_opt_param(&self, instance: &UpdateSchemeCandidate) -> usize {
        match self.opt_param {
            OptimizationParam::Accuracy => instance.stats.delta_acc as usize, // We can guarantee this as all negative ones have been killed!
            OptimizationParam::Efficiency => 0,
        }
    }

    fn get_constraint(&self, instance: &UpdateSchemeCandidate) -> (usize, usize) {
        match self.constraints {
            Constraints::Memory(val) => (
                val,
                (instance.stats.bp_memory as f64 / 1024. / 8.0).round() as usize,
            ),
            Constraints::MACs(_) => (0, 0),
            Constraints::Efficiency(_) => (0, 0),
        }
    }
}
