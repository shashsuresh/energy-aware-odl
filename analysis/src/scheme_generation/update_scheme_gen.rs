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
    fn generate_schemes(&mut self, all_options: Vec<T>) -> Vec<T>;
    fn eliminate_unreasonable(&self, all_options: Vec<T>) -> Vec<T>;
}
