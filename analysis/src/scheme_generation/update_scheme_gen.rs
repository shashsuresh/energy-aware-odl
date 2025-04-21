#[allow(unused)]
/// Represents the parameter
/// a given scheme generator aims to maximize
pub enum OptimizationParam {
    Accuracy,
    Efficiency,
}

#[allow(unused)]
/// Represents the constraints that can be used for
/// Greedy and DynamicGreedy scheme generators
pub enum Constraints {
    Memory(usize),
    MACs(usize),
    Efficiency(usize),
}

/// All scheme generators must implement this trait
pub trait SchemeGenerator<T> {
    /// A method to return the value that the provided `OptimizationParam` instance represents
    fn get_opt_param(&self, instance: &T) -> usize;
    /// A method to return the value that the provided `Constraints` instance represents
    fn get_constraint(&self, instance: &T) -> (usize, usize);
    /// A method to generate schemes, based on the algorithm the optimization scheme runs
    fn generate_schemes(&mut self, all_options: Vec<T>) -> Vec<T>;
    /// A method that can eliminate solutions that are not suitable for consideration
    fn eliminate_unreasonable(&self, all_options: Vec<T>) -> Vec<T>;
}
