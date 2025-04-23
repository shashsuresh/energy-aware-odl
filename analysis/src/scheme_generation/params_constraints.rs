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
