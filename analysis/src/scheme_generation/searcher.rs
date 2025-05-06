/// All structs that a greedy or other search can
/// be performed on require the implementation of this
/// search trait - this is primarily done to ensure that
/// all the necessary conditions can be checked on all
/// types
///
/// `T`: Type of all search objects
pub trait Searchable<T> {
    fn get_cost(&self, instance: &T) -> usize;
    fn get_opt_param(&self, instance: &T) -> f64;
    fn is_duplicate(&self, instance_1: &T, instance_2: &T) -> bool;
    fn is_allowed(&self, instance: &T) -> bool;
}
