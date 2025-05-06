/// All structs that a greedy or other search can
/// be performed on require the implementation of this
/// search trait - this is primarily done to ensure that
/// all the necessary conditions can be checked on all
/// types
///
/// `T`: Type of all search objects
pub trait Searchable<T> {
    /// Returns the cost of an instance of `T`
    fn get_cost(&self, instance: &T) -> usize;
    /// Returns the value that is to be maximized
    fn get_opt_param(&self, instance: &T) -> f64;
    /// Returns true if the two instances of `T` can be deemed duplicates
    fn is_duplicate(&self, instance_1: &T, instance_2: &T) -> bool;
    /// Returns true if the instance can be inserted, used for custom conditions
    fn is_allowed(&self, instance: &T) -> bool;
    /// Returns the id of an instance (a unique identifier like item number or type)
    fn get_id(&self, instance: &T) -> usize;
}
