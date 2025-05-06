use std::marker::PhantomData;

use super::searcher::Searchable;

/// A greedy search struct
/// Not implemented as a trait as there is only
/// one way of doing greedy search
///
/// `T`: Solution type
/// `U`: Caller type
pub struct Greedy<T, U> {
    budget: usize,
    _solutions: PhantomData<T>, // Just to allow the use of generics
    _caller: PhantomData<U>,    // Just to allow the use of generics
}

impl<T, U: Searchable<T>> Greedy<T, U> {
    /// Create a new Greedy search instance
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            _solutions: PhantomData,
            _caller: PhantomData,
        }
    }

    /// Run the greedy search algorithm
    pub fn search(&self, good_candidates_sorted: Vec<T>, scheme_gen: U) -> Vec<T> {
        let mut available_budget = self.budget;
        let mut result = Vec::new();

        for candidate in good_candidates_sorted {
            if !result
                .iter()
                .any(|layer_selected| scheme_gen.is_duplicate(layer_selected, &candidate))
                && scheme_gen.is_allowed(&candidate)
                && scheme_gen.get_cost(&candidate) <= available_budget
            {
                available_budget -= scheme_gen.get_cost(&candidate);
                result.push(candidate);
            }
        }

        result
    }
}
