use std::{collections::HashMap, marker::PhantomData};

use super::searchable::Searchable;

/// A tabular DP implementation
/// Not implemented as a trait as there is only
/// one way of doing this search
///
/// `T`: Solution type
/// `U`: Caller type
pub struct TabularDP<T, U> {
    solution_table: Vec<Vec<f64>>, // Results of different sub problems will be stored here
    mapped_variants: Option<HashMap<usize, Vec<T>>>, // Hashmap for variants, we support DP with and without variants
    choices: Option<Vec<T>>,
    _caller: PhantomData<U>,
}

impl<T: Clone, U: Searchable<T>> TabularDP<T, U> {
    /// Create a new TabularDP solver instance where items have variants
    pub fn new_with_variants(
        budget: usize,
        all_choices: Vec<T>,
        variant_count: usize,
        caller: &U,
    ) -> Self {
        let rows = (all_choices.len() / variant_count) + 1;
        let columns = budget + 1;

        // Create a table, initializing all values to 0
        let solution_table = vec![vec![0.; columns]; rows];

        // Make a hashmap with the item as a key and the variants as values
        let mut mapped_options: HashMap<usize, Vec<T>> = HashMap::new();
        all_choices.into_iter().for_each(|instance| {
            mapped_options
                .entry(caller.get_id(&instance))
                .and_modify(|variants| variants.push(instance.clone()))
                .or_insert(vec![instance]);
        });

        Self {
            solution_table,
            mapped_variants: Some(mapped_options),
            choices: None,
            _caller: PhantomData,
        }
    }
    /// Create a new TabularDP solver instance where items do not have variants
    pub fn new_without_variants(budget: usize, all_choices: Vec<T>) -> Self {
        let rows = all_choices.len() + 1;
        let columns = budget + 1;

        // Create a table, initializing all values to 0
        let solution_table = vec![vec![0.; columns]; rows];

        Self {
            solution_table,
            mapped_variants: None,
            choices: Some(all_choices),
            _caller: PhantomData,
        }
    }

    /// Find the optimal solution by finding optimal solutions to sub problems.
    /// This is a wrapper function, that acts as an interface
    /// to variants/non-variant function calls
    pub fn search(&mut self, caller: &U) -> Vec<T> {
        match self.mapped_variants {
            Some(_) => {
                todo!()
            }
            None => {
                self.no_variants_search(caller);
                self.get_no_variant_solutions_from_table(caller)
            }
        }
    }

    /// Private method that returns the items that were selected
    /// to reach the optimal solution
    fn get_no_variant_solutions_from_table(&self, caller: &U) -> Vec<T> {
        let rows = self.solution_table.len();
        let mut cost_tracker = self.solution_table[0].len() - 1;
        let mut prev_max = self.solution_table[rows - 1][cost_tracker];
        let mut solution = Vec::new();
        let choices = self.choices.clone().unwrap();
        for item in (1..rows - 1).rev() {
            if self.solution_table[item][cost_tracker] != prev_max {
                solution.push(choices[item].clone());
                cost_tracker -= caller.get_cost(&choices[item]);
                prev_max = self.solution_table[item][cost_tracker]
            }
        }
        solution
    }

    /// Search through a set of unique choices (without variants)
    /// and find the optimal solution
    fn no_variants_search(&mut self, caller: &U) {
        let choices = self.choices.clone().unwrap();
        let rows = self.solution_table.len();
        for item in 1..rows {
            let columns = self.solution_table[item].len();
            for total_cost in 1..columns {
                let prev_item = item - 1;
                let item_zero_indexed = item - 1;
                let score_without_item = self.solution_table[prev_item][total_cost];
                let mut score_with_item = 0.;
                let item_cost = caller.get_cost(&choices[item_zero_indexed]);
                if item_cost <= total_cost {
                    score_with_item += caller.get_opt_param(&choices[item_zero_indexed]);
                    let available_budget = total_cost - item_cost;
                    score_with_item += self.solution_table[prev_item][available_budget];
                }
                self.solution_table[item][total_cost] = score_with_item.max(score_without_item);
            }
        }
    }
}
