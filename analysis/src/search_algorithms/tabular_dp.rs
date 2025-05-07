use std::{collections::HashMap, fmt::Debug, marker::PhantomData};

use crate::scheme_generation::searchable::Searchable;

/// A tabular DP implementation
/// Not implemented as a trait as there is only
/// one way of doing this search
///
/// `T`: Solution type
/// `U`: Caller type
pub struct TabularDP<T, U> {
    solution_table: Vec<Vec<f64>>, // Results of different sub problems will be stored here
    mapped_variants: Option<HashMap<usize, Vec<T>>>, // Hashmap for variants, we support DP with and without variants
    variants_inserted: Option<HashMap<(usize, usize), T>>,
    choices: Option<Vec<T>>,
    _caller: PhantomData<U>,
}

impl<T: Clone + Debug, U: Searchable<T>> TabularDP<T, U> {
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
            variants_inserted: Some(HashMap::new()),
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
            variants_inserted: None,
            _caller: PhantomData,
        }
    }

    /// Find the optimal solution by finding optimal solutions to sub problems.
    /// This is a wrapper function, that acts as an interface
    /// to variants/non-variant function calls
    pub fn search(&mut self, caller: &U) -> Vec<T> {
        self.solve_problem(caller);
        self.get_optimal_solution(caller)
    }

    /// Private method that returns the items that were selected
    /// to reach the optimal solution
    fn get_optimal_solution(&self, caller: &U) -> Vec<T> {
        let rows = self.solution_table.len() - 1;
        let mut cost_tracker = self.solution_table[0].len() - 1;
        let mut prev_max = self.solution_table[rows][cost_tracker];
        let mut solution = Vec::new();
        for item in (1..rows).rev() {
            if self.solution_table[item][cost_tracker] != prev_max {
                let member = if let Some(var_map) = &self.variants_inserted {
                    if let Some(var) = var_map.get(&((item + 1), cost_tracker)) {
                        solution.push(var.clone());
                        Some(var.clone())
                    } else {
                        // If we have inserted something with a variant,
                        // we should never get here
                        None
                    }
                } else {
                    let choices = self.choices.clone().unwrap();
                    solution.push(choices[item].clone());
                    Some(choices[item].clone())
                };
                //member is none - something went wrong, so we can unwrap here
                cost_tracker -= caller.get_cost(&member.unwrap());
                prev_max = self.solution_table[item][cost_tracker]
            }
        }
        solution
    }

    /// Search through a set of unique choices (without variants)
    /// and find the optimal solution
    fn solve_problem(&mut self, caller: &U) {
        let rows = self.solution_table.len();
        let use_variants = self.mapped_variants.is_some();
        // item by item
        for item in 1..rows {
            let columns = self.solution_table[item].len();
            // cost value by cost value
            for total_cost in 1..columns {
                let prev_item = item - 1;
                let score_without_item = self.solution_table[prev_item][total_cost];
                let mut score_with_item = 0.;
                let mut best_variant = None;

                // We need to get the best variant and update parameters accordingly of the item we are considering
                let sub_problem_helper = if use_variants {
                    best_variant = self.get_best_variant(caller, total_cost, item);
                    if let Some(var) = &best_variant {
                        SubProblemHelper::new(true, caller.get_cost(var), caller.get_opt_param(var))
                    } else {
                        // No variant satisfied, so we will not be including this item in our optimal helper
                        SubProblemHelper::new(false, 0, 0.)
                    }
                }
                // Use the item directly
                else {
                    let choices = self.choices.clone().unwrap();
                    let item_zero_indexed = item - 1;
                    let item_cost = caller.get_cost(&choices[item_zero_indexed]);
                    SubProblemHelper::new(
                        item_cost <= total_cost,
                        item_cost,
                        caller.get_opt_param(&choices[item_zero_indexed]),
                    )
                };

                // If the present layer can fit in the budget
                if sub_problem_helper.include_item {
                    // Get the "value of including this item"
                    score_with_item += sub_problem_helper.item_score;
                    let available_budget = total_cost - sub_problem_helper.item_cost;
                    score_with_item += self.solution_table[prev_item][available_budget];
                }

                // Check if we should include the object or not, if we do include it and it has a variant
                // keep track of it to help retrieve the variant when determining the optimal solution
                if score_with_item > score_without_item {
                    self.solution_table[item][total_cost] = score_with_item;
                    if let Some(ref mut map) = self.variants_inserted {
                        if let Some(selected_variant) = &best_variant {
                            map.insert((item, total_cost), selected_variant.clone());
                        }
                    }
                } else {
                    self.solution_table[item][total_cost] = score_without_item;
                }
            }
        }
    }

    /// Return the best variant for a given sub-problem
    fn get_best_variant(&self, caller: &U, budget: usize, item: usize) -> Option<T> {
        self.mapped_variants
            .clone()
            .unwrap()
            .get(&(item - 1))
            .unwrap()
            .iter()
            .filter(|variant| caller.get_cost(variant) <= budget)
            .max_by(|x, y| {
                caller
                    .get_opt_param(x)
                    .partial_cmp(&caller.get_opt_param(y))
                    .unwrap()
            })
            .cloned()
    }
}

/// To aid sub-problem solving
/// when no variants are involved
pub struct SubProblemHelper {
    pub include_item: bool,
    pub item_cost: usize,
    pub item_score: f64,
}

impl SubProblemHelper {
    pub fn new(include_item: bool, item_cost: usize, item_score: f64) -> Self {
        Self {
            include_item,
            item_cost,
            item_score,
        }
    }
}
