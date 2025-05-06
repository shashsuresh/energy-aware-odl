use analysis::scheme_generation::{searchable::Searchable, tabular_dp::TabularDP};

// Knapsack to fill
pub struct Knapsack;

#[derive(Clone)]
pub struct Object {
    cost: usize,
    value: usize,
}

impl Searchable<Object> for Knapsack {
    fn get_cost(&self, instance: &Object) -> usize {
        instance.cost
    }

    fn get_opt_param(&self, instance: &Object) -> f64 {
        instance.value as f64
    }

    fn is_duplicate(&self, _instance_1: &Object, _instance_2: &Object) -> bool {
        false
    }

    fn is_allowed(&self, _instance: &Object) -> bool {
        true
    }

    fn get_id(&self, _instance: &Object) -> usize {
        0
    }
}

#[test]
fn test_fill_knapsack_base() {
    let knapsack_to_fill = Knapsack {};

    let test_data = vec![
        Object {
            cost: 24,
            value: 12,
        },
        Object { cost: 10, value: 9 },
        Object { cost: 10, value: 9 },
        Object { cost: 7, value: 5 },
    ];
    let mut searcher: TabularDP<Object, Knapsack> = TabularDP::new_without_variants(25, test_data);
    let results = searcher.search(&knapsack_to_fill);
    let mut knapsack_value = 0;
    for item in results {
        knapsack_value += item.value
    }
    assert_eq!(knapsack_value, 18)
}
