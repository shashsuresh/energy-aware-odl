use analysis::{
    scheme_generation::searchable::Searchable, search_algorithms::tabular_dp::TabularDP,
};

// Knapsack to fill
pub struct Knapsack;

// Object instance - what can go in our knapsack
#[derive(Clone, Debug)]
pub struct Object {
    id: usize,
    cost: usize,
    value: f64,
}

//A knapsack must implement he Searchable trait
impl Searchable<Object> for Knapsack {
    fn get_cost(&self, instance: &Object) -> usize {
        instance.cost
    }

    fn get_opt_param(&self, instance: &Object) -> f64 {
        instance.value
    }

    fn is_duplicate(&self, instance_1: &Object, instance_2: &Object) -> bool {
        instance_1.id == instance_2.id
    }

    fn is_allowed(&self, _instance: &Object) -> bool {
        true
    }

    fn get_id(&self, instance: &Object) -> usize {
        instance.id
    }
}

#[test]
// Simple knapsack problem
// Check if the result is optimal
fn test_fill_knapsack_base() {
    let knapsack_to_fill = Knapsack {};

    // A few objects to choose from
    let test_data = vec![
        Object {
            cost: 24,
            value: 12.,
            id: 1,
        },
        Object {
            cost: 10,
            value: 9.,
            id: 2,
        },
        Object {
            cost: 10,
            value: 9.,
            id: 3,
        },
        Object {
            cost: 7,
            value: 5.,
            id: 4,
        },
    ];
    let mut searcher: TabularDP<Object, Knapsack> = TabularDP::new_without_variants(25, test_data);
    let results = searcher.search(&knapsack_to_fill);
    let mut knapsack_value = 0.;
    for item in results {
        knapsack_value += item.value
    }
    assert_eq!(knapsack_value, 18.)
}

#[test]
fn test_fill_knapsack_with_variants() {
    let knapsack_to_fill = Knapsack {};

    // A few objects to choose from
    // IDs have to be continuous and zero indexed
    let test_data = vec![
        Object {
            cost: 24,
            value: 12.,
            id: 0,
        },
        Object {
            cost: 24,
            value: 6.,
            id: 0,
        },
        Object {
            cost: 10,
            value: 5.,
            id: 1,
        },
        Object {
            cost: 10,
            value: 10.,
            id: 1,
        },
        Object {
            cost: 7,
            value: 5.,
            id: 2,
        },
        Object {
            cost: 7,
            value: 11.,
            id: 2,
        },
    ];
    let mut searcher: TabularDP<Object, Knapsack> =
        TabularDP::new_with_variants(25, test_data, 2, &Knapsack);
    let results = searcher.search(&knapsack_to_fill);
    let mut knapsack_value = 0.;
    for item in &results {
        knapsack_value += item.value
    }
    println!("{:?}", results);
    assert_eq!(knapsack_value, 21.)
}
