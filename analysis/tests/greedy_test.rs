use analysis::{scheme_generation::searchable::Searchable, search_algorithms::greedy::Greedy};

// We don't really need anything here, so just a blank struct
pub struct Knapsack {
    allow_duplicates: bool,
}

impl Searchable<Object> for Knapsack {
    fn get_cost(&self, instance: &Object) -> usize {
        instance.cost
    }

    fn get_opt_param(&self, instance: &Object) -> f64 {
        instance.value as f64
    }

    fn is_duplicate(&self, instance_1: &Object, instance_2: &Object) -> bool {
        if self.allow_duplicates {
            false // No duplicates test
        } else {
            instance_1.id == instance_2.id
        }
    }

    fn is_allowed(&self, _instance: &Object) -> bool {
        true // No other conditions
    }

    // We do no really care about the ID in his case
    fn get_id(&self, _instance: &Object) -> usize {
        0
    }
}

pub struct Object {
    id: String,
    cost: usize,
    value: usize,
}

#[test]
fn test_basic_knapsack() {
    let knapsack_to_fill = Knapsack {
        allow_duplicates: true,
    };
    let searcher: Greedy<Object, Knapsack> = Greedy::new(25);
    let test_data = vec![
        Object {
            cost: 24,
            value: 12,
            id: "A".to_string(),
        },
        Object {
            cost: 1,
            value: 12,
            id: "A".to_string(),
        },
        Object {
            cost: 10,
            value: 9,
            id: "B".to_string(),
        },
        Object {
            cost: 10,
            value: 9,
            id: "B".to_string(),
        },
        Object {
            cost: 7,
            value: 5,
            id: "C".to_string(),
        },
    ];
    let solution = searcher.search(test_data, &knapsack_to_fill);
    let mut knapsack_value = 0;
    for item in solution {
        knapsack_value += item.value
    }
    assert_eq!(knapsack_value, 24)
}

#[test]
fn test_basic_knapsack_no_duplicates() {
    let test_data = vec![
        Object {
            cost: 24,
            value: 12,
            id: "A".to_string(),
        },
        Object {
            cost: 1,
            value: 12,
            id: "A".to_string(),
        },
        Object {
            cost: 10,
            value: 9,
            id: "B".to_string(),
        },
        Object {
            cost: 10,
            value: 9,
            id: "B".to_string(),
        },
        Object {
            cost: 7,
            value: 5,
            id: "C".to_string(),
        },
    ];
    let knapsack_to_fill = Knapsack {
        allow_duplicates: false,
    };
    let searcher: Greedy<Object, Knapsack> = Greedy::new(25);
    let solution = searcher.search(test_data, &knapsack_to_fill);
    let mut knapsack_value = 0;
    for item in solution {
        knapsack_value += item.value
    }
    assert_eq!(knapsack_value, 12)
}
