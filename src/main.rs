use hd_vsa_mnist::prune_data::{find_prunable_positions, prune_data};
use rand::thread_rng;
// For initializing random feature vectors
use rand::{rngs::SmallRng, SeedableRng};

// Needed to write partial lines to the console
use std::io::{self, Write};
use std::ops::Deref;
use std::time::Instant;

use hd_vsa_mnist::hd_model::{HDModel, UntrainedHDModel};
use hd_vsa_mnist::mnist::load_mnist;

fn main() {
    // TODO make this stuff CLI args
    let train_filename = "mnist_train.csv";
    let test_filename = "mnist_test.csv";
    let dimensionality = 512; // Number of bits in the model
    let n_examples = 10000; // Number of examples to load - can be set lower for testing

    let mut rng = SmallRng::seed_from_u64(0);

    let start = Instant::now();

    // Load the dataset - raw images and labels, no vectorization
    print!("Loading data... ");
    let _ = io::stdout().flush();
    let now = Instant::now();
    let (train_images, train_y) = load_mnist(train_filename, n_examples);
    println!(
        "Loaded {} examples from {} [{}ms]",
        train_images.len(),
        train_filename,
        now.elapsed().as_millis()
    );

    let prunable_positions = find_prunable_positions(&train_images, dimensionality);
    println!("Found {} prunable values", prunable_positions.len());
    let train_images = prune_data(&train_images, &prunable_positions);

    assert!(!train_images.is_empty(), "No training images found!");
    let image_area = train_images[0].len();

    // Set up the model, including feature vectors and quantum vectors
    print!("Initializing model... ");
    let _ = io::stdout().flush();
    let now = Instant::now();
    let model = UntrainedHDModel::new(dimensionality, image_area, 2, 10, &mut rng);
    println!("Done [{}ms]", now.elapsed().as_millis());

    // Encode the training images using the model
    //print!("Encoding images using feature vectors... ");
    //let _ = io::stdout().flush();
    //let now = Instant::now();
    //let train_x = model.encode(&train_images);
    //println!("Done [{}ms]", now.elapsed().as_millis());

    // Train the model
    println!("=== Training model ===");
    let _ = io::stdout().flush();
    let now = Instant::now();
    //let mut model = model.train(&train_x, &train_y);
    //model.retrain(&train_x, &train_y);
    let mut model = ga_train(4, 20, 5, model, &train_images, &train_y);
    let train_x = model.encode(&train_images);
    model.retrain(&train_x, &train_y);
    println!("=== Done [{}ms] ===", now.elapsed().as_millis());

    // Load the test data
    print!("Loading test data... ");
    let _ = io::stdout().flush();
    let now = Instant::now();
    let (test_images, test_y) = load_mnist(test_filename, usize::MAX);
    println!(
        "Loaded {} examples from {} [{}ms]",
        test_images.len(),
        test_filename,
        now.elapsed().as_millis()
    );
    let test_images = prune_data(&test_images, &prunable_positions);

    // Encode the test data
    print!("Encoding test data... ");
    let _ = io::stdout().flush();
    let now = Instant::now();
    let test_x = model.encode(&test_images);
    println!("Done [{}ms]", now.elapsed().as_millis());

    // Similarly, test the binary classifier
    print!("Testing binary classifier... ");
    let _ = io::stdout().flush();
    let now = Instant::now();
    let acc = model.evaluate(&test_x, &test_y) as f64 / test_y.len() as f64;
    println!(
        "Done - Accuracy = {} [{}ms]",
        acc,
        now.elapsed().as_millis()
    );

    println!("Total time: {}ms", start.elapsed().as_millis());
}

fn ga_train(
    n_descendants: usize,
    n_survivors: usize,
    flip_levels: usize,
    initial_model: UntrainedHDModel,
    train_x: &[impl Deref<Target = [usize]>],
    train_y: &[usize],
) -> HDModel {
    let mut survivors = Vec::new();
    survivors.extend((0..(n_survivors - 1)).map(|_| {
        let model = initial_model.mutate(flip_levels, &mut thread_rng());
        let train_x = model.encode(train_x);
        let mut trained = model.train(&train_x, train_y);
        trained.retrain(&train_x, train_y);
        //let acc = trained.evaluate(&train_x, train_y);
        let acc = trained.count_total_class_hamming() as usize;
        (trained, acc)
    }));

    {
        let intial_encoded_train_x = initial_model.encode(train_x);
        let initial_model = initial_model.train(&intial_encoded_train_x, train_y);
        let initial_acc = initial_model.evaluate(&intial_encoded_train_x, train_y);
        survivors.push((initial_model, initial_acc));
    }

    survivors.sort_by_key(|(_, acc)| usize::MAX - *acc);

    let mut best_acc = survivors[0].1;
    let mut i = 0;
    println!("[GA Iteration {}] Best accuracy: {}", i, best_acc);
    let mut epochs_since_improvement = 0;
    while epochs_since_improvement < 5 {
        let mut population: Vec<(HDModel, usize)> = survivors
            .clone()
            .iter()
            .flat_map(|(model, _acc)| {
                (0..n_descendants).map(|_| {
                    let model = model.mutate(flip_levels, &mut thread_rng());
                    let train_x = model.encode(train_x);
                    let mut trained = model.train(&train_x, train_y);
                    trained.retrain(&train_x, train_y);
                    //let acc = trained.evaluate(&train_x, train_y);
                    let acc = trained.count_total_class_hamming() as usize;
                    (trained, acc)
                })
            })
            .collect();
        population.extend(survivors.clone().into_iter());
        population.sort_by_key(|(_, acc)| usize::MAX - *acc);
        let new_best_acc = population[0].1;
        i += 1;
        println!("[GA Iteration {}] Best accuracy: {}", i, new_best_acc);

        if new_best_acc > best_acc {
            epochs_since_improvement = 0;
        } else {
            epochs_since_improvement += 1;
        }
        best_acc = new_best_acc;
        survivors = population[..n_survivors].to_vec();
    }

    survivors[0].0.clone()
}
