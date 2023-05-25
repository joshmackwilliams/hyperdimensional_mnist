use hd_vsa_mnist::prune_data::{find_prunable_positions, prune_data};
// For initializing random feature vectors
use rand::{rngs::SmallRng, SeedableRng};
use rayon::prelude::*;
use clap::Parser;

// Needed to write partial lines to the console
use std::io::{self, Write};
use std::time::Instant;

use hd_vsa_mnist::hd_model::HDModel;
use hd_vsa_mnist::mnist::load_mnist;

/// Run HD learning on the MNIST dataset
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    // Path to the training data, in CSV format
    #[arg(long, default_value = "mnist_train.csv")]
    train_filename: String,
    // Path to the test data, in CSV format
    #[arg(long, default_value = "mnist_test.csv")]
    test_filename: String,
    // Number of bits in the model
    #[arg(short, long, default_value_t = 1024)]
    dimensionality: usize,
    // Number of examples to load
    #[arg(short, long, default_value_t = 60000)]
    n_examples: usize,
}

fn main() {
    let args = Args::parse();

    let mut rng = SmallRng::seed_from_u64(0);

    let start = Instant::now();

    // Load the dataset - raw images and labels, no vectorization
    print!("Loading data... ");
    let _ = io::stdout().flush();
    let now = Instant::now();
    let (train_images, train_y) = load_mnist(&args.train_filename, args.n_examples);
    println!(
        "Loaded {} examples from {} [{}ms]",
        train_images.len(),
        &args.train_filename,
        now.elapsed().as_millis()
    );

    let prunable_positions = find_prunable_positions(&train_images, 2);
    println!("Found {} prunable values", prunable_positions.len());
    let train_images = prune_data(&train_images, &prunable_positions);

    assert!(!train_images.is_empty(), "No training images found!");
    let image_area = train_images[0].len();

    // Set up the model, including feature vectors and quantum vectors
    print!("Initializing model... ");
    let _ = io::stdout().flush();
    let now = Instant::now();
    let mut model = HDModel::new(args.dimensionality, image_area, 10, &mut rng);
    println!("Done [{}ms]", now.elapsed().as_millis());

    // Encode the training images using the model
    print!("Encoding images using feature vectors... ");
    let _ = io::stdout().flush();
    let now = Instant::now();
    let train_x = model.encode(&train_images);
    println!("Done [{}ms]", now.elapsed().as_millis());

    // Train the model
    println!("=== Training model ===");
    let _ = io::stdout().flush();
    let now = Instant::now();
    model.train(&train_x, &train_y);
    println!("=== Done [{}ms] ===", now.elapsed().as_millis());

    // Load the test data
    print!("Loading test data... ");
    let _ = io::stdout().flush();
    let now = Instant::now();
    let (test_images, test_y) = load_mnist(&args.test_filename, usize::MAX);
    println!(
        "Loaded {} examples from {} [{}ms]",
        test_images.len(),
        &args.test_filename,
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
    let correct: usize = test_x
        .par_chunks(test_x.len() / test_images.len())
        .zip(test_y.par_iter())
        .map(|(x, y)| if model.classify(x) == *y { 1 } else { 0 })
        .sum();
    let acc = correct as f64 / test_images.len() as f64;
    println!(
        "Done - Accuracy = {} [{}ms]",
        acc,
        now.elapsed().as_millis()
    );

    println!("Total time: {}ms", start.elapsed().as_millis());
}
