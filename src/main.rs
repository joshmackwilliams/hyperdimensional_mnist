// For initializing random feature vectors
use rand::{rngs::SmallRng, SeedableRng};

// Needed to write partial lines to the console
use std::io::{self, Write};
use std::time::Instant;

use hd_vsa_mnist::integer::IntegerHDModel;
use hd_vsa_mnist::mnist::load_mnist;

fn main() {
    let train_filename = "mnist_train.csv";
    let test_filename = "mnist_test.csv";
    let n_chunks = 64; // Dimensionality of the model / 32
    let mut rng = SmallRng::seed_from_u64(0);

    // Load the dataset - raw images and labels, no vectorization
    print!("Loading data... ");
    let _ = io::stdout().flush();
    let now = Instant::now();
    let (train_images, train_y) = load_mnist(train_filename);
    println!(
        "Loaded {} examples from {} [{}ms]",
        train_images.len(),
        train_filename,
        now.elapsed().as_millis()
    );

    assert!(!train_images.is_empty(), "No training images found!");
    let image_area = train_images[0].len();

    // Set up the model, including feature vectors and quantum vectors
    print!("Initializing model... ");
    let _ = io::stdout().flush();
    let now = Instant::now();
    let mut model = IntegerHDModel::new(n_chunks, image_area, 256, 10, &mut rng);
    println!("Done [{}ms]", now.elapsed().as_millis());

    // Encode the training images using the model
    print!("Encoding images using feature vectors... ");
    let _ = io::stdout().flush();
    let now = Instant::now();
    let train_x = model.encode(&train_images);
    println!("Done [{}ms]", now.elapsed().as_millis());

    // Train the model
    print!("Training model... ");
    let _ = io::stdout().flush();
    let now = Instant::now();
    model.train(&train_x, &train_y);
    println!("Done [{}ms]", now.elapsed().as_millis());

    // Load the test data
    print!("Loading test data... ");
    let _ = io::stdout().flush();
    let now = Instant::now();
    let (test_images, test_y) = load_mnist(test_filename);
    println!(
        "Loaded {} examples from {} [{}ms]",
        test_images.len(),
        test_filename,
        now.elapsed().as_millis()
    );

    // Encode the test data
    print!("Encoding test data... ");
    let _ = io::stdout().flush();
    let now = Instant::now();
    let test_x = model.encode(&test_images);
    println!("Done [{}ms]", now.elapsed().as_millis());

    // Classify the test data and compute accuracy
    print!("Testing model... ");
    let _ = io::stdout().flush();
    let now = Instant::now();
    let mut correct = 0;
    for (x, y) in test_x.chunks(test_x.len() / test_images.len()).zip(test_y) {
        let class = model.classify(x);
        if class == y {
            correct += 1;
        }
    }
    let acc = correct as f64 / test_images.len() as f64;
    println!("Done - Accuracy = {} [{}ms]", acc, now.elapsed().as_millis());
}
