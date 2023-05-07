// Ndarray is used throughout the project to avoid using vectors for everything
use ndarray::{s, Array1, Array2, ArrayView1};

// For initializing random feature vectors
use rand::{rngs::SmallRng, SeedableRng};

// Needed to write partial lines to the console
use std::io::{self, Write};

use hd_vsa_mnist::binary::{majority, make_feature_vectors, classify};
use hd_vsa_mnist::mnist::{encode_images, load_mnist};
use hd_vsa_mnist::Chunk;

fn main() {
    let train_filename = "mnist_train.csv";
    let test_filename = "mnist_test.csv";
    let n_chunks = 128;
    let mut rng = SmallRng::seed_from_u64(0);

    // Load the dataset - raw images and labels, no vectorization
    print!("Loading data... ");
    let _ = io::stdout().flush();
    let (train_images, train_labels) = load_mnist(train_filename);
    println!(
        "Loaded {} examples from {}",
        train_images.len(),
        train_filename
    );

    assert!(!train_images.is_empty(), "No training images found");
    let image_area = train_images[0].len();

    // Set up the feature vectors
    // For now, we use two levels. This will be more sophisticated in the future
    print!("Generating feature vectors... ");
    let _ = io::stdout().flush();
    let feature_vecs = make_feature_vectors(image_area, n_chunks, &mut rng);
    println!("Done");

    // Use feature vectors to encode the images
    print!("Encoding images using feature vectors... ");
    let _ = io::stdout().flush();
    let train_x = encode_images(train_images, feature_vecs.view(), n_chunks);
    let train_y = Array1::from(train_labels);
    println!("Done");

    print!("Computing class vectors from examples... ");
    let _ = io::stdout().flush();

    // Sort the examples by label
    let mut class_examples: Vec<Vec<ArrayView1<Chunk>>> = Vec::new();
    class_examples.resize_with(10, Vec::new);
    for i in 0..train_x.shape()[0] {
        class_examples[train_y[i]].push(train_x.slice(s![i, ..]));
    }

    // Compute class vectors
    let mut class_vectors = Array2::<Chunk>::zeros((10, n_chunks));
    for (class, examples) in class_examples.iter().enumerate() {
        majority(examples, n_chunks, class_vectors.slice_mut(s![class, ..]));
    }
    println!("Done");

    // Load the test data
    print!("Loading test data... ");
    let _ = io::stdout().flush();
    let (test_images, test_labels) = load_mnist(test_filename);
    println!(
        "Loaded {} examples from {}",
        test_images.len(),
        test_filename
    );

    // Encode the test data as we did with the training data
    print!("Encoding test data... ");
    let _ = io::stdout().flush();
    let test_x = encode_images(test_images, feature_vecs.view(), n_chunks);
    let test_y = Array1::from(test_labels);
    println!("Done");

    // Classify the test data and compute accuracy
    print!("Testing model... ");
    let _ = io::stdout().flush();
    let mut correct = 0;
    for i in 0..test_x.shape()[0] {
        let x = test_x.slice(s![i, ..]);
        let y = &test_y[i];
        let class = classify(x, class_vectors.view());
        if class == *y {
            correct += 1;
        }
    }

    let acc = correct as f64 / test_x.shape()[0] as f64;
    println!("Done - Accuracy = {}", acc);
}
