use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut1};
use rand::prelude::SliceRandom;
use std::io::{self, Write};

type Chunk = usize;
const CHUNKS_PER_VEC: usize = 128;
const CHUNK_SIZE: usize = Chunk::BITS as usize;
const ELEMENTS_PER_VEC: usize = CHUNKS_PER_VEC * CHUNK_SIZE;

const MNIST_SIDE_LENGTH: usize = 28;
const MNIST_AREA: usize = MNIST_SIDE_LENGTH * MNIST_SIDE_LENGTH;

pub fn load_mnist(filename: &str) -> (Vec<Vec<usize>>, Vec<usize>) {
    // Get a reader
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(filename)
        .expect("Failed to open CSV file");

    let mut labels = Vec::new();
    let mut images = Vec::new();

    // Parse the records into examples
    for result in reader.records() {
        // Parse the record
        let record = result.expect("Failed to parse CSV record");

        // Read the label
        let label = record[0].parse::<usize>().expect("Failed to parse label");
        labels.push(label);

        let mut image = Vec::new();

        // Read the image
        for i in 0..MNIST_AREA {
            let value = record[i + 1]
                .parse::<usize>()
                .expect("Failed to parse pixel");
            image.push(value);
        }
        images.push(image);
    }

    (images, labels)
}

fn encode_images(images: Vec<Vec<usize>>, feature_vecs: ArrayView3<Chunk>) -> Array2<Chunk> {
    // Allocate a spot to store an array of feature vectors
    // We don't use an array to avoid copying
    let mut image_features = [feature_vecs.slice(s![0, 0, ..]); MNIST_AREA];

    // Allocate our output array
    let mut output = Array2::<Chunk>::zeros((images.len(), MNIST_AREA));

    // Iterate over every image
    for (i, image) in images.into_iter().enumerate() {
        // We compute one vector for each pixel
        for feature in 0..MNIST_AREA {
            image_features[feature] = feature_vecs.slice(s![image[feature], feature, ..]);
        }

        // Now, we compute the majority of all of the computed vectors
        approx_majority(&image_features, output.slice_mut(s![i, ..]));
    }

    output
}

// Not used currently - the approximate majority function seems to work just as well
fn _majority(values: &[ArrayView1<Chunk>], mut output: ArrayViewMut1<Chunk>) {
    let threshold = values.len() / 2;
    for chunk in 0..CHUNKS_PER_VEC {
        output[chunk] = 0;
        for bit in 0..CHUNK_SIZE {
            let mut count = 0;
            for value in values {
                count += (value[chunk] >> bit) & 1;
            }
            if count > threshold {
                output[chunk] |= 1 << bit;
            }
        }
    }
}

fn approx_majority(values: &[ArrayView1<Chunk>], mut output: ArrayViewMut1<Chunk>) {
    const MAX_RANKS: usize = 24;
    let mut ranks_a: [Chunk; MAX_RANKS] = [0; MAX_RANKS];
    let mut ranks_b: [Chunk; MAX_RANKS] = [0; MAX_RANKS];
    let mut a_filled = [false; MAX_RANKS];
    let mut b_filled = [false; MAX_RANKS];
    let n_ranks = 1 + (values.len() as f64).log(3.0).floor() as usize;
    for chunk in 0..CHUNKS_PER_VEC {
        ranks_a.fill(0);
        ranks_b.fill(0);
        a_filled.fill(false);
        b_filled.fill(false);
        for value in values {
            let mut chunk_contents = value[chunk];
            let mut rank = 0;
            loop {
                if !a_filled[rank] {
                    ranks_a[rank] = chunk_contents;
                    a_filled[rank] = true;
                    break;
                } else if !b_filled[rank] {
                    ranks_b[rank] = chunk_contents;
                    b_filled[rank] = true;
                    break;
                } else {
                    a_filled[rank] = false;
                    b_filled[rank] = false;
                    chunk_contents = (chunk_contents & ranks_a[rank])
                        | (chunk_contents & ranks_b[rank])
                        | (ranks_a[rank] & ranks_b[rank]);
                    rank += 1;
                }
            }
        }
        output[chunk] = ranks_a[n_ranks - 1];
    }
}

fn hamming(a: ArrayView1<Chunk>, b: ArrayView1<Chunk>) -> usize {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| (a ^ b).count_ones() as usize)
        .sum()
}

fn classify(x: ArrayView1<Chunk>, class_vectors: ArrayView2<Chunk>) -> usize {
    let mut best_hamming = usize::max_value();
    let mut best_class = 0;
    for class in 0..10 {
        let hamming_dist = hamming(x, class_vectors.slice(s![class, ..]));
        if hamming_dist < best_hamming {
            best_class = class;
            best_hamming = hamming_dist;
        }
    }
    best_class
}

fn make_feature_vectors(n_levels: usize) -> Array3<Chunk> {
    // Allocate space
    let mut feature_vecs = Array3::<Chunk>::zeros((n_levels, MNIST_AREA, CHUNKS_PER_VEC));

    // We'll need an rng for this
    let mut rng = rand::thread_rng();

    for feature in 0..MNIST_AREA {
        // Make a random vector for level 0
        for chunk in 0..CHUNKS_PER_VEC {
            let r = rand::random();
            feature_vecs[(0, feature, chunk)] = r;
        }

        // Create a random order of bits to flip
        let mut flip_order = (0..ELEMENTS_PER_VEC).collect::<Vec<usize>>();
        flip_order.shuffle(&mut rng);
        let flip_per_level = ELEMENTS_PER_VEC / (n_levels - 1);

        for level in 1..n_levels {
            // Copy the previous level
            for chunk in 0..CHUNKS_PER_VEC {
                feature_vecs[(level, feature, chunk)] = feature_vecs[(level - 1, feature, chunk)];
            }

            // Flip some bits
            for i in 0..flip_per_level {
                let index = flip_order[i + (level - 1) * flip_per_level];
                let chunk_to_flip = index / CHUNK_SIZE;
                let bit_to_flip = index % CHUNK_SIZE;
                feature_vecs[(level, feature, chunk_to_flip)] ^= 1 << bit_to_flip;
            }
        }
    }
    feature_vecs
}

fn main() {
    let train_filename = "mnist_train.csv";
    let test_filename = "mnist_test.csv";

    // Set up the feature vectors
    // For now, we use two levels. This will be more sophisticated in the future
    print!("Generating feature vectors... ");
    let _ = io::stdout().flush();
    let feature_vecs = make_feature_vectors(256);
    println!("Done");

    // Load the dataset - raw images and labels, no vectorization
    print!("Loading data... ");
    let _ = io::stdout().flush();
    let (train_images, train_labels) = load_mnist(train_filename);
    println!(
        "Loaded {} examples from {}",
        train_images.len(),
        train_filename
    );

    // Use feature vectors to encode the images
    print!("Encoding images using feature vectors... ");
    let _ = io::stdout().flush();
    let train_x = encode_images(train_images, feature_vecs.view());
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
    let mut class_vectors = Array2::<Chunk>::zeros((10, CHUNKS_PER_VEC));
    for (class, examples) in class_examples.iter().enumerate() {
        approx_majority(examples, class_vectors.slice_mut(s![class, ..]));
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
    let test_x = encode_images(test_images, feature_vecs.view());
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
