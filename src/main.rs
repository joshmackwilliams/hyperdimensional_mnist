// Convenience crate, used in one function to fill a fixed-size array from an iterator
use arrayvec::ArrayVec;

// Ndarray is used throughout the project to avoid using vectors for everything
use ndarray::{
    parallel::prelude::*, s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3,
    ArrayViewMut1, Axis,
};

// For initializing random feature vectors
use rand::{prelude::SliceRandom, rngs::SmallRng, Rng, SeedableRng};

// Needed to write partial lines to the console
use std::io::{self, Write};

// Constants relating to binary vectors in general
type Chunk = usize;

// Constants relating to MNIST specifically
const MNIST_SIDE_LENGTH: usize = 28;
const MNIST_AREA: usize = MNIST_SIDE_LENGTH * MNIST_SIDE_LENGTH;

// Load MNIST images from a csv file.
// Returns a tuple of (images, labels)
// The expected format is:
// - No headers
// - One image per row
// - Each row starts with the class label 0-9
// - The rest of the row consists of 28x28 pixel values
// - The pixel values are represented as integers, 0-255
pub fn load_mnist(filename: &str) -> (Vec<Vec<usize>>, Vec<usize>) {
    // Use the CSV crate
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(filename)
        .expect("Failed to open CSV file");

    // We use parallel arrays to store the data for easy handling later on
    reader
        .records()
        .map(|result| {
            let record = result.expect("Failed to parse CSV record");
            let label = record[0].parse::<usize>().expect("Failed to parse label");
            let image = record
                .iter()
                .skip(1) // Skip the label
                .map(|x| x.parse::<usize>().expect("Failed to parse pixel"))
                .collect();
            (image, label)
        })
        .unzip()
}

// Encode each of a vector of images (as loaded from the csv file).
// Expects a 3D array of feature vectors such that feature_vecs[i][j][k] is
// the kth chunk of the feature vector corresponding to pixel j at intensity i.
fn encode_images(
    images: Vec<Vec<usize>>,
    feature_vecs: ArrayView3<Chunk>,
    n_chunks: usize,
) -> Array2<Chunk> {
    // Allocate our output array
    let mut output = Array2::<Chunk>::zeros((images.len(), MNIST_AREA));
    output
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(images.into_par_iter())
        .for_each(|(output, image)| {
            // Allocate a spot to store an array of feature vectors
            // We don't use an array to avoid copying
            let image_features: ArrayVec<ArrayView1<Chunk>, MNIST_AREA> = (0..MNIST_AREA)
                .map(|feature| feature_vecs.slice(s![image[feature], feature, ..]))
                .collect();

            // Now, we compute the majority of all of the computed vectors
            approx_majority(&image_features, n_chunks, output);
        });
    output
}

// Not used currently - the approximate majority function seems to work just as well
// UPDATE - this is because we got lucky. 60000 and (28 * 28) are both just above powers of 3
fn _majority(values: &[ArrayView1<Chunk>], n_chunks: usize, mut output: ArrayViewMut1<Chunk>) {
    let threshold = values.len() / 2;
    for chunk in 0..n_chunks {
        output[chunk] = 0;
        for bit in 0..Chunk::BITS {
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

fn approx_majority(
    values: &[ArrayView1<Chunk>],
    n_chunks: usize,
    mut output: ArrayViewMut1<Chunk>,
) {
    const MAX_RANKS: usize = 24;
    let mut ranks_a: [Chunk; MAX_RANKS] = [0; MAX_RANKS];
    let mut ranks_b: [Chunk; MAX_RANKS] = [0; MAX_RANKS];
    let mut a_filled = [false; MAX_RANKS];
    let mut b_filled = [false; MAX_RANKS];
    let n_ranks = 1 + (values.len() as f64).log(3.0).floor() as usize;
    for chunk in 0..n_chunks {
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

fn make_quanta_vectors(n: usize, n_chunks: usize, mut rng: &mut impl Rng) -> Array2<Chunk> {
    let mut quanta_vecs: Array2<Chunk> = Array2::zeros((n, n_chunks));

    // Randomly initialize the first vector
    for chunk in 0..n_chunks {
        quanta_vecs[(0, chunk)] = rng.gen();
    }

    // Create a random order of bits to flip
    let mut flip_order: Vec<usize> = (0..(Chunk::BITS as usize * n_chunks)).collect();
    flip_order.shuffle(&mut rng);
    let flip_per_level = (Chunk::BITS as usize * n_chunks) / (n - 1) / 2;

    // Flip the bits
    for level in 1..n {
        // Copy the previous level
        for chunk in 0..n_chunks {
            quanta_vecs[(level, chunk)] = quanta_vecs[(level - 1, chunk)];
        }

        // Flip some bits
        for i in 0..flip_per_level {
            let index = flip_order[i + (level - 1) * flip_per_level];
            let chunk_to_flip = index / Chunk::BITS as usize;
            let bit_to_flip = index % Chunk::BITS as usize;
            quanta_vecs[(level, chunk_to_flip)] ^= 1 << bit_to_flip;
        }
    }

    quanta_vecs
}

// Batch-generates random feature vectors
fn make_random_vectors(n: usize, n_chunks: usize, rng: &mut impl Rng) -> Array2<Chunk> {
    Array2::from_shape_simple_fn((n, n_chunks), || rng.gen())
}

// This function needs to be refactored to be more reusable
fn make_feature_vectors(n_chunks: usize, rng: &mut impl Rng) -> Array3<Chunk> {
    let intensity_vecs: Array2<Chunk> = make_quanta_vectors(256, n_chunks, rng);
    let pos_vecs: Array2<Chunk> = make_random_vectors(MNIST_AREA, n_chunks, rng);

    let mut feature_vecs: Array3<Chunk> = Array3::zeros((256, MNIST_AREA, n_chunks));
    for pos in 0..MNIST_AREA {
        for chunk in 0..n_chunks {
            for intensity in 0..256 {
                feature_vecs[(intensity, pos, chunk)] =
                    pos_vecs[(pos, chunk)] ^ intensity_vecs[(intensity, chunk)];
            }
        }
    }

    feature_vecs
}

fn main() {
    let n_chunks = 128;

    let mut rng = SmallRng::seed_from_u64(0);

    let train_filename = "mnist_train.csv";
    let test_filename = "mnist_test.csv";

    // Set up the feature vectors
    // For now, we use two levels. This will be more sophisticated in the future
    print!("Generating feature vectors... ");
    let _ = io::stdout().flush();
    let feature_vecs = make_feature_vectors(n_chunks, &mut rng);
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
        approx_majority(examples, n_chunks, class_vectors.slice_mut(s![class, ..]));
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
