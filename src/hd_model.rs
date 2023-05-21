use rand::Rng;

use crate::majority::fast_approx_majority;

use super::counting_binary_vector::CountingBinaryVector;
use super::{BinaryChunk, ChunkElement, CHUNK_ELEMENTS, CHUNK_SIZE};

// Separating the untrained version allows us to encode training data before actually training
pub struct HDModel {
    // Binary-valued feature x quanta vectors
    feature_quanta_vectors: Vec<CountingBinaryVector>,
    // Number of chunks in the model's feature vectors (actual dimensionality / binary chunk size)
    model_dimensionality_chunks: usize,
    // Number of quanta for each feature
    input_quanta: usize,
    // One vector to represent each class
    class_vectors: Vec<CountingBinaryVector>,
}

impl HDModel {
    pub fn new(
        model_dimensionality: usize,
        input_dimensionality: usize,
        input_quanta: usize,
        n_classes: usize,
        // RNG for feature vector
        rng: &mut impl Rng,
    ) -> Self {
        // Compute model dimensionality in chunks by taking the ceiling of the model dimensionality / chunk size
        let model_dimensionality_chunks = (model_dimensionality + CHUNK_SIZE - 1) / CHUNK_SIZE;

        // Randomly initialize quanta vectors
        let feature_quanta_vectors = (0..(input_dimensionality * input_quanta))
            .map(|_| CountingBinaryVector::random(model_dimensionality_chunks, rng))
            .collect();

        // Randomly initialize class vectors
        let class_vectors = (0..(n_classes))
            .map(|_| CountingBinaryVector::random(model_dimensionality_chunks, rng))
            .collect();

        HDModel {
            feature_quanta_vectors,
            model_dimensionality_chunks,
            input_quanta,
            class_vectors,
        }
    }

    // Encode a batch of images
    fn batch_encode(&self, input: &[Vec<usize>]) -> Vec<BinaryChunk> {
        // Preallocate the output space
        let mut output =
            vec![BinaryChunk::default(); input.len() * self.model_dimensionality_chunks];
        output
            // Compute one image vector at a time
            .chunks_mut(self.model_dimensionality_chunks)
            // Pair each vector with the corresponding image
            .zip(input.into_iter())
            .for_each(|(output, image)| {
                self.encode(image, output);
            });
        output
    }

    // Encode a single image
    pub fn encode(&self, input: &[usize], output: &mut [BinaryChunk]) {
        output
            .iter_mut()
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                *chunk = fast_approx_majority(input.iter().enumerate().map(|(feature, &value)| {
                    self.feature_quanta_vectors[feature * self.input_quanta + value].as_binary()
                        [chunk_index]
                }));
            });
    }

    // Consume self and return a trained model, which can be used to classify examples
    pub fn train(&mut self, examples: &[Vec<usize>], labels: &[usize]) {
        let examples = self.batch_encode(examples);

        // Replaced initial training entirely with retraining step
        let mut epoch: usize = 0;
        let mut best = usize::MAX;
        let mut epochs_since_improvement = 0;
        while epochs_since_improvement < 5 {
            let mut missed = 0_usize;

            // Classify each example
            examples
                .chunks(self.model_dimensionality_chunks)
                .zip(labels.iter())
                .for_each(|(example, &label)| {
                    let predicted = self.classify_encoded(example);

                    // If we get it wrong, reinforce the involved class vectors
                    if predicted != label {
                        self.class_vectors[label].add(example);
                        self.class_vectors[predicted].subtract(example);
                        missed += 1;
                    }
                });

            // Print status
            println!("[Epoch {}] Misclassified: {}", epoch, missed);

            // Evaluate performance - did we improve on our all time best?
            if missed < best {
                epochs_since_improvement = 0;
                best = missed;
            } else {
                epochs_since_improvement += 1;
            }
            epoch += 1;
        }
    }

    // Classify one example
    pub fn classify_encoded(&self, input: &[BinaryChunk]) -> usize {
        self.class_vectors
            .iter()
            .enumerate()
            .min_by_key(|(_, x)| hamming_distance(input, x.as_binary()))
            .unwrap()
            .0
    }

    pub fn classify(&self, input: &[usize]) -> usize {
        let mut encoded = vec![BinaryChunk::default(); self.model_dimensionality_chunks];
        self.encode(input, &mut encoded);
        self.classify_encoded(&encoded)
    }
}

// Utility function - find the hamming distance between two binary vectors
pub fn hamming_distance(a: &[BinaryChunk], b: &[BinaryChunk]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| count_ones_in_chunk(x ^ y))
        .sum()
}

#[inline]
fn count_ones_in_chunk(x: BinaryChunk) -> u32 {
    let r: &[ChunkElement; CHUNK_ELEMENTS] = x.as_ref();
    r.iter().map(|x| x.count_ones()).sum()
}
