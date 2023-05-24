use super::counting_binary_vector::CountingBinaryVector;
use super::majority::fast_approx_majority;
use super::BinaryChunk;
use rand::Rng;
use rayon::prelude::*;

pub struct HDModel {
    // Binary-valued feature x quanta vectors
    feature_vectors: Vec<BinaryChunk>,
    // Number of chunks in the model's feature vectors (actual dimensionality / binary chunk size)
    model_dimensionality_chunks: usize,
    // One vector representing the prototype of each class
    class_vectors: Vec<CountingBinaryVector>,
}

impl HDModel {
    pub fn new(
        model_dimensionality: usize,
        input_dimensionality: usize,
        n_classes: usize,
        rng: &mut impl Rng,
    ) -> Self {
        // Compute model dimensionality in chunks by taking the ceiling of the model dimensionality / chunk size
        let model_dimensionality_chunks =
            (model_dimensionality + BinaryChunk::BITS as usize - 1) / BinaryChunk::BITS as usize;

        // Allocate space for quanta vectors
        let n_feature_vector_chunks = input_dimensionality * 2 * model_dimensionality_chunks;
        let feature_vectors: Vec<BinaryChunk> = (0..n_feature_vector_chunks)
            .map(|_| random_chunk(rng))
            .collect();

        // Zero-initialize class vectors
        let class_vectors = (0..n_classes)
            .map(|_| CountingBinaryVector::new(model_dimensionality_chunks))
            .collect();

        HDModel {
            feature_vectors,
            model_dimensionality_chunks,
            class_vectors,
        }
    }

    // Encode a batch of images
    pub fn encode(&self, input: &[Vec<usize>]) -> Vec<BinaryChunk> {
        input
            .par_iter()
            .flat_map(|image| {
                (0..self.model_dimensionality_chunks)
                    .into_par_iter()
                    .map(|chunk_index| {
                        fast_approx_majority(image.iter().enumerate().map(|(feature, &value)| {
                            self.feature_vectors[((feature * 2 + value)
                                * self.model_dimensionality_chunks)
                                + chunk_index]
                        }))
                    })
            })
            .collect()
    }

    // Consume self and return a trained model, which can be used to classify examples
    pub fn train(&mut self, examples: &[BinaryChunk], labels: &[usize]) {
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
                    let predicted = self.classify(example);

                    // If we get it wrong, reinforce the involved class vectors
                    if predicted != label {
                        self.class_vectors[label].add(example.iter().copied());
                        self.class_vectors[predicted].add(example.iter().copied().map(|x| !x));
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
    pub fn classify(&self, input: &[BinaryChunk]) -> usize {
        self.class_vectors
            .iter()
            .enumerate()
            .min_by_key(|(_, x)| hamming_distance(input, x.as_binary()))
            .unwrap()
            .0
    }
}

// Utility function - find the hamming distance between two binary vectors
fn hamming_distance(a: &[BinaryChunk], b: &[BinaryChunk]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| count_ones_in_chunk(*x ^ y))
        .sum()
}

#[inline]
fn count_ones_in_chunk(x: BinaryChunk) -> u32 {
    x.as_array_ref().iter().map(|x| x.count_ones()).sum()
}

#[inline]
fn random_chunk(rng: &mut impl Rng) -> BinaryChunk {
    array_init::array_init(|_| rng.gen()).into()
}
