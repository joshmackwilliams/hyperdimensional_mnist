use rand::Rng;

use crate::majority::fast_approx_majority;

use super::counting_binary_vector::CountingBinaryVector;
use super::{BinaryChunk, ChunkElement, CHUNK_ELEMENTS, CHUNK_SIZE};

// Separating the untrained version allows us to encode training data before actually training
pub struct UntrainedHDModel {
    // Binary-valued feature x quanta vectors
    feature_quanta_vectors: Vec<CountingBinaryVector>,
    // Number of chunks in the model's feature vectors (actual dimensionality / binary chunk size)
    model_dimensionality_chunks: usize,
    // Number of class vectors computed by the model
    n_classes: usize,
    // Number of quanta for each feature
    input_quanta: usize,
}

impl UntrainedHDModel {
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

        // Allocate space for quanta vectors
        let mut feature_quanta_vectors: Vec<CountingBinaryVector> =
            Vec::with_capacity(input_dimensionality * input_quanta * model_dimensionality_chunks);

        // Create random quanta vectors
        feature_quanta_vectors.extend(
            (0..(input_dimensionality * input_quanta))
                .map(|_| CountingBinaryVector::random(model_dimensionality_chunks, rng)),
        );

        UntrainedHDModel {
            feature_quanta_vectors,
            model_dimensionality_chunks,
            n_classes,
            input_quanta,
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
    pub fn train(self, examples: &[Vec<usize>], labels: &[usize]) -> HDModel {
        let examples = self.batch_encode(examples);

        // Allocate space for integer-valued class vectors
        let mut class_vectors =
            vec![CountingBinaryVector::new(self.model_dimensionality_chunks); self.n_classes];

        // Compute initial class vectors
        examples
            .chunks(self.model_dimensionality_chunks)
            .zip(labels.iter())
            .for_each(|(example, &label)| {
                class_vectors[label].add(example);
            });

        // Initialize a model so we can classify examples during retraining
        let mut model = HDModel {
            untrained_model: self,
            class_vectors,
        };

        // Retraining step greatly improves accuracy
        // We keep running until there's no improvement for five epochs
        // TODO think about how to parallelize this
        // I've tried batching, but the binary model seems brittle in that case
        let mut epoch: usize = 0;
        let mut best = usize::MAX;
        let mut epochs_since_improvement = 0;
        while epochs_since_improvement < 5 {
            let mut missed = 0_usize;

            // Classify each example
            examples
                .chunks(model.untrained_model.model_dimensionality_chunks)
                .zip(labels.iter())
                .for_each(|(example, &label)| {
                    let predicted = model.classify_encoded(example);

                    // If we get it wrong, reinforce the involved class vectors
                    if predicted != label {
                        model.class_vectors[label].add(example);
                        model.class_vectors[predicted].subtract(example);
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

        model
    }
}

// Trained version of the model
// Actually just a wrapper around the untrained model, along with the trained class vectors
// Note - The counting vectors are included here so that we can keep training online.
//        This project does no such thing yet, but it could be useful as some point.
//        Besides, the counting vectors are actually pretty small.
pub struct HDModel {
    untrained_model: UntrainedHDModel,
    class_vectors: Vec<CountingBinaryVector>,
}

impl HDModel {

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
        let mut encoded = vec![BinaryChunk::default(); self.untrained_model.model_dimensionality_chunks];
        self.untrained_model.encode(input, &mut encoded);
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
