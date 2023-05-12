use std::ops::Deref;

use rand::{prelude::SliceRandom, Rng};

use crate::majority::fast_approx_majority;

use super::counting_binary_vector::CountingBinaryVector;
use super::{BinaryChunk, ChunkElement, CHUNK_ELEMENTS, CHUNK_SIZE};

// Separating the untrained version allows us to encode training data before actually training
#[derive(Clone)]
pub struct UntrainedHDModel {
    // Binary-valued feature x quanta vectors
    feature_quanta_vectors: Vec<BinaryChunk>,
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
        // Compute the actual model dimensionality
        let model_dimensionality = model_dimensionality_chunks * CHUNK_SIZE;

        // Allocate space for quanta vectors
        let mut feature_quanta_vectors: Vec<BinaryChunk> =
            Vec::with_capacity(input_dimensionality * input_quanta * model_dimensionality_chunks);

        for feature in 0..input_dimensionality {
            let feature_offset_chunks = feature * input_quanta * model_dimensionality_chunks;

            // Randomly generate the first quantum vector
            feature_quanta_vectors
                .extend((0..model_dimensionality_chunks).map(|_| random_chunk(rng)));

            // Randomly order some bits to flip
            let mut order_to_flip: Vec<usize> = (0..model_dimensionality).collect();
            order_to_flip.shuffle(rng);
            let flip_per_quantum = model_dimensionality / 2 / (input_quanta - 1);

            // Make a vector for each quanta
            for quantum in 1..input_quanta {
                // Some useful values
                let previous_quantum = quantum - 1;
                let quantum_offset_chunks = quantum * model_dimensionality_chunks;
                let previous_quantum_offset_chunks = previous_quantum * model_dimensionality_chunks;

                // Copy the previous quanta
                for i in 0..model_dimensionality_chunks {
                    feature_quanta_vectors.push(
                        feature_quanta_vectors
                            [feature_offset_chunks + previous_quantum_offset_chunks + i],
                    );
                }

                // Flip some bits
                for i in 0..flip_per_quantum {
                    let index_to_flip = order_to_flip[i + (quantum - 1) * flip_per_quantum];
                    let chunk_to_flip = index_to_flip / CHUNK_SIZE;
                    let element_to_flip =
                        (index_to_flip % CHUNK_SIZE) / ChunkElement::BITS as usize;
                    let bit_to_flip = index_to_flip % ChunkElement::BITS as usize;
                    let m: &mut [ChunkElement; CHUNK_ELEMENTS] = feature_quanta_vectors
                        [feature_offset_chunks + quantum_offset_chunks + chunk_to_flip]
                        .as_mut();
                    m[element_to_flip] ^= 1 << bit_to_flip;
                }
            }
        }

        // Allocate space for binary-valued feature vectors
        let mut feature_vectors =
            Vec::with_capacity(input_dimensionality * model_dimensionality_chunks);
        feature_vectors.resize_with(input_dimensionality * model_dimensionality_chunks, || {
            random_chunk(rng)
        });

        UntrainedHDModel {
            feature_quanta_vectors,
            model_dimensionality_chunks,
            n_classes,
            input_quanta,
        }
    }

    // Encode a batch of images (multithreaded)
    pub fn encode(&self, input: &[impl Deref<Target = [usize]>]) -> Vec<BinaryChunk> {
        // Preallocate the output space
        let mut output =
            vec![BinaryChunk::default(); input.len() * self.model_dimensionality_chunks];
        output
            // Compute one image vector at a time
            .chunks_mut(self.model_dimensionality_chunks)
            // Pair each vector with the corresponding image
            .zip(input.iter())
            .for_each(|(output, image)| {
                output
                    .iter_mut()
                    .enumerate()
                    .for_each(|(chunk_index, chunk)| {
                        *chunk = fast_approx_majority(image.iter().enumerate().map(
                            |(feature, &value)| {
                                self.feature_quanta_vectors[((feature * self.input_quanta + value)
                                    * self.model_dimensionality_chunks)
                                    + chunk_index]
                            },
                        ));
                    });
            });
        output
    }

    // Consume self and return a trained model, which can be used to classify examples
    pub fn train(self, examples: &[BinaryChunk], labels: &[usize]) -> HDModel {
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

        HDModel {
            untrained_model: self,
            class_vectors,
        }
    }

    pub fn mutate(&self, flip_levels: usize, rng: &mut impl Rng) -> UntrainedHDModel {
        UntrainedHDModel {
            feature_quanta_vectors: self
                .feature_quanta_vectors
                .iter()
                .map(|v| mutate_chunk(*v, flip_levels, rng))
                .collect(),
            model_dimensionality_chunks: self.model_dimensionality_chunks,
            n_classes: self.n_classes,
            input_quanta: self.input_quanta,
        }
    }
}

// Trained version of the model
// Actually just a wrapper around the untrained model, along with the trained class vectors
// Note - The counting vectors are included here so that we can keep training online.
//        This project does no such thing yet, but it could be useful as some point.
//        Besides, the counting vectors are actually pretty small.
#[derive(Clone)]
pub struct HDModel {
    untrained_model: UntrainedHDModel,
    class_vectors: Vec<CountingBinaryVector>,
}

impl HDModel {
    // Retraininng is separated from training so we can skip it in hypervector design
    pub fn retrain(&mut self, examples: &[BinaryChunk], labels: &[usize]) {
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
                .chunks(self.untrained_model.model_dimensionality_chunks)
                .zip(labels.iter())
                .for_each(|(example, &label)| {
                    let predicted = self.classify(example);

                    // If we get it wrong, reinforce the involved class vectors
                    if predicted != label {
                        self.class_vectors[label].add(example);
                        self.class_vectors[predicted].subtract(example);
                        missed += 1;
                    }
                });

            // Print status
            //println!("[Epoch {}] Misclassified: {}", epoch, missed);

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

    // Encode a batch of images (wrapper around untrained model)
    pub fn encode(&self, input: &[Vec<usize>]) -> Vec<BinaryChunk> {
        self.untrained_model.encode(input)
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

    pub fn mutate(&self, flip_levels: usize, rng: &mut impl Rng) -> UntrainedHDModel {
        self.untrained_model.mutate(flip_levels, rng)
    }

    pub fn evaluate(&self, examples: &[BinaryChunk], labels: &[usize]) -> usize {
        examples
            .chunks(self.untrained_model.model_dimensionality_chunks)
            .zip(labels.iter())
            .filter(|(example, &label)| self.classify(example) == label)
            .count()
    }

    pub fn count_total_class_hamming(&self) -> u32 {
        let mut sum = 0_u32;
        for i in 0..self.untrained_model.n_classes {
            for j in i..self.untrained_model.n_classes {
                sum += hamming_distance(self.class_vectors[i].as_binary(), self.class_vectors[j].as_binary());
            }
        }
        sum
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

#[inline]
fn random_chunk(rng: &mut impl Rng) -> BinaryChunk {
    let mut d = BinaryChunk::default();
    let r: &mut [ChunkElement; CHUNK_ELEMENTS] = d.as_mut();
    r.fill_with(|| rng.gen::<ChunkElement>());
    d
}

#[inline]
fn make_mutation_mask(flip_levels: usize, rng: &mut impl Rng) -> BinaryChunk {
    let mut d = random_chunk(rng);
    for _ in 0..(flip_levels - 1) {
        d &= random_chunk(rng);
    }
    d
}

#[inline]
fn mutate_chunk(d: BinaryChunk, flip_levels: usize, rng: &mut impl Rng) -> BinaryChunk {
    d ^ make_mutation_mask(flip_levels, rng)
}
