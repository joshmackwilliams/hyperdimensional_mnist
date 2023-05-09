use rand::{prelude::SliceRandom, Rng};
use rayon::prelude::*;

pub type BinaryChunk = u64;
pub type Integer = i16;

pub struct UntrainedHDModel {
    // Binary-valued feature x quanta vectors
    feature_quanta_vectors: Vec<BinaryChunk>,
    // Number of chunks in the model's feature vectors (actual dimensionality / binary chunk size)
    model_dimensionality_chunks: usize,
    // The actual model dimensionality
    model_dimensionality: usize,
    // Number of features in each input example
    input_dimensionality: usize,
    // Number of class vectors computed by the model
    n_classes: usize,
    // Number of quanta for each feature
    input_quanta: usize,
}

impl UntrainedHDModel {
    pub fn new(
        model_dimensionality_chunks: usize,
        input_dimensionality: usize,
        input_quanta: usize,
        n_classes: usize,
        // RNG for model initialization
        rng: &mut impl Rng,
    ) -> Self {
        // Compute the actual model dimensionality
        let model_dimensionality = model_dimensionality_chunks * BinaryChunk::BITS as usize;

        // Allocate space for quanta vectors
        let mut feature_quanta_vectors: Vec<BinaryChunk> =
            Vec::with_capacity(input_dimensionality * input_quanta * model_dimensionality_chunks);

        for feature in 0..input_dimensionality {
            let feature_offset_chunks = feature * input_quanta * model_dimensionality_chunks;

            // Randomly generate the first quantum vector
            feature_quanta_vectors
                .extend((0..model_dimensionality_chunks).map(|_| rng.gen::<BinaryChunk>()));

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
                    let chunk_to_flip = index_to_flip / BinaryChunk::BITS as usize;
                    let bit_to_flip = index_to_flip % BinaryChunk::BITS as usize;
                    feature_quanta_vectors
                        [feature_offset_chunks + quantum_offset_chunks + chunk_to_flip] ^=
                        1 << bit_to_flip;
                }
            }
        }

        // Allocate space for binary-valued feature vectors
        let mut feature_vectors =
            Vec::with_capacity(input_dimensionality * model_dimensionality_chunks);
        feature_vectors.resize_with(input_dimensionality * model_dimensionality_chunks, || {
            rng.gen::<BinaryChunk>()
        });

        UntrainedHDModel {
            feature_quanta_vectors,
            model_dimensionality_chunks,
            model_dimensionality,
            input_dimensionality,
            n_classes,
            input_quanta,
        }
    }

    // Encode a batch of images
    pub fn encode(&self, input: &[Vec<usize>]) -> Vec<BinaryChunk> {
        // Preallocate the output space
        let mut output = vec![0 as BinaryChunk; input.len() * self.model_dimensionality_chunks];
        output
            // Compute one image vector at a time
            .par_chunks_mut(self.model_dimensionality_chunks)
            // Pair each vector with the corresponding image
            .zip(input.into_par_iter())
            .for_each_with(
                Vec::with_capacity(self.input_dimensionality),
                |chunks: &mut Vec<BinaryChunk>, (output, image)| {
                    for (chunk_index, chunk) in output.iter_mut().enumerate() {
                        chunks.clear();
                        chunks.extend(image.iter().enumerate().map(|(feature, &value)| {
                            self.feature_quanta_vectors[(feature
                                * self.input_quanta
                                * self.model_dimensionality_chunks)
                                + (value * self.model_dimensionality_chunks)
                                + chunk_index]
                        }));
                        *chunk = fast_approx_majority(chunks);
                    }
                },
            );
        output
    }

    pub fn train(self, examples: &[BinaryChunk], labels: &[usize]) -> HDModel {
        // Allocate space for integer-valued class vectors
        let mut class_vectors = vec![0 as Integer; self.n_classes * self.model_dimensionality];

        examples
            .chunks(self.model_dimensionality_chunks)
            .zip(labels.iter())
            .for_each(|(example, &label)| {
                for i in 0..self.model_dimensionality {
                    let label_index = i + (label * self.model_dimensionality);
                    class_vectors[label_index] =
                        class_vectors[label_index].saturating_add(bit_as_integer(example, i));
                }
            });

        let n_classes = self.n_classes;
        let model_dimensionality_chunks = self.model_dimensionality_chunks;

        let mut model = HDModel {
            untrained_model: self,
            binary_class_vectors: vec![0 as BinaryChunk; n_classes * model_dimensionality_chunks],
        };

        binarize(&class_vectors, &mut model.binary_class_vectors);

        let mut epoch: usize = 0;
        let mut best = 0_usize;
        let mut epochs_since_improvement = 0;
        while epochs_since_improvement < 5 {
            let mut correct = 0_usize;
            examples
                .chunks(model.untrained_model.model_dimensionality_chunks)
                .zip(labels.iter())
                .for_each(|(example, &label)| {
                    let predicted = model.classify_binary(example);
                    if predicted != label {
                        for i in 0..model.untrained_model.model_dimensionality {
                            let bit = bit_as_integer(example, i);
                            let label_index =
                                i + (label * model.untrained_model.model_dimensionality);
                            class_vectors[label_index] =
                                class_vectors[label_index].saturating_add(bit);
                            let predicted_index =
                                i + (predicted * model.untrained_model.model_dimensionality);
                            class_vectors[predicted_index] =
                                class_vectors[predicted_index].saturating_sub(bit);
                        }
                        binarize(&class_vectors, &mut model.binary_class_vectors);
                    } else {
                        correct += 1;
                    }
                });
            println!(
                "[Retraining] Correct examples at epoch {}: {}",
                epoch, correct
            );
            if correct > best {
                epochs_since_improvement = 0;
                best = correct;
            } else {
                epochs_since_improvement += 1;
            }
            epoch += 1;
        }

        model
    }
}

pub struct HDModel {
    untrained_model: UntrainedHDModel,
    binary_class_vectors: Vec<BinaryChunk>,
}

impl HDModel {
    // Encode a batch of images
    pub fn encode(&self, input: &[Vec<usize>]) -> Vec<BinaryChunk> {
        self.untrained_model.encode(input)
    }

    pub fn classify_binary(&self, input: &[BinaryChunk]) -> usize {
        self.binary_class_vectors
            .chunks(self.untrained_model.model_dimensionality_chunks)
            .enumerate()
            .min_by_key(|(_, x)| hamming_distance(input, x))
            .unwrap()
            .0
    }
}

fn binarize(input: &[Integer], target: &mut [BinaryChunk]) {
    target
        .iter_mut()
        .zip(
            input
                .chunks(BinaryChunk::BITS as usize)
                .map(|x| binarize_signed_chunk(x)),
        )
        .for_each(|(x, y)| *x = y);
}

// Treat an integer vector as binary and compute the hamming distance
pub fn hamming_distance_integer(binary_vector: &[BinaryChunk], integer_vector: &[Integer]) -> u32 {
    let mut count = 0;
    for (chunk_index, &chunk) in binary_vector.iter().enumerate() {
        let chunk_offset = chunk_index * BinaryChunk::BITS as usize;
        for bit_index in 0..BinaryChunk::BITS as usize {
            let integer_value = integer_vector[chunk_offset + bit_index];
            count += (((integer_value >> Integer::BITS - 1) as BinaryChunk ^ (chunk >> bit_index))
                & 1) as u32;
        }
    }
    count
}

pub fn hamming_distance(a: &[BinaryChunk], b: &[BinaryChunk]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

fn binarize_signed_chunk(chunk: &[Integer]) -> BinaryChunk {
    let mut output = 0;
    for (bit, &value) in chunk.iter().enumerate() {
        output |= (((value >> (Integer::BITS - 1)) & 1) as BinaryChunk) << bit;
    }
    output
}

// Compute the magnitude of an integer vector
fn square_magnitude(a: &[Integer]) -> i64 {
    a.iter()
        .map(|&x| {
            let x = x as i64;
            x * x
        })
        .sum::<i64>()
}

// Compute the dot product of an integer vector with a binary vector
fn dot(binary_vec: &[BinaryChunk], integer_vec: &[Integer]) -> i64 {
    let mut sum: i64 = 0;

    // Iterate over the chunks
    for (chunk_index, &chunk) in binary_vec.iter().enumerate() {
        let chunk_bit_offset = chunk_index * BinaryChunk::BITS as usize;
        // Iterate over the bits
        for bit_index in 0..BinaryChunk::BITS as usize {
            // Grab the value of this feature from the integer vector
            let feature_value = integer_vec[bit_index + chunk_bit_offset];
            let multiplier = ((chunk >> bit_index) & 1) as i64 * -2 + 1;
            sum += feature_value as i64 * multiplier;
        }
    }

    sum
}

// Compute cosine similarity between and integer vector and a binary vector
pub fn square_cosine_similarity(a: &[BinaryChunk], b: &[Integer]) -> f64 {
    let d = dot(a, b);
    //(d * d * d.signum()) / (square_magnitude(b) / 16)
    (d as f64) / (square_magnitude(b) as f64).sqrt()
}

// Utility function to get a bit from a vector as -1 or 1
fn bit_as_integer(chunks: &[BinaryChunk], bit_index: usize) -> Integer {
    let chunk = chunks[bit_index / BinaryChunk::BITS as usize];
    let bit = bit_index % BinaryChunk::BITS as usize;
    (((chunk >> bit) & 1) as Integer) * -2 + 1
}

fn fast_approx_majority(chunks: &[BinaryChunk]) -> BinaryChunk {
    if chunks.len() == 1 {
        return chunks[0];
    }
    let one_third_ceil = (chunks.len() + 2) / 3;
    let one_third_floor = chunks.len() / 3;
    let a = fast_approx_majority(&chunks[..one_third_ceil]);
    let b = fast_approx_majority(&chunks[one_third_floor..(one_third_floor + one_third_ceil)]);
    let c = fast_approx_majority(&chunks[(chunks.len() - one_third_ceil)..]);
    return (a & b) | (b & c) | (c & a);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_as_i32() {
        let v = [0b10101010101010101010101010101010 as BinaryChunk];
        assert_eq!(bit_as_integer(&v, 3), -1);
        assert_eq!(bit_as_integer(&v, 4), 1);
        assert_eq!(bit_as_integer(&v, 0), 1);
    }

    #[test]
    pub fn test_dot() {
        let mut v = [0; BinaryChunk::BITS as usize];
        v[0..8].copy_from_slice(&[1, 2, -3, -4, 5, 6, -7, -8]);
        let w = [0b10110000];
        assert_eq!(dot(&w, &v), 1 + 2 - 3 - 4 - 5 - 6 - 7 + 8);
    }

    #[test]
    pub fn test_hamming_distance_integer() {
        let mut v = [1; BinaryChunk::BITS as usize];
        v[0..8].copy_from_slice(&[1, 2, -3, -4, 5, 6, -7, -8]);
        let w = [0b10110000];
        assert_eq!(hamming_distance_integer(&w, &v), 5);
    }

    #[test]
    pub fn test_binarize_signed_chunk() {
        let mut v = [1; BinaryChunk::BITS as usize];
        v[0..8].copy_from_slice(&[1, 2, -3, -4, 5, 6, -7, -8]);
        assert_eq!(binarize_signed_chunk(&v), 0b11001100);
    }
}
