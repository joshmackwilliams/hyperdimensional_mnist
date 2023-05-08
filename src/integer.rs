use rand::{prelude::SliceRandom, Rng};
use rayon::prelude::*;

pub struct UntrainedIntegerHDModel {
    // Binary-valued quanta vectors
    quanta_vectors: Vec<u32>,
    // Binary-valued feature vectors
    feature_vectors: Vec<u32>,
    // Number of chunks in the model's feature vectors (actual dimensionality / 32)
    model_dimensionality_chunks: usize,
    // The actual model dimensionality
    model_dimensionality: usize,
    // Number of features in each input example
    input_dimensionality: usize,
    // Number of class vectors computed by the model
    n_classes: usize,
}

impl UntrainedIntegerHDModel {
    pub fn new(
        model_dimensionality_chunks: usize,
        input_dimensionality: usize,
        input_quanta: usize,
        n_classes: usize,
        // RNG for model initialization
        rng: &mut impl Rng,
    ) -> Self {
        // Compute the actual model dimensionality
        let model_dimensionality = model_dimensionality_chunks * 32;

        // Allocate space for quanta vectors
        let mut quanta_vectors: Vec<u32> =
            Vec::with_capacity(model_dimensionality_chunks * input_quanta);

        // Randomly generate the first quantum vector
        quanta_vectors.extend((0..model_dimensionality_chunks).map(|_| rng.gen::<u32>()));

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
                quanta_vectors.push(quanta_vectors[previous_quantum_offset_chunks + i]);
            }

            // Flip some bits
            for i in 0..flip_per_quantum {
                let index_to_flip = order_to_flip[i + (quantum - 1) * flip_per_quantum];
                let chunk_to_flip = index_to_flip / 32;
                let bit_to_flip = index_to_flip % 32;
                quanta_vectors[quantum_offset_chunks + chunk_to_flip] ^= 1 << bit_to_flip;
            }
        }

        // Allocate space for binary-valued feature vectors
        let mut feature_vectors =
            Vec::with_capacity(input_dimensionality * model_dimensionality_chunks);
        feature_vectors.resize_with(input_dimensionality * model_dimensionality_chunks, || {
            rng.gen::<u32>()
        });

        UntrainedIntegerHDModel {
            quanta_vectors,
            feature_vectors,
            model_dimensionality_chunks,
            model_dimensionality,
            input_dimensionality,
            n_classes,
        }
    }

    // Encode a batch of images
    pub fn encode(&self, input: &[Vec<usize>]) -> Vec<u32> {
        // Preallocate the output space
        let mut output = vec![0_u32; input.len() * self.model_dimensionality_chunks];
        // threshold = half the input dimensionality
        // This serves as the threshold for the majority function that we use on images
        let threshold = self.input_dimensionality as u32 >> 1;
        output
            // Compute one image vector at a time
            .par_chunks_mut(self.model_dimensionality_chunks)
            // Pair each vector with the corresponding image
            .zip(input.into_par_iter())
            .for_each(|(output, image)| {
                // We'll do this chunk-by-chunk - handling fixed-size chunks makes things easier
                for (chunk_index, chunk) in output.iter_mut().enumerate() {
                    // Temporary stack-based scratch space for our majority function
                    let mut counts: [u32; 32] = [0; 32];
                    // Compute this chunk of the feature vector for each pixel
                    image.iter().enumerate().for_each(|(i, value)| {
                        // The vector for a pixel is the position XOR the value
                        let xored_chunk = self.feature_vectors
                            [i * self.model_dimensionality_chunks + chunk_index]
                            ^ self.quanta_vectors
                                [*value * self.model_dimensionality_chunks + chunk_index];
                        // Separate out each bit and count it
                        count_bits_unsigned(xored_chunk, &mut counts);
                    });
                    // Condense our list of counts into a binarized bhunk and add it to the output
                    *chunk = binarize_unsigned_chunk(&counts, threshold);
                }
            });
        output
    }

    pub fn train(self, examples: &[u32], labels: &[usize]) -> IntegerHDModel {
        // Allocate space for integer-valued class vectors
        let class_vectors = vec![0_i32; self.n_classes * self.model_dimensionality];

        let mut model = IntegerHDModel {
            untrained_model: self,
            class_vectors,
            binary_class_vectors: Vec::new(),
        };

        examples
            .chunks(model.untrained_model.model_dimensionality_chunks)
            .zip(labels.iter())
            .for_each(|(example, &label)| {
                for i in 0..model.untrained_model.model_dimensionality {
                    model.class_vectors
                        [i + (label * model.untrained_model.model_dimensionality)] +=
                        bit_as_i32(example, i);
                }
            });

        let mut epoch: usize = 0;
        let mut best = 0_usize;
        let mut epochs_since_improvement = 0;
        while epochs_since_improvement < 5 {
            let mut correct = 0_usize;
            examples
                .chunks(model.untrained_model.model_dimensionality_chunks)
                .zip(labels.iter())
                .for_each(|(example, &label)| {
                    let predicted = model.classify_binary_from_integer(example);
                    if predicted != label {
                        for i in 0..model.untrained_model.model_dimensionality {
                            let bit = bit_as_i32(example, i);
                            model.class_vectors
                                [i + (label * model.untrained_model.model_dimensionality)] += bit;
                            model.class_vectors
                                [i + (predicted * model.untrained_model.model_dimensionality)] -=
                                bit;
                        }
                    } else {
                        correct += 1;
                    }
                });
            println!(
                "[Binary retraining] Correct examples at epoch {}: {}",
                epoch, correct
            );
            if correct < best {
                epochs_since_improvement += 1;
            } else {
                epochs_since_improvement = 0;
                best = correct;
            }
            epoch += 1;
        }

        model.binary_class_vectors = model
            .class_vectors
            .chunks(32)
            .map(|x| binarize_signed_chunk(x))
            .collect();

        let mut epoch: usize = 0;
        let mut best = 0_usize;
        let mut epochs_since_improvement = 0;
        while epochs_since_improvement < 5 {
            let mut correct = 0_usize;
            examples
                .chunks(model.untrained_model.model_dimensionality_chunks)
                .zip(labels.iter())
                .for_each(|(example, &label)| {
                    let predicted = model.classify(example);
                    if predicted != label {
                        for i in 0..model.untrained_model.model_dimensionality {
                            let bit = bit_as_i32(example, i);
                            model.class_vectors
                                [i + (label * model.untrained_model.model_dimensionality)] += bit;
                            model.class_vectors
                                [i + (predicted * model.untrained_model.model_dimensionality)] -=
                                bit;
                        }
                    } else {
                        correct += 1;
                    }
                });
            println!(
                "[Integer retraining] Correct examples at epoch {}: {}",
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

pub struct IntegerHDModel {
    untrained_model: UntrainedIntegerHDModel,
    class_vectors: Vec<i32>,
    binary_class_vectors: Vec<u32>,
}

impl IntegerHDModel {
    // Encode a batch of images
    pub fn encode(&self, input: &[Vec<usize>]) -> Vec<u32> {
        self.untrained_model.encode(input)
    }

    pub fn classify(&self, input: &[u32]) -> usize {
        self.class_vectors
            .chunks(self.untrained_model.model_dimensionality)
            .map(|x| square_cosine_similarity(input, x))
            .enumerate()
            .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
            .unwrap()
            .0
    }

    fn classify_binary_from_integer(&self, input: &[u32]) -> usize {
        self.class_vectors
            .chunks(self.untrained_model.model_dimensionality)
            .enumerate()
            .min_by_key(|(_, x)| hamming_distance_integer(input, x))
            .unwrap()
            .0
    }

    pub fn classify_binary(&self, input: &[u32]) -> usize {
        self.binary_class_vectors
            .chunks(self.untrained_model.model_dimensionality_chunks)
            .enumerate()
            .min_by_key(|(_, x)| hamming_distance(input, x))
            .unwrap()
            .0
    }
}

// Treat an integer vector as binary and compute the hamming distance
pub fn hamming_distance_integer(binary_vector: &[u32], integer_vector: &[i32]) -> u32 {
    let mut count = 0;
    for (chunk_index, &chunk) in binary_vector.iter().enumerate() {
        let chunk_offset = chunk_index << 5;
        let mut chunk_shifting = chunk;
        for bit_index in (0..32).rev() {
            let integer_value = integer_vector[chunk_offset + bit_index];
            count += ((integer_value as u32) ^ chunk_shifting) >> 31;
            chunk_shifting <<= 1;
        }
    }
    count
}

pub fn hamming_distance(a: &[u32], b: &[u32]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

// Majority function, operating on a list of counts and a threshold
// - counts[i] holds the number of examples in which bit i was set
// - threshold is set to half the number of examples
// - returns the inverse result of applying the majority function
fn binarize_unsigned_chunk(counts: &[u32], threshold: u32) -> u32 {
    let mut output = 0;
    for (bit, &count) in counts.iter().enumerate() {
        output |= ((count < threshold) as u32) << bit;
    }
    output
}

fn binarize_signed_chunk(chunk: &[i32]) -> u32 {
    let mut output = (chunk[0] & (1 << 31)) as u32;
    for &element in chunk.iter().skip(1) {
        output >>= 1;
        output |= (element as u32) & (1 << 31);
    }
    output
}

// Given a binarized chunk, separate each bit out and add it to an array of counts
fn count_bits_unsigned(chunk: u32, counts: &mut [u32]) {
    for (bit, count) in counts.iter_mut().enumerate() {
        *count += chunk >> (31 - bit) & 1;
    }
}

// Compute the magnitude of an integer vector
fn square_magnitude(a: &[i32]) -> i64 {
    a.iter().map(|x| (x * x) as i64).sum::<i64>()
}

// Compute the dot product of an integer vector with a binary vector
fn dot(binary_vec: &[u32], integer_vec: &[i32]) -> i64 {
    let mut sum: i64 = 0;

    // Iterate over the chunks
    for (chunk_index, &chunk) in binary_vec.iter().enumerate() {
        let chunk_bit_offset = chunk_index << 5;
        // Iterate over the bits
        for bit_index in 0..32 {
            // Grab the value of this feature from the integer vector
            let feature_value = integer_vec[bit_index + chunk_bit_offset];
            // If it needs to be negated, we'll do it manually
            // Hopefully saves some time over converting to 1 or -1 and doing a multiplication
            let xor_mask = ((chunk as i32) << (31 - bit_index)) >> 31;
            let add = xor_mask & 1;
            sum += ((feature_value ^ xor_mask) + add) as i64;
        }
    }

    sum
}

// Compute cosine similarity between and integer vector and a binary vector
pub fn square_cosine_similarity(a: &[u32], b: &[i32]) -> f64 {
    let d = dot(a, b);
    //(d * d * d.signum()) / (square_magnitude(b) / 16)
    (d as f64) / (square_magnitude(b) as f64).sqrt()
}

// Utility function to get a bit from a vector as -1 or 1
fn bit_as_i32(chunks: &[u32], bit_index: usize) -> i32 {
    let chunk = chunks[bit_index >> 5];
    let bit = bit_index & 31;
    ((chunk as i32) << (31 - bit) >> 31) | 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_as_i32() {
        let v = [0b10101010101010101010101010101010_u32];
        assert_eq!(bit_as_i32(&v, 3), -1);
        assert_eq!(bit_as_i32(&v, 4), 1);
        assert_eq!(bit_as_i32(&v, 0), 1);
    }

    #[test]
    pub fn test_dot() {
        let mut v = [0; 32];
        v[0..8].copy_from_slice(&[1, 2, -3, -4, 5, 6, -7, -8]);
        let w = [0b10110000];
        assert_eq!(dot(&w, &v), 1 + 2 - 3 - 4 - 5 - 6 - 7 + 8);
    }

    #[test]
    pub fn test_hamming_distance_integer() {
        let mut v = [1; 32];
        v[0..8].copy_from_slice(&[1, 2, -3, -4, 5, 6, -7, -8]);
        let w = [0b10110000];
        assert_eq!(hamming_distance_integer(&w, &v), 5);
    }

    #[test]
    pub fn test_binarize_signed_chunk() {
        let mut v = [1; 32];
        v[0..8].copy_from_slice(&[1, 2, -3, -4, 5, 6, -7, -8]);
        assert_eq!(binarize_signed_chunk(&v), 0b11001100);
    }
}
