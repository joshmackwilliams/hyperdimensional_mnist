use crate::Chunk;
use ndarray::Array2;
use rand::{prelude::SliceRandom, Rng};

// Makes vectors for a discrete numerical feature
// Each vector differs from adjacent vectors by only a few bits
pub fn make_quanta_vectors(n: usize, n_chunks: usize, mut rng: &mut impl Rng) -> Array2<Chunk> {
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
