use crate::Chunk;
use ndarray::Array2;
use rand::Rng;

// Batch-generates random feature vectors
pub fn make_random_vectors(n: usize, n_chunks: usize, rng: &mut impl Rng) -> Array2<Chunk> {
    Array2::from_shape_simple_fn((n, n_chunks), || rng.gen())
}
