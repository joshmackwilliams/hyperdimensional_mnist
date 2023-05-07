use crate::Chunk;
use ndarray::ArrayView1;

// Computes bitwise hamming distance between two binary vectors
pub fn hamming(a: ArrayView1<Chunk>, b: ArrayView1<Chunk>) -> usize {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| (a ^ b).count_ones() as usize)
        .sum()
}
