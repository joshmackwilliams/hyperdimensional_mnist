use crate::Chunk;
use ndarray::{ArrayView1, ArrayView2};
use crate::binary::hamming;

pub fn classify(x: ArrayView1<Chunk>, class_vectors: ArrayView2<Chunk>) -> usize {
    class_vectors
        .rows()
        .into_iter()
        .enumerate()
        .min_by_key(|(_, vector)| hamming(x, *vector))
        .unwrap()
        .0
}
