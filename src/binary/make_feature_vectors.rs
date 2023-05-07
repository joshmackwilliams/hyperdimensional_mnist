use crate::binary::{make_quanta_vectors, make_random_vectors};
use crate::Chunk;
use ndarray::{Array2, Array3};
use rand::Rng;

// This function needs to be refactored to be more reusable
pub fn make_feature_vectors(
    n_vectors: usize,
    n_chunks: usize,
    rng: &mut impl Rng,
) -> Array3<Chunk> {
    let intensity_vecs: Array2<Chunk> = make_quanta_vectors(256, n_chunks, rng);
    let pos_vecs: Array2<Chunk> = make_random_vectors(n_vectors, n_chunks, rng);

    let mut feature_vecs: Array3<Chunk> = Array3::zeros((256, n_vectors, n_chunks));
    for pos in 0..n_vectors {
        for chunk in 0..n_chunks {
            for intensity in 0..256 {
                feature_vecs[(intensity, pos, chunk)] =
                    pos_vecs[(pos, chunk)] ^ intensity_vecs[(intensity, chunk)];
            }
        }
    }

    feature_vecs
}
