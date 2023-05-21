#![feature(portable_simd)]

pub mod mnist;
pub mod hd_model;
pub mod counting_binary_vector;
pub mod majority;
pub mod prune_data;

pub type ChunkElement = usize;
pub const CHUNK_ELEMENTS: usize = 8;
pub type BinaryChunk = std::simd::Simd<ChunkElement, CHUNK_ELEMENTS>;
pub const CHUNK_SIZE: usize = std::mem::size_of::<BinaryChunk>() * 8; // Multiply by 8 to get number of bits

use rand::Rng;
#[inline]
fn random_chunk(rng: &mut impl Rng) -> BinaryChunk {
    let mut d = BinaryChunk::default();
    let r: &mut [ChunkElement; CHUNK_ELEMENTS] = d.as_mut();
    r.fill_with(|| rng.gen::<ChunkElement>());
    d
}
