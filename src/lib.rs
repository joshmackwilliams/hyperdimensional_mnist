#![feature(portable_simd)]

pub mod mnist;
pub mod hd_model;
pub mod counting_binary_vector;

pub type ChunkElement = usize;
pub const CHUNK_ELEMENTS: usize = 8;
pub type BinaryChunk = std::simd::Simd<ChunkElement, CHUNK_ELEMENTS>;
pub const CHUNK_SIZE: usize = std::mem::size_of::<BinaryChunk>() * 8; // Multiply by 8 to get number of bits