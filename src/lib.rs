#![feature(portable_simd)]

pub mod mnist;
pub mod hd_model;
pub mod counting_binary_vector;

pub type BinaryChunk = std::simd::usizex8;
pub type ChunkElement = usize;
pub const CHUNK_ELEMENTS: usize = 8;
pub const CHUNK_SIZE: usize = ChunkElement::BITS as usize * CHUNK_ELEMENTS;