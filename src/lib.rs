#![feature(portable_simd)]

pub mod mnist;
pub mod hd_model;
pub mod counting_binary_vector;

pub type BinaryChunk = std::simd::usizex4;
pub const CHUNK_SIZE: usize = std::mem::size_of::<BinaryChunk>();