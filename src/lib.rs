pub mod mnist;
pub mod hd_model;
pub mod counting_binary_vector;
pub mod majority;
pub mod prune_data;

pub type ChunkElement = u64;
pub type BinaryChunk = wide::u64x4;