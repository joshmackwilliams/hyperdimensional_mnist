pub mod mnist;
pub mod hd_model;
pub mod counting_binary_vector;

pub type BinaryChunk = usize; // Seems to run fastest when this value matches arch size