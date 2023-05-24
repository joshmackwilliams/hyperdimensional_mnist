use super::BinaryChunk;

// Number of bits in each (signed) counter.
// 8 is the lowest it can go without accuracy degradation.
const COUNTER_BITS: usize = 12;

// A binary vector that can efficiently compute the majority of its components.
//
// We store the bits of each counter across several parallel binary chunks, to take
// advantage of parallelism implicit in the CPU's binary instructions.
//
// That is, a set of 64 8-bit counters is stored as 8 64-bit numbers.
// The first holds the LSB of all counters, the second holds the next bit, and so on
// until the final chunk holds the sign bit.
//
// All numbers are stored as two's complement.
// Note that all additions and subtractions are saturating, not overflowing.
#[derive(Clone)]
pub struct CountingBinaryVector {
    data: Vec<BinaryChunk>,
    n_chunks: usize,
}

impl CountingBinaryVector {
    // 0-initialize the vector
    pub fn new(n_chunks: usize) -> Self {
        Self {
            data: vec![BinaryChunk::default(); n_chunks * COUNTER_BITS],
            n_chunks,
        }
    }

    // Get the sign bits to use in hamming distance calculations
    pub fn as_binary(&self) -> &[BinaryChunk] {
        &self.data[((COUNTER_BITS - 1) * self.n_chunks)..(COUNTER_BITS * self.n_chunks)]
    }

    // Add a binary vector to this vector
    pub fn add(&mut self, chunks: impl IntoIterator<Item = BinaryChunk>) {
        for (chunk_index, chunk) in chunks.into_iter().enumerate() {
            // Counter intuitive, but in this project, 0 represents positive and 1 represents negative
            // Intuition here is that we're storing the sign values of negatives
            self.add_chunk(chunk_index, chunk)
        }
    }

    #[inline]
    fn add_chunk(&mut self, chunk_index: usize, chunk: BinaryChunk) {
        // Saturation check
        // We are saturated where the sign bit matches the operator, and all other bits do not match the operator
        let sign_bits = self.data[(COUNTER_BITS - 1) * self.n_chunks + chunk_index];
        let mut mask = sign_bits ^ chunk;
        for bit in 0..(COUNTER_BITS - 1) {
            mask |= !(self.data[bit * self.n_chunks + chunk_index] ^ chunk);
        }

        // Do the operation
        for bit in 0..COUNTER_BITS {
            let current_value = &mut self.data[bit * self.n_chunks + chunk_index];
            let carry = *current_value ^ chunk;
            *current_value ^= mask;
            mask &= carry;
        }
    }
}
