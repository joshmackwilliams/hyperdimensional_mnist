use super::BinaryChunk;

// Number of bits in each (signed) counter.
// 8 is the lowest it can go without accuracy degradation.
// This parameter has a significant impact on training performance.
const COUNTER_BITS: usize = 8;

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
            data: vec![0; n_chunks * COUNTER_BITS],
            n_chunks,
        }
    }

    // Get the sign bits to use in hamming distance calculations
    pub fn as_binary(&self) -> &[BinaryChunk] {
        &self.data[((COUNTER_BITS - 1) * self.n_chunks)..(COUNTER_BITS * self.n_chunks)]
    }

    // Add a binary vector to this vector
    pub fn add(&mut self, chunks: &[BinaryChunk]) {
        for (chunk_index, &chunk) in chunks.iter().enumerate() {
            // Counter intuitive, but in this project, 0 represents positive and 1 represents negative
            // Intuition here is that we're storing the sign values of negatives
            self.increment(chunk_index, !chunk);
            self.decrement(chunk_index, chunk);
        }
    }

    // Subtract a binary vector from this vector (same as adding the inverse, but we invert it inline here)
    pub fn subtract(&mut self, chunks: &[BinaryChunk]) {
        for (chunk_index, &chunk) in chunks.iter().enumerate() {
            self.decrement(chunk_index, !chunk);
            self.increment(chunk_index, chunk);
        }
    }

    // Increment the counters at all positions where the given chunk is "1", ignoring zeros.
    fn increment(&mut self, chunk_index: usize, mut chunk: BinaryChunk) {
        // Check for positive saturation
        // First, we can only have positive saturation where the sign bit is 0
        let mut positive_saturation = !self.data[(COUNTER_BITS - 1) * self.n_chunks + chunk_index];
        // Then, we can only have it where all other bits are 1
        for bit in 0..(COUNTER_BITS - 1) {
            positive_saturation &= self.data[bit * self.n_chunks + chunk_index];
        }
        // Don't do addition in saturated bits
        chunk &= !positive_saturation;

        // Do the addition
        for bit in 0..COUNTER_BITS {
            let bit_offset = bit * self.n_chunks;
            let current_value = self.data[bit_offset + chunk_index];
            self.data[bit_offset + chunk_index] = current_value ^ chunk;
            chunk = current_value & chunk;
        }
    }

    // As above, but decrement instead of incrementing.
    fn decrement(&mut self, chunk_index: usize, mut chunk: BinaryChunk) {
        // Check for negative saturation
        // This is exactly the opposite of the positive saturation check in the increment function
        let mut negative_saturation = self.data[(COUNTER_BITS - 1) * self.n_chunks + chunk_index];
        for bit in 0..(COUNTER_BITS - 1) {
            negative_saturation &= !self.data[bit * self.n_chunks + chunk_index];
        }
        // Don't do subtraction in saturated bits
        chunk &= !negative_saturation;

        // Do the subtraction. Now, instead of being "carry", our chunk is "borrow"
        for bit in 0..COUNTER_BITS {
            let bit_offset = bit * self.n_chunks;
            let current_value = self.data[bit_offset + chunk_index];
            self.data[bit_offset + chunk_index] = current_value ^ chunk;
            chunk = (!current_value) & chunk;
        }
    }
}
