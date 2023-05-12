use super::BinaryChunk;

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
    // Number of bits in each (signed) counter.
    // This parameter has a significant impact on training performance.
    n_bits: usize,
}

impl CountingBinaryVector {
    // 0-initialize the vector
    pub fn new(n_chunks: usize, n_bits: usize) -> Self {
        Self {
            data: vec![BinaryChunk::default(); n_chunks * n_bits],
            n_chunks,
            n_bits,
        }
    }

    // Get the sign bits to use in hamming distance calculations
    pub fn as_binary(&self) -> &[BinaryChunk] {
        &self.data[((self.n_bits - 1) * self.n_chunks)..(self.n_bits * self.n_chunks)]
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
    #[inline]
    fn increment(&mut self, chunk_index: usize, mut chunk: BinaryChunk) {
        // Check for positive saturation
        // First, we can only have positive saturation where the sign bit is 0
        let mut positive_saturation = !self.data[(self.n_bits - 1) * self.n_chunks + chunk_index];
        // Then, we can only have it where all other bits are 1
        for bit in 0..(self.n_bits - 1) {
            positive_saturation &= self.data[bit * self.n_chunks + chunk_index];
        }
        // Don't do addition in saturated bits
        chunk &= !positive_saturation;

        // Do the addition
        for bit in 0..self.n_bits {
            let bit_offset = bit * self.n_chunks;
            let current_value = self.data[bit_offset + chunk_index];
            self.data[bit_offset + chunk_index] = current_value ^ chunk;
            chunk = current_value & chunk;
        }
    }

    // As above, but decrement instead of incrementing.
    #[inline]
    fn decrement(&mut self, chunk_index: usize, mut chunk: BinaryChunk) {
        // Check for negative saturation
        // This is exactly the opposite of the positive saturation check in the increment function
        let mut negative_saturation = self.data[(self.n_bits - 1) * self.n_chunks + chunk_index];
        for bit in 0..(self.n_bits - 1) {
            negative_saturation &= !self.data[bit * self.n_chunks + chunk_index];
        }
        // Don't do subtraction in saturated bits
        chunk &= !negative_saturation;

        // Do the subtraction. Now, instead of being "carry", our chunk is "borrow"
        for bit in 0..self.n_bits {
            let bit_offset = bit * self.n_chunks;
            let current_value = self.data[bit_offset + chunk_index];
            self.data[bit_offset + chunk_index] = current_value ^ chunk;
            chunk = (!current_value) & chunk;
        }
    }

    // Simmulate adding a vector many times, but use shifts to make the multiplication more
    // efficient than repeatedly calling add().
    // This will be useful when we do HD design, and need to rapidly compute class vectors
    pub fn add_multiplied(&mut self, chunks: &[BinaryChunk], multiplier: usize) {
        for (chunk_index, &chunk) in chunks.iter().enumerate() {
            let mut shift = 0;
            let mut multiplier = multiplier; // Local, mutable copy

            // Two variables to track saturated positions
            // This is necessary because, when adding shifted values, you may not perfectly saturate a count
            let mut positive_saturation = BinaryChunk::default();
            let mut negative_saturation = BinaryChunk::default();

            // Deconstruct the multiplier, bit by bit
            while multiplier != 0 {
                // Save LSB of the multiplier
                let first_bit = multiplier & 1;

                // First increment the loop variables
                multiplier >>= 1;
                shift += 1;

                // If LSB was 0, skip
                if first_bit == 0 {
                    continue;
                }

                // Add the shifted values
                positive_saturation |= self.increment_shifted(chunk_index, !chunk, shift - 1);
                negative_saturation |= self.decrement_shifted(chunk_index, chunk, shift - 1);
            }

            // Sataurate any now-saturated positions
            self.positive_saturate(chunk_index, positive_saturation);
            self.negative_saturate(chunk_index, negative_saturation);
        }
    }

    // Almost the same as increment(), except that it left-shifts the input before adding it
    // Returns a mask indicating which positions were saturated, but does not saturate them
    #[inline]
    fn increment_shifted(
        &mut self,
        chunk_index: usize,
        mut chunk: BinaryChunk,
        lshift: usize,
    ) -> BinaryChunk {
        let mut positive_saturation = !self.data[(self.n_bits - 1) * self.n_chunks + chunk_index];
        for bit in lshift..(self.n_bits - 1) {
            positive_saturation &= self.data[bit * self.n_chunks + chunk_index];
        }
        chunk &= !positive_saturation;

        for bit in lshift..self.n_bits {
            let bit_offset = bit * self.n_chunks;
            let current_value = self.data[bit_offset + chunk_index];
            self.data[bit_offset + chunk_index] = current_value ^ chunk;
            chunk = current_value & chunk;
        }
        positive_saturation
    }

    #[inline]
    fn decrement_shifted(
        &mut self,
        chunk_index: usize,
        mut chunk: BinaryChunk,
        lshift: usize,
    ) -> BinaryChunk {
        let mut negative_saturation = self.data[(self.n_bits - 1) * self.n_chunks + chunk_index];
        for bit in lshift..(self.n_bits - 1) {
            negative_saturation &= !self.data[bit * self.n_chunks + chunk_index];
        }
        chunk &= !negative_saturation;

        for bit in lshift..self.n_bits {
            let bit_offset = bit * self.n_chunks;
            let current_value = self.data[bit_offset + chunk_index];
            self.data[bit_offset + chunk_index] = current_value ^ chunk;
            chunk = (!current_value) & chunk;
        }
        negative_saturation
    }

    // Positive-saturate the positions where the chunk has a 1
    #[inline]
    fn positive_saturate(&mut self, chunk_index: usize, chunk: BinaryChunk) {
        // Set sign bits to 0
        self.data[(self.n_bits - 1) * self.n_chunks + chunk_index] &= !chunk;
        // Set all others to 1
        for bit in 0..(self.n_bits - 1) {
            self.data[bit * self.n_chunks + chunk_index] |= chunk;
        }
    }

    // As above, but negative saturate (set signs to 1, all others to 0)
    #[inline]
    fn negative_saturate(&mut self, chunk_index: usize, chunk: BinaryChunk) {
        self.data[(self.n_bits - 1) * self.n_chunks + chunk_index] |= chunk;
        for bit in 0..(self.n_bits - 1) {
            self.data[bit * self.n_chunks + chunk_index] &= !chunk;
        }
    }
}
