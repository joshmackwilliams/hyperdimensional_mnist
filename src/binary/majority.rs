use crate::Chunk;
use ndarray::{ArrayView1, ArrayViewMut1};

// A fast approximation of the majority function that works on three items at a time
pub fn majority(values: &[ArrayView1<Chunk>], n_chunks: usize, mut output: ArrayViewMut1<Chunk>) {
    const MAX_RANKS: usize = 24;
    let mut ranks_a: [Chunk; MAX_RANKS] = [0; MAX_RANKS];
    let mut ranks_b: [Chunk; MAX_RANKS] = [0; MAX_RANKS];
    let mut a_filled = [false; MAX_RANKS];
    let mut b_filled = [false; MAX_RANKS];
    let n_ranks = 1 + (values.len() as f64).log(3.0).floor() as usize;
    for chunk in 0..n_chunks {
        ranks_a.fill(0);
        ranks_b.fill(0);
        a_filled.fill(false);
        b_filled.fill(false);
        for value in values {
            let mut chunk_contents = value[chunk];
            let mut rank = 0;
            loop {
                if !a_filled[rank] {
                    ranks_a[rank] = chunk_contents;
                    a_filled[rank] = true;
                    break;
                } else if !b_filled[rank] {
                    ranks_b[rank] = chunk_contents;
                    b_filled[rank] = true;
                    break;
                } else {
                    a_filled[rank] = false;
                    b_filled[rank] = false;
                    chunk_contents = (chunk_contents & ranks_a[rank])
                        | (chunk_contents & ranks_b[rank])
                        | (ranks_a[rank] & ranks_b[rank]);
                    rank += 1;
                }
            }
        }
        output[chunk] = ranks_a[n_ranks - 1];
    }
}

// Not used currently - the approximate majority function seems to work just as well
// UPDATE - this is because we got lucky. 60000 and (28 * 28) are both just above powers of 3
fn _majority(values: &[ArrayView1<Chunk>], n_chunks: usize, mut output: ArrayViewMut1<Chunk>) {
    let threshold = values.len() / 2;
    for chunk in 0..n_chunks {
        output[chunk] = 0;
        for bit in 0..Chunk::BITS {
            let mut count = 0;
            for value in values {
                count += (value[chunk] >> bit) & 1;
            }
            if count > threshold {
                output[chunk] |= 1 << bit;
            }
        }
    }
}
