use super::BinaryChunk;

const MAX_RANKS: u32 = 8;

#[inline]
pub fn fast_approx_majority(
    input: impl ExactSizeIterator<Item = BinaryChunk> + Clone,
) -> BinaryChunk {
    let mut values = [BinaryChunk::default(); (2 * MAX_RANKS as usize - 1) + 1];
    let mut size = 0;

    let original_count = input.len();
    let mut actual_count = 1;
    while actual_count < original_count {
        actual_count *= 3;
    }
    let mut input = input.cycle().take(actual_count);

    for step in 1_usize..=(actual_count / 3) {
        let a = input.next().unwrap();
        let b = input.next().unwrap();
        let c = input.next().unwrap();
        values[size] = (a & b) | (b & c) | (c & a);
        size += 1;
        let mut step_copy = step;
        while step_copy % 3 == 0 {
            step_copy /= 3;
            let a = values[size - 1];
            let b = values[size - 2];
            let c = values[size - 3];
            values[size - 3] = (a & b) | (b & c) | (c & a);
            size -= 2;
        }
    }

    debug_assert!(size == 1);
    values[0]
}
