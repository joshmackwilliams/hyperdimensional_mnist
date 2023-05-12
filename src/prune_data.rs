use std::ops::Deref;

// Functions to prune our data for efficiency
// We prune any position where 98% of the data has the same value

pub fn find_prunable_positions(data: &[impl Deref<Target = [usize]>], quanta: usize) -> Vec<usize> {
    // Assume that the dimensionality of all data is the same
    let dimensionality = data[0].len();
    let num_examples = data.len();
    let threshold = (0.98 * num_examples as f64).floor() as usize;

    // Build a histogram of values occurring in each pixel
    let mut histogram = vec![0_usize; dimensionality * quanta];
    data.iter()
        .flat_map(|image| image.iter().enumerate())
        .for_each(|(i, value)| {
            histogram[i * quanta + value] += 1;
        });

    // Find all fields where 98% of data share the same value
    histogram
        .chunks(quanta)
        .enumerate()
        .filter(|(_, values)| values.iter().any(|&value| value >= threshold))
        .map(|(i, _)| i)
        .collect()
}

pub fn prune_data(data: &[impl Deref<Target = [usize]>], positions: &[usize]) -> Vec<Vec<usize>> {
    data.iter()
        .map(|image| {
            let mut skip_pos = 0;
            image
                .iter()
                .enumerate()
                .filter_map(|(i, &value)| {
                    if i == positions[skip_pos] {
                        skip_pos += 1;
                        None
                    } else {
                        Some(value)
                    }
                })
                .collect()
        })
        .collect()
}
