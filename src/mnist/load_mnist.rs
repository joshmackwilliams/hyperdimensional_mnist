// Load MNIST images from a csv file.
// Returns a tuple of (images, labels)
// The expected format is:
// - No headers
// - One image per row
// - Each row starts with the class label 0-9
// - The rest of the row consists of 28x28 pixel values
// - The pixel values are represented as integers, 0-255
pub fn load_mnist(filename: &str, n_examples: usize) -> (Vec<Vec<usize>>, Vec<usize>) {
    // Use the CSV crate
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(filename)
        .expect("Failed to open CSV file");

    // We use parallel arrays to store the data for easy handling later on
    reader
        .records()
        .map(|result| {
            let record = result.expect("Failed to parse CSV record");
            let label = record[0].parse::<usize>().expect("Failed to parse label");
            let image = record
                .iter()
                .skip(1) // Skip the label
                .map(|x| x.parse::<usize>().expect("Failed to parse pixel") / 26) // 10 quanta
                .collect();
            (image, label)
        })
        .take(n_examples)
        .unzip()
}
