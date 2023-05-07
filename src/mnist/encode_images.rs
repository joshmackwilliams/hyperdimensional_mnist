use crate::binary::majority;
use crate::Chunk;
use ndarray::{parallel::prelude::*, s, Array2, ArrayView1, ArrayView3, Axis};

// Encode each of a vector of images (as loaded from the csv file).
// Expects a 3D array of feature vectors such that feature_vecs[i][j][k] is
// the kth chunk of the feature vector corresponding to pixel j at intensity i.
pub fn encode_images(
    images: Vec<Vec<usize>>,
    feature_vecs: ArrayView3<Chunk>,
    n_chunks: usize,
) -> Array2<Chunk> {
    assert!(!images.is_empty(), "No images to encode");
    let image_size = images[0].len();

    // Allocate our output array
    let mut output = Array2::<Chunk>::zeros((images.len(), image_size));
    output
        // For some reason, rows_mut doesn't work with parallelism, but this is equivalent
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        // Pair each output row with the corresponding image
        .zip(images.into_par_iter())
        // Use the for_each_with to keep a vector as per-thread "scratch space"
        .for_each_with(
            Vec::<ArrayView1<Chunk>>::new(),
            |image_features, (output, image)| {
                // The clear() function is used to clear the vector without reallocating
                image_features.clear();
                // Then, we extend it (all done in place) with the new features
                image_features.extend(
                    (0..image_size)
                        .map(|feature| feature_vecs.slice(s![image[feature], feature, ..])),
                );
                // Now, we compute the majority of all of the computed vectors
                majority(image_features, n_chunks, output);
            },
        );
    output
}
