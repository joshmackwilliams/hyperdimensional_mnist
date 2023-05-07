// Code relating specifically to binary vectors

mod majority;
pub use majority::majority;

mod hamming;
pub use hamming::hamming;

mod make_quanta_vectors;
pub use make_quanta_vectors::make_quanta_vectors;

mod make_random_vectors;
pub use make_random_vectors::make_random_vectors;

mod make_feature_vectors;
pub use make_feature_vectors::make_feature_vectors;

mod classify;
pub use classify::classify;