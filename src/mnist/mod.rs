// This module contains all the MNIST-specific code in the project

mod load_mnist;
pub use load_mnist::load_mnist;

mod encode_images;
pub use encode_images::encode_images;