use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use rand::{prelude::SliceRandom, Rng};

type IntegerValue = i32;

pub struct IntegerHDModel {
    quanta_vectors: Array2<bool>,
    feature_vectors: Array2<bool>,
    class_vectors: Array2<IntegerValue>,
    model_dimensionality: usize,
    input_dimensionality: usize,
    input_quanta: usize,
    n_classes: usize,
}

impl IntegerHDModel {
    pub fn new(
        model_dimensionality: usize,
        input_dimensionality: usize,
        input_quanta: usize,
        n_classes: usize,
        rng: &mut impl Rng,
    ) -> Self {
        let mut quanta_vectors: Array2<bool> = Array::default((input_quanta, model_dimensionality));
        for i in 0..model_dimensionality {
            quanta_vectors[(0, i)] = rng.gen();
        }

        let mut order_to_flip: Vec<usize> = (0..model_dimensionality).collect();
        order_to_flip.shuffle(rng);
        let flip_per_quantum = model_dimensionality / 2 / (input_quanta - 1);

        for quantum in 1..input_quanta {
            for i in 0..model_dimensionality {
                quanta_vectors[(quantum, i)] = quanta_vectors[(quantum - 1, i)];
            }
            for i in 0..flip_per_quantum {
                let index_to_flip = order_to_flip[i + (quantum - 1) * flip_per_quantum];
                quanta_vectors[(quantum, index_to_flip)] ^= true;
            }
        }

        let feature_vectors: Array2<bool> =
            Array::from_shape_simple_fn((input_dimensionality, model_dimensionality), || rng.gen());

        let class_vectors: Array2<IntegerValue> = Array::zeros((n_classes, model_dimensionality));

        IntegerHDModel {
            quanta_vectors,
            feature_vectors,
            class_vectors,
            model_dimensionality,
            input_dimensionality,
            input_quanta,
            n_classes,
        }
    }

    pub fn encode(&self, input: &[Vec<usize>]) -> Array2<bool> {
        let mut output: Array2<bool> =
            Array::default((input.len(), self.model_dimensionality));
        output
            // For some reason, rows_mut doesn't work with parallelism, but this is equivalent
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            // Pair each output row with the corresponding image
            .zip(input.into_par_iter())
            .for_each(|(mut output, image)| {
                for position in 0..self.model_dimensionality {
                    let mut count: usize = 0;
                    image.iter().enumerate().for_each(|(i, value)| {
                        count += (self.feature_vectors[(i, position)]
                            ^ self.quanta_vectors[(*value, position)]) as usize
                    });
                    output[position] = count > image.len() / 2;
                }
            });
        output
    }

    pub fn train(&mut self, examples: ArrayView2<bool>, labels: ArrayView1<IntegerValue>) {
        examples
            .rows()
            .into_iter()
            .zip(labels.iter())
            .for_each(|(example, label)| {
                self.class_vectors.slice_mut(s![*label as usize, ..]).iter_mut().zip(example.iter()).for_each(|(element, example)| {
                    *element += (*example as IntegerValue) * 2 - 1;
                });
            });

        let mut epoch = 0;
        let mut best = 0_usize;
        let mut epochs_since_improvement = 0;
        while epochs_since_improvement < 5 {
            let mut correct = 0_usize;
            examples
                .rows()
                .into_iter()
                .zip(labels.iter())
                .for_each(|(example, label)| {
                    let predicted = self.classify(example);
                    if predicted != (*label as usize) {
                        self.class_vectors.slice_mut(s![*label as usize, ..]).iter_mut().zip(example.iter()).for_each(|(element, example)| {
                            *element += (*example as IntegerValue) * 2 - 1;
                        });
                        self.class_vectors.slice_mut(s![predicted as usize, ..]).iter_mut().zip(example.iter()).for_each(|(element, example)| {
                            *element -= (*example as IntegerValue) * 2 - 1;
                        });
                    } else {
                        correct += 1;
                    }
                });
            println!("Correct examples at epoch {}: {}", epoch, correct);
            if correct < best {
                epochs_since_improvement += 1;
            } else {
                epochs_since_improvement = 0;
                best = correct;
            }
            epoch += 1;
        }
    }

    pub fn classify(&self, input: ArrayView1<bool>) -> usize {
        self.class_vectors
            .rows()
            .into_iter()
            .enumerate()
            .max_by_key(|(_, x)| cosine_similarity(input, *x))
            .unwrap()
            .0
    }
}

fn magnitude(a: ArrayView1<i32>) -> f64 {
    let square_sum: f64 = a
        .iter()
        .map(|x| {
            let x: f64 = (*x).into();
            x * x
        })
        .sum();
    square_sum.sqrt()
}

fn dot(a: ArrayView1<bool>, b: ArrayView1<i32>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x as u8 as f64) * 2.0 - 1.0) * (*y as f64))
        .sum()
}

pub fn cosine_similarity(
    a: ArrayView1<bool>,
    b: ArrayView1<i32>,
) -> u32 {
    ((dot(a, b) / magnitude(b)) * 1048576.0) as u32
}
