// BGE-M3 sparse linear layer weights
// Loaded from sparse_linear.safetensors (converted from BAAI/bge-m3 sparse_linear.pt)
// token_weight = ReLU(hidden_state @ weight + bias)

use safetensors::SafeTensors;
use std::sync::OnceLock;

static WEIGHTS: OnceLock<Bgem3SparseWeights> = OnceLock::new();

pub struct Bgem3SparseWeights {
    pub weight: Vec<f32>,
    pub bias: f32,
}

impl Bgem3SparseWeights {
    fn load() -> Self {
        const SAFETENSORS_DATA: &[u8] = include_bytes!("weights/sparse_linear.safetensors");

        let tensors = SafeTensors::deserialize(SAFETENSORS_DATA)
            .expect("Failed to deserialize sparse_linear.safetensors");

        let weight_view = tensors.tensor("weight").expect("Missing 'weight' tensor");
        let (weight_chunks, weight_remainder) = weight_view.data().as_chunks::<4>();
        assert!(
            weight_remainder.is_empty(),
            "'weight' tensor byte length is not divisible by 4"
        );
        let weight: Vec<f32> = weight_chunks
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();

        let bias_view = tensors.tensor("bias").expect("Missing 'bias' tensor");
        let bias = f32::from_le_bytes([
            bias_view.data()[0],
            bias_view.data()[1],
            bias_view.data()[2],
            bias_view.data()[3],
        ]);

        Self { weight, bias }
    }
}

pub fn get_weights() -> &'static Bgem3SparseWeights {
    WEIGHTS.get_or_init(Bgem3SparseWeights::load)
}
