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
        let weight: Vec<f32> = weight_view
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
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
