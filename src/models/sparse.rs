use crate::{common::SparseEmbedding, ModelInfo};
use ndarray::{ArrayViewD, Axis, CowArray, Dim};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseModel {
    /// Splade PP Sparse vector model Qdrant/Splade_PP_en_v1
    SPLADEPPV1,
}

pub fn models_list() -> Vec<ModelInfo<SparseModel>> {
    vec![ModelInfo {
        model: SparseModel::SPLADEPPV1,
        dim: 0,
        description: String::from("Splade sparse vector model for commercial use, v1"),
        model_code: String::from("Qdrant/Splade_PP_en_v1"),
        model_file: String::from("model.onnx"),
    }]
}

impl SparseModel {
    pub fn post_process(
        &self,
        model_output: &ArrayViewD<f32>,
        attention_mask: &CowArray<i64, Dim<[usize; 2]>>,
    ) -> Vec<SparseEmbedding> {
        match self {
            SparseModel::SPLADEPPV1 => {
                // Apply ReLU and logarithm transformation
                let relu_log = model_output.mapv(|x| (1.0 + x.max(0.0)).ln());

                // Convert to f32 and expand the dimensions
                let attention_mask = attention_mask.mapv(|x| x as f32).insert_axis(Axis(2));

                // Weight the transformed values by the attention mask
                let weighted_log = relu_log * attention_mask;

                // Get the max scores
                let scores = weighted_log.fold_axis(Axis(1), f32::NEG_INFINITY, |r, &v| r.max(v));

                scores
                    .rows()
                    .into_iter()
                    .map(|row_scores| {
                        let mut values: Vec<f32> = Vec::new();
                        // Is a usize big enough?
                        let mut indices: Vec<usize> = Vec::new();

                        row_scores.into_iter().enumerate().for_each(|(idx, f)| {
                            if f.is_sign_positive() {
                                values.push(*f);
                                indices.push(idx);
                            }
                        });

                        SparseEmbedding { values, indices }
                    })
                    .collect()
            }
        }
    }
}
