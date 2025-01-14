use std::fmt::Display;

use crate::ModelInfo;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseModel {
    /// prithivida/Splade_PP_en_v1
    SPLADEPPV1,
}

pub fn models_list() -> Vec<ModelInfo<SparseModel>> {
    vec![ModelInfo {
        model: SparseModel::SPLADEPPV1,
        dim: 0,
        description: String::from("Splade sparse vector model for commercial use, v1"),
        model_code: String::from("Qdrant/Splade_PP_en_v1"),
        model_file: String::from("model.onnx"),
        additional_files: Vec::new(),
    }]
}

impl Display for SparseModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let model_info = models_list()
            .into_iter()
            .find(|model| model.model == *self)
            .unwrap();
        write!(f, "{}", model_info.model_code)
    }
}
