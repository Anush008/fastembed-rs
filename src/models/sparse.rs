use std::{fmt::Display, str::FromStr};

use crate::ModelInfo;

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub enum SparseModel {
    /// prithivida/Splade_PP_en_v1
    #[default]
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
        output_key: None,
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

impl FromStr for SparseModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        models_list()
            .into_iter()
            .find(|m| m.model_code.eq_ignore_ascii_case(s))
            .map(|m| m.model)
            .ok_or_else(|| format!("Unknown sparse model: {s}"))
    }
}

impl TryFrom<String> for SparseModel {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}
