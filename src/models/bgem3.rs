use crate::ModelInfo;
use std::{fmt::Display, str::FromStr};

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub enum Bgem3Model {
    /// gpahal/bge-m3-onnx-int8
    #[default]
    BGEM3Q,
}

pub fn models_list() -> Vec<ModelInfo<Bgem3Model>> {
    vec![
        ModelInfo {
            model: Bgem3Model::BGEM3Q,
            dim: 1024,
            description: String::from("BGE-M3 model quantized to INT8, outputs dense, sparse, and ColBERT embeddings in a single forward pass"),
            model_code: String::from("gpahal/bge-m3-onnx-int8"),
            model_file: String::from("model_quantized.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        }
    ]
}

impl Display for Bgem3Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let model_info = models_list()
            .into_iter()
            .find(|model| model.model == *self)
            .ok_or(std::fmt::Error)?;
        write!(f, "{}", model_info.model_code)
    }
}

impl FromStr for Bgem3Model {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        models_list()
            .into_iter()
            .find(|m| m.model_code.eq_ignore_ascii_case(s))
            .map(|m| m.model)
            .ok_or_else(|| format!("Unknown BGEM3 model: {s}"))
    }
}

impl TryFrom<String> for Bgem3Model {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

#[cfg(test)]
pub(crate) fn all_variants() -> Vec<Bgem3Model> {
    fn _exhaustive_guard(m: &Bgem3Model) {
        match m {
            Bgem3Model::BGEM3Q => (),
        }
    }
    vec![Bgem3Model::BGEM3Q]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_has_model_info() {
        let listed: Vec<_> = models_list().into_iter().map(|i| i.model).collect();
        for variant in all_variants() {
            assert!(
                listed.contains(&variant),
                "{variant:?} is missing from models_list(); get_model_info would panic"
            );
        }
    }
}
