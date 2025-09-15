use std::{fmt::Display, str::FromStr};

use super::model_info::ModelInfo;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub enum ImageEmbeddingModel {
    /// Qdrant/clip-ViT-B-32-vision
    #[default]
    ClipVitB32,
    /// Qdrant/resnet50-onnx
    Resnet50,
    /// Qdrant/Unicom-ViT-B-16
    UnicomVitB16,
    /// Qdrant/Unicom-ViT-B-32
    UnicomVitB32,
    /// nomic-ai/nomic-embed-vision-v1.5
    NomicEmbedVisionV15,
}

pub fn models_list() -> Vec<ModelInfo<ImageEmbeddingModel>> {
    let models_list = vec![
        ModelInfo {
            model: ImageEmbeddingModel::ClipVitB32,
            dim: 512,
            description: String::from("CLIP vision encoder based on ViT-B/32"),
            model_code: String::from("Qdrant/clip-ViT-B-32-vision"),
            model_file: String::from("model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: ImageEmbeddingModel::Resnet50,
            dim: 2048,
            description: String::from("ResNet-50 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__."),
            model_code: String::from("Qdrant/resnet50-onnx"),
            model_file: String::from("model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: ImageEmbeddingModel::UnicomVitB16,
            dim: 768,
            description: String::from("Unicom Unicom-ViT-B-16 from open-metric-learning"),
            model_code: String::from("Qdrant/Unicom-ViT-B-16"),
            model_file: String::from("model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: ImageEmbeddingModel::UnicomVitB32,
            dim: 512,
            description: String::from("Unicom Unicom-ViT-B-32 from open-metric-learning"),
            model_code: String::from("Qdrant/Unicom-ViT-B-32"),
            model_file: String::from("model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
        ModelInfo {
            model: ImageEmbeddingModel::NomicEmbedVisionV15,
            dim: 768,
            description: String::from("Nomic NomicEmbedVisionV15"),
            model_code: String::from("nomic-ai/nomic-embed-vision-v1.5"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: Vec::new(),
            output_key: None,
        },
    ];

    // TODO: Use when out in stable
    // assert_eq!(
    //     std::mem::variant_count::<ImageEmbeddingModel>(),
    //     models_list.len(),
    //     "models::models() is not exhaustive"
    // );

    models_list
}

impl Display for ImageEmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let model_info = models_list()
            .into_iter()
            .find(|model| model.model == *self)
            .unwrap();
        write!(f, "{}", model_info.model_code)
    }
}

impl FromStr for ImageEmbeddingModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        models_list()
            .into_iter()
            .find(|m| m.model_code.eq_ignore_ascii_case(s))
            .map(|m| m.model)
            .ok_or_else(|| format!("Unknown embedding model: {s}"))
    }
}

impl TryFrom<String> for ImageEmbeddingModel {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}
