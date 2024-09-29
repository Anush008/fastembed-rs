use std::fmt::Display;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RerankerModel {
    /// BAAI/bge-reranker-base
    BGERerankerBase,
    /// rozgo/bge-reranker-v2-m3
    BGERerankerV2M3,
    /// jinaai/jina-reranker-v1-turbo-en
    JINARerankerV1TurboEn,
    /// jinaai/jina-reranker-v2-base-multilingual
    JINARerankerV2BaseMultiligual,
}

pub fn reranker_model_list() -> Vec<RerankerModelInfo> {
    let reranker_model_list = vec![
        RerankerModelInfo {
            model: RerankerModel::BGERerankerBase,
            description: String::from("reranker model for English and Chinese"),
            model_code: String::from("BAAI/bge-reranker-base"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: vec![],
        },
        RerankerModelInfo {
            model: RerankerModel::BGERerankerV2M3,
            description: String::from("reranker model for multilingual"),
            model_code: String::from("rozgo/bge-reranker-v2-m3"),
            model_file: String::from("model.onnx"),
            additional_files: vec![String::from("model.onnx.data")],
        },
        RerankerModelInfo {
            model: RerankerModel::JINARerankerV1TurboEn,
            description: String::from("reranker model for English"),
            model_code: String::from("jinaai/jina-reranker-v1-turbo-en"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: vec![],
        },
        RerankerModelInfo {
            model: RerankerModel::JINARerankerV2BaseMultiligual,
            description: String::from("reranker model for multilingual"),
            model_code: String::from("jinaai/jina-reranker-v2-base-multilingual"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: vec![],
        },
    ];
    reranker_model_list
}

/// Data struct about the available reanker models
#[derive(Debug, Clone)]
pub struct RerankerModelInfo {
    pub model: RerankerModel,
    pub description: String,
    pub model_code: String,
    pub model_file: String,
    pub additional_files: Vec<String>,
}

impl Display for RerankerModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let model_info = reranker_model_list()
            .into_iter()
            .find(|model| model.model == *self)
            .expect("Model not found in supported models list.");
        write!(f, "{}", model_info.model_code)
    }
}
