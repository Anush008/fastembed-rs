#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RerankerModel {
    /// BAAI/bge-reranker-base
    BGERerankerBase,
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
        },
        RerankerModelInfo {
            model: RerankerModel::JINARerankerV1TurboEn,
            description: String::from("reranker model for English"),
            model_code: String::from("jinaai/jina-reranker-v1-turbo-en"),
            model_file: String::from("onnx/model.onnx"),
        },
        RerankerModelInfo {
            model: RerankerModel::JINARerankerV2BaseMultiligual,
            description: String::from("reranker model for multilingual"),
            model_code: String::from("jinaai/jina-reranker-v2-base-multilingual"),
            model_file: String::from("onnx/model.onnx"),
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
}
