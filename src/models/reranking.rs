#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RerankerModel {
    /// BAAI/bge-reranker-base
    BGERerankerBase,
}

pub fn reranker_model_list() -> Vec<RerankerModelInfo> {
    let reranker_model_list = vec![RerankerModelInfo {
        model: RerankerModel::BGERerankerBase,
        description: String::from("reranker model for english and chinese"),
        model_code: String::from("BAAI/bge-reranker-base"),
        model_file: String::from("onnx/model.onnx"),
    }];
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
