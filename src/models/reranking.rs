use variant_count::VariantCount;

#[derive(Debug, Clone, PartialEq, Eq, VariantCount)]
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
    assert_eq!(
        RerankerModel::VARIANT_COUNT,
        reranker_model_list.len(),
        "models::models() is not exhaustive"
    );

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
