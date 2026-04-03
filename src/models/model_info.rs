use crate::{OutputKey, RerankerModel};

/// Data struct about the available models
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ModelInfo<T> {
    pub model: T,
    pub dim: usize,
    pub description: String,
    pub model_code: String,
    pub model_file: String,
    pub additional_files: Vec<String>,
    pub output_key: Option<OutputKey>,
}

/// Data struct about the available reranker models
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RerankerModelInfo {
    pub model: RerankerModel,
    pub description: String,
    pub model_code: String,
    pub model_file: String,
    pub additional_files: Vec<String>,
    /// If true, this model is too large for automated CI tests and is excluded from the default
    /// test loop. Users can still load it explicitly.
    pub large: bool,
    /// Optional prompt template for models that require a specific input format.
    /// Use `{query}` and `{doc}` as placeholders. When set, pairs are formatted
    /// as a single sequence instead of using the tokenizer's native pair encoding.
    pub prompt_template: Option<String>,
}
