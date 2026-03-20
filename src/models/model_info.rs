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
}
