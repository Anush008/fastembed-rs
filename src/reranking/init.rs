use super::DEFAULT_MAX_LENGTH;
use crate::{
    init::{HasMaxLength, InitOptionsWithLength},
    RerankerModel, TokenizerFiles,
};
use ort::{execution_providers::ExecutionProviderDispatch, session::Session};
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[derive(Debug)]
pub struct TextRerank {
    pub tokenizer: Tokenizer,
    pub(crate) session: Session,
    pub(crate) need_token_type_ids: bool,
}

impl HasMaxLength for RerankerModel {
    const MAX_LENGTH: usize = DEFAULT_MAX_LENGTH;
}

/// Options for initializing the reranking models
pub type RerankInitOptions = InitOptionsWithLength<RerankerModel>;

/// Options for initializing UserDefinedRerankerModel
///
/// Model files are held by the UserDefinedRerankerModel struct
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RerankInitOptionsUserDefined {
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub max_length: usize,
    /// Number of intra-op threads for ONNX Runtime. `None` (the default) uses
    /// every available CPU core via `std::thread::available_parallelism`.
    /// Set this to cap CPU usage (e.g. on laptops) at the cost of throughput.
    pub intra_threads: Option<usize>,
}

impl Default for RerankInitOptionsUserDefined {
    fn default() -> Self {
        Self {
            execution_providers: Default::default(),
            max_length: DEFAULT_MAX_LENGTH,
            intra_threads: None,
        }
    }
}

impl RerankInitOptionsUserDefined {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the execution providers for the model
    pub fn with_execution_providers(
        mut self,
        execution_providers: Vec<ExecutionProviderDispatch>,
    ) -> Self {
        self.execution_providers = execution_providers;
        self
    }

    /// Set the maximum sequence length
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// Set the number of intra-op threads ONNX Runtime uses. By default
    /// (`None`) all available CPU cores are used; capping this limits CPU
    /// usage at the cost of per-inference throughput.
    pub fn with_intra_threads(mut self, intra_threads: usize) -> Self {
        self.intra_threads = Some(intra_threads);
        self
    }
}

/// Convert RerankInitOptions to RerankInitOptionsUserDefined
///
/// This is useful for when the user wants to use the same options for both the default and user-defined models
impl From<RerankInitOptions> for RerankInitOptionsUserDefined {
    fn from(options: RerankInitOptions) -> Self {
        RerankInitOptionsUserDefined {
            execution_providers: options.execution_providers,
            max_length: options.max_length,
            intra_threads: options.intra_threads,
        }
    }
}

/// Enum for the source of the onnx file
///
/// User-defined models can either be in memory or on disk
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OnnxSource {
    Memory(Vec<u8>),
    File(PathBuf),
}

impl From<Vec<u8>> for OnnxSource {
    fn from(bytes: Vec<u8>) -> Self {
        OnnxSource::Memory(bytes)
    }
}

impl From<PathBuf> for OnnxSource {
    fn from(path: PathBuf) -> Self {
        OnnxSource::File(path)
    }
}

/// Struct for "bring your own" reranking models
///
/// The onnx_file and tokenizer_files are expecting the files' bytes
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct UserDefinedRerankingModel {
    pub onnx_source: OnnxSource,
    pub tokenizer_files: TokenizerFiles,
}

impl UserDefinedRerankingModel {
    pub fn new(onnx_source: impl Into<OnnxSource>, tokenizer_files: TokenizerFiles) -> Self {
        Self {
            onnx_source: onnx_source.into(),
            tokenizer_files,
        }
    }
}

/// Rerank result.
#[derive(Debug, PartialEq, Clone)]
pub struct RerankResult {
    pub document: Option<String>,
    pub score: f32,
    pub index: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn userdefined_builders_set_fields() {
        let o = RerankInitOptionsUserDefined::new()
            .with_max_length(128)
            .with_intra_threads(2);
        assert_eq!(o.max_length, 128);
        assert_eq!(o.intra_threads, Some(2));
    }
}
