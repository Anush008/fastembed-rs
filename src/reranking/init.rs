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
}

impl Default for RerankInitOptionsUserDefined {
    fn default() -> Self {
        Self {
            execution_providers: Default::default(),
            max_length: DEFAULT_MAX_LENGTH,
        }
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
