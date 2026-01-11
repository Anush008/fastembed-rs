use crate::common::TokenizerFiles;
use crate::init::{HasMaxLength, InitOptionsWithLength};
use ort::{execution_providers::ExecutionProviderDispatch, session::Session};
use std::collections::HashSet;
use tokenizers::Tokenizer;

use super::DEFAULT_MAX_LENGTH;

/// Supported late interaction models
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub enum LateInteractionModel {
    /// colbert-ir/colbertv2.0
    #[default]
    ColBERTV2,
    /// answerdotai/answerai-colbert-small-v1
    AnswerAIColBERTSmallV1,
}

impl HasMaxLength for LateInteractionModel {
    const MAX_LENGTH: usize = DEFAULT_MAX_LENGTH;
}

/// Options for initializing late interaction models
pub type LateInteractionInitOptions = InitOptionsWithLength<LateInteractionModel>;

/// Options for user-defined late interaction models
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LateInteractionInitOptionsUserDefined {
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub max_length: usize,
}

impl Default for LateInteractionInitOptionsUserDefined {
    fn default() -> Self {
        Self {
            execution_providers: Default::default(),
            max_length: DEFAULT_MAX_LENGTH,
        }
    }
}

impl LateInteractionInitOptionsUserDefined {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_execution_providers(
        mut self,
        execution_providers: Vec<ExecutionProviderDispatch>,
    ) -> Self {
        self.execution_providers = execution_providers;
        self
    }

    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }
}

impl From<LateInteractionInitOptions> for LateInteractionInitOptionsUserDefined {
    fn from(options: LateInteractionInitOptions) -> Self {
        Self {
            execution_providers: options.execution_providers,
            max_length: options.max_length,
        }
    }
}

/// User-defined late interaction model
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserDefinedLateInteractionModel {
    pub onnx_file: Vec<u8>,
    pub tokenizer_files: TokenizerFiles,
    pub query_marker_token_id: u32,
    pub document_marker_token_id: u32,
    pub mask_token: String,
    pub min_query_length: usize,
    pub dim: usize,
}

impl UserDefinedLateInteractionModel {
    pub fn new(
        onnx_file: Vec<u8>,
        tokenizer_files: TokenizerFiles,
        query_marker_token_id: u32,
        document_marker_token_id: u32,
        mask_token: String,
        min_query_length: usize,
        dim: usize,
    ) -> Self {
        Self {
            onnx_file,
            tokenizer_files,
            query_marker_token_id,
            document_marker_token_id,
            mask_token,
            min_query_length,
            dim,
        }
    }
}

/// Data struct for late interaction model info
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LateInteractionModelInfo {
    pub model: LateInteractionModel,
    pub dim: usize,
    pub description: String,
    pub model_code: String,
    pub model_file: String,
    pub additional_files: Vec<String>,
    pub query_marker_token_id: u32,
    pub document_marker_token_id: u32,
    pub mask_token: String,
    pub min_query_length: usize,
}

/// Late interaction embedding output - variable length per document
pub type LateInteractionEmbedding = Vec<Vec<f32>>;

/// Rust representation of a late interaction embedding model
pub struct LateInteractionTextEmbedding {
    pub(crate) tokenizer: Tokenizer,
    pub(crate) query_tokenizer: Tokenizer,
    pub(crate) session: Session,
    pub(crate) need_token_type_ids: bool,
    pub(crate) query_marker_token_id: u32,
    pub(crate) document_marker_token_id: u32,
    #[expect(dead_code, reason = "Used to initialise tokeniser")]
    pub(crate) mask_token_id: u32,
    pub(crate) pad_token_id: u32,
    pub(crate) skip_list: HashSet<u32>,
    #[expect(dead_code, reason = "Used to initialise tokeniser")]
    pub(crate) min_query_length: usize,
    pub(crate) dim: usize,
}
