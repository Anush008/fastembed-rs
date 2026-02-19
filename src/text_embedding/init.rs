//! Initialization options for the text embedding models.
//!

use crate::{
    common::TokenizerFiles,
    init::{HasMaxLength, InitOptionsWithLength},
    pooling::Pooling,
    EmbeddingModel, OutputKey, QuantizationMode,
};
use ort::{execution_providers::ExecutionProviderDispatch, session::Session};
use tokenizers::Tokenizer;

use super::DEFAULT_MAX_LENGTH;

impl HasMaxLength for EmbeddingModel {
    const MAX_LENGTH: usize = DEFAULT_MAX_LENGTH;
}

/// Options for initializing the TextEmbedding model
pub type TextInitOptions = InitOptionsWithLength<EmbeddingModel>;

/// Options for initializing UserDefinedEmbeddingModel
///
/// Model files are held by the UserDefinedEmbeddingModel struct
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct InitOptionsUserDefined {
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub max_length: usize,
}

impl InitOptionsUserDefined {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
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

impl Default for InitOptionsUserDefined {
    fn default() -> Self {
        Self {
            execution_providers: Default::default(),
            max_length: DEFAULT_MAX_LENGTH,
        }
    }
}

/// Convert InitOptions to InitOptionsUserDefined
///
/// This is useful for when the user wants to use the same options for both the default and user-defined models
impl From<TextInitOptions> for InitOptionsUserDefined {
    fn from(options: TextInitOptions) -> Self {
        InitOptionsUserDefined {
            execution_providers: options.execution_providers,
            max_length: options.max_length,
        }
    }
}

/// Struct for "bring your own" embedding models
///
/// The onnx_file and tokenizer_files are expecting the files' bytes
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserDefinedEmbeddingModel {
    pub onnx_file: Vec<u8>,
    pub external_initializers: Vec<ExternalInitializerFile>,
    pub tokenizer_files: TokenizerFiles,
    pub pooling: Option<Pooling>,
    pub quantization: QuantizationMode,
    pub output_key: Option<OutputKey>,
}

/// Struct for adding external initializers to "bring your own" embedding models
///
/// The buffer is expecting the data of the external initializer and the file_name
/// must match the one referenced by the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalInitializerFile {
    pub file_name: String,
    pub buffer: Vec<u8>,
}

impl UserDefinedEmbeddingModel {
    pub fn new(onnx_file: Vec<u8>, tokenizer_files: TokenizerFiles) -> Self {
        Self {
            onnx_file,
            external_initializers: Vec::new(),
            tokenizer_files,
            quantization: QuantizationMode::None,
            pooling: None,
            output_key: None,
        }
    }

    pub fn with_quantization(mut self, quantization: QuantizationMode) -> Self {
        self.quantization = quantization;
        self
    }

    pub fn with_pooling(mut self, pooling: Pooling) -> Self {
        self.pooling = Some(pooling);
        self
    }

    pub fn with_external_initializer(mut self, file_name: String, buffer: Vec<u8>) -> Self {
        self.external_initializers
            .push(ExternalInitializerFile { file_name, buffer });
        self
    }
}

/// Rust representation of the TextEmbedding model
pub struct TextEmbedding {
    pub tokenizer: Tokenizer,
    pub(crate) pooling: Option<Pooling>,
    pub(crate) session: Session,
    pub(crate) need_token_type_ids: bool,
    pub(crate) quantization: QuantizationMode,
    pub(crate) output_key: Option<OutputKey>,
}
