use ort::session::Session;
use tokenizers::Tokenizer;

use crate::{
    init::{HasMaxLength, InitOptionsWithLength},
    models::sparse::SparseModel,
    TokenizerFiles,
};

use super::DEFAULT_MAX_LENGTH;

impl HasMaxLength for SparseModel {
    const MAX_LENGTH: usize = DEFAULT_MAX_LENGTH;
}

/// Options for initializing the SparseTextEmbedding model
pub type SparseInitOptions = InitOptionsWithLength<SparseModel>;

/// Struct for "bring your own" embedding models
///
/// The onnx_file and tokenizer_files are expecting the files' bytes
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct UserDefinedSparseModel {
    pub onnx_file: Vec<u8>,
    pub tokenizer_files: TokenizerFiles,
}

impl UserDefinedSparseModel {
    pub fn new(onnx_file: Vec<u8>, tokenizer_files: TokenizerFiles) -> Self {
        Self {
            onnx_file,
            tokenizer_files,
        }
    }
}

/// Rust representation of the SparseTextEmbedding model
pub struct SparseTextEmbedding {
    pub tokenizer: Tokenizer,
    pub(crate) session: Session,
    pub(crate) need_token_type_ids: bool,
    pub(crate) model: SparseModel,
}
