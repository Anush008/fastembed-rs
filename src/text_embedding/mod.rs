//! Text embedding module, containing the main struct [TextEmbedding] and its
//! initialization options.

use crate::models::text_embedding::EmbeddingModel;

// Constants.
const DEFAULT_BATCH_SIZE: usize = 256;
const DEFAULT_MAX_LENGTH: usize = 512;
const DEFAULT_EMBEDDING_MODEL: EmbeddingModel = EmbeddingModel::BGESmallENV15;

// Output precedence and transforming functions.
pub mod output;

// Initialization options.
mod init;
pub use init::*;

// The implementation of the embedding models.
mod r#impl;
