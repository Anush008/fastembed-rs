//! Text embedding module, containing the main struct [TextEmbedding] and its
//! initialization options.

// Constants.
const DEFAULT_BATCH_SIZE: usize = 256;
const DEFAULT_MAX_LENGTH: usize = 512;

// Output precedence and transforming functions.
pub mod output;

// Initialization options.
mod init;
pub use init::*;

// The implementation of the embedding models.
mod r#impl;
