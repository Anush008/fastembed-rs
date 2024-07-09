//! [FastEmbed](https://github.com/Anush008/fastembed-rs) - Fast, light, accurate library built for retrieval embedding generation.
//!
//! The library provides the TextEmbedding struct to interface with text embedding models.
//!
//! ### Instantiating [TextEmbedding](crate::TextEmbedding)
//! ```
//! use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};
//!
//!# fn model_demo() -> anyhow::Result<()> {
//! // With default InitOptions
//! let model = TextEmbedding::try_new(Default::default())?;
//!
//! // List all supported models
//! dbg!(TextEmbedding::list_supported_models());
//!
//! // With custom InitOptions
//! let model = TextEmbedding::try_new(InitOptions {
//!     model_name: EmbeddingModel::BGEBaseENV15,
//!     show_download_progress: false,
//!     ..Default::default()
//! })?;
//! # Ok(())
//! # }
//! ```
//! Find more info about the available options in the [InitOptions](crate::InitOptions) documentation.
//!
//! ### Embeddings generation
//!```
//!# use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};
//!# fn embedding_demo() -> anyhow::Result<()> {
//!# let model: TextEmbedding = TextEmbedding::try_new(Default::default())?;
//! let documents = vec![
//!    "passage: Hello, World!",
//!    "query: Hello, World!",
//!    "passage: This is an example passage.",
//!    // You can leave out the prefix but it's recommended
//!    "fastembed-rs is licensed under MIT"
//!    ];
//!
//! // Generate embeddings with the default batch size, 256
//! let embeddings = model.embed(documents, None)?;
//!
//! println!("Embeddings length: {}", embeddings.len()); // -> Embeddings length: 4
//! # Ok(())
//! # }
//! ```
//!

mod common;
mod models;
mod reranking;
mod text_embedding;

#[cfg(test)]
mod tests;

pub use ort::ExecutionProviderDispatch;

pub use crate::common::{read_file_to_bytes, Embedding};
pub use crate::models::reranking::{RerankerModel, RerankerModelInfo};
pub use crate::models::text_embedding::{EmbeddingModel, ModelInfo};
pub use crate::reranking::{RerankInitOptions, RerankResult, TextRerank};
pub use crate::text_embedding::{
    InitOptions, InitOptionsUserDefined, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel,
};
