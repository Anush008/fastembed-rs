//! [FastEmbed](https://github.com/Anush008/fastembed-rs) - Fast, light, accurate library built for retrieval embedding generation.
//!
//! The library provides the TextEmbedding struct to interface with text embedding models.
//!
#![cfg_attr(
    feature = "online",
    doc = r#"
 ### Instantiating [TextEmbedding](crate::TextEmbedding)
 ```
 use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};

# fn model_demo() -> anyhow::Result<()> {
 // With default InitOptions
 let model = TextEmbedding::try_new(Default::default())?;

 // List all supported models
 dbg!(TextEmbedding::list_supported_models());

 // With custom InitOptions
 let model = TextEmbedding::try_new(InitOptions {
     model_name: EmbeddingModel::BGEBaseENV15,
     show_download_progress: false,
     ..Default::default()
 })?;
 # Ok(())
 # }
 ```
"#
)]
//! Find more info about the available options in the [InitOptions](crate::InitOptions) documentation.
//!
#![cfg_attr(
    feature = "online",
    doc = r#"
 ### Embeddings generation
```
# use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};
# fn embedding_demo() -> anyhow::Result<()> {
# let model: TextEmbedding = TextEmbedding::try_new(Default::default())?;
 let documents = vec![
    "passage: Hello, World!",
    "query: Hello, World!",
    "passage: This is an example passage.",
    // You can leave out the prefix but it's recommended
    "fastembed-rs is licensed under MIT"
    ];

 // Generate embeddings with the default batch size, 256
 let embeddings = model.embed(documents, None)?;

 println!("Embeddings length: {}", embeddings.len()); // -> Embeddings length: 4
 # Ok(())
 # }
 ```
"#
)]

mod common;
mod models;
mod pooling;
mod reranking;
mod sparse_text_embedding;
mod text_embedding;

#[cfg(feature = "online")]
#[cfg(test)]
mod tests;

pub use ort::ExecutionProviderDispatch;

pub use crate::common::{read_file_to_bytes, Embedding, Error, SparseEmbedding, TokenizerFiles};
pub use crate::models::reranking::{RerankerModel, RerankerModelInfo};
pub use crate::models::{model_info::ModelInfo, text_embedding::EmbeddingModel};
pub use crate::reranking::{
    RerankInitOptions, RerankInitOptionsUserDefined, RerankResult, TextRerank,
    UserDefinedRerankingModel,
};
pub use crate::sparse_text_embedding::{SparseInitOptions, SparseTextEmbedding};
pub use crate::text_embedding::{
    InitOptions, InitOptionsUserDefined, TextEmbedding, UserDefinedEmbeddingModel,
};
