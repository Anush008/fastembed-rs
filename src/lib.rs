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
 let model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
 )?;
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
mod image_embedding;
mod models;
pub mod output;
mod pooling;
mod reranking;
mod sparse_text_embedding;
mod text_embedding;

pub use ort::execution_providers::ExecutionProviderDispatch;

pub use crate::common::{
    read_file_to_bytes, Embedding, Error, SparseEmbedding, TokenizerFiles, DEFAULT_CACHE_DIR,
};
pub use crate::image_embedding::{
    ImageEmbedding, ImageInitOptions, ImageInitOptionsUserDefined, UserDefinedImageEmbeddingModel,
};
pub use crate::models::image_embedding::ImageEmbeddingModel;
pub use crate::models::reranking::{RerankerModel, RerankerModelInfo};
pub use crate::models::{
    model_info::ModelInfo, quantization::QuantizationMode, text_embedding::EmbeddingModel,
};
pub use crate::output::{EmbeddingOutput, OutputKey, OutputPrecedence, SingleBatchOutput};
pub use crate::pooling::Pooling;
pub use crate::reranking::{
    OnnxSource, RerankInitOptions, RerankInitOptionsUserDefined, RerankResult, TextRerank,
    UserDefinedRerankingModel,
};
pub use crate::sparse_text_embedding::{
    SparseInitOptions, SparseTextEmbedding, UserDefinedSparseModel,
};
pub use crate::text_embedding::{
    InitOptions, InitOptionsUserDefined, TextEmbedding, UserDefinedEmbeddingModel,
};
