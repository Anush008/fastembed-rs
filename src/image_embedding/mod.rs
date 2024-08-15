use crate::models::image_embedding::ImageEmbeddingModel;
const DEFAULT_BATCH_SIZE: usize = 256;
const DEFAULT_EMBEDDING_MODEL: ImageEmbeddingModel = ImageEmbeddingModel::ClipVitB32;

mod utils;

mod init;
pub use init::*;

mod r#impl;
