use std::path::{Path, PathBuf};

use ort::{execution_providers::ExecutionProviderDispatch, session::Session};

use crate::{ImageEmbeddingModel, DEFAULT_CACHE_DIR};

use super::{utils::Compose, DEFAULT_EMBEDDING_MODEL};

/// Options for initializing the ImageEmbedding model
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ImageInitOptions {
    pub model_name: ImageEmbeddingModel,
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub cache_dir: PathBuf,
    pub show_download_progress: bool,
}

impl ImageInitOptions {
    pub fn new(model_name: ImageEmbeddingModel) -> Self {
        Self {
            model_name,
            ..Default::default()
        }
    }

    pub fn with_cache_dir(mut self, cache_dir: PathBuf) -> Self {
        self.cache_dir = cache_dir;
        self
    }

    pub fn with_execution_providers(
        mut self,
        execution_providers: Vec<ExecutionProviderDispatch>,
    ) -> Self {
        self.execution_providers = execution_providers;
        self
    }

    pub fn with_show_download_progress(mut self, show_download_progress: bool) -> Self {
        self.show_download_progress = show_download_progress;
        self
    }
}

impl Default for ImageInitOptions {
    fn default() -> Self {
        Self {
            model_name: DEFAULT_EMBEDDING_MODEL,
            execution_providers: Default::default(),
            cache_dir: Path::new(DEFAULT_CACHE_DIR).to_path_buf(),
            show_download_progress: true,
        }
    }
}

/// Options for initializing UserDefinedImageEmbeddingModel
///
/// Model files are held by the UserDefinedImageEmbeddingModel struct
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ImageInitOptionsUserDefined {
    pub execution_providers: Vec<ExecutionProviderDispatch>,
}

impl ImageInitOptionsUserDefined {
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
}

/// Convert ImageInitOptions to ImageInitOptionsUserDefined
///
/// This is useful for when the user wants to use the same options for both the default and user-defined models
impl From<ImageInitOptions> for ImageInitOptionsUserDefined {
    fn from(options: ImageInitOptions) -> Self {
        ImageInitOptionsUserDefined {
            execution_providers: options.execution_providers,
        }
    }
}

/// Struct for "bring your own" embedding models
///
/// The onnx_file and preprocessor_files are expecting the files' bytes
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct UserDefinedImageEmbeddingModel {
    pub onnx_file: Vec<u8>,
    pub preprocessor_file: Vec<u8>,
}

impl UserDefinedImageEmbeddingModel {
    pub fn new(onnx_file: Vec<u8>, preprocessor_file: Vec<u8>) -> Self {
        Self {
            onnx_file,
            preprocessor_file,
        }
    }
}

/// Rust representation of the ImageEmbedding model
pub struct ImageEmbedding {
    pub(crate) preprocessor: Compose,
    pub(crate) session: Session,
}
