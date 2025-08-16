use super::utils::Compose;
use crate::{init::InitOptions, ImageEmbeddingModel};
use ort::{execution_providers::ExecutionProviderDispatch, session::Session};

/// Options for initializing the ImageEmbedding model
pub type ImageInitOptions = InitOptions<ImageEmbeddingModel>;

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
