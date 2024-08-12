use crate::{
    common::{normalize, Compose, Transform, TransformData, DEFAULT_CACHE_DIR},
    models::image_embedding::{models_list, ImageEmbeddingModel},
    Embedding, ModelInfo,
};
use anyhow::{anyhow, Result};
use hf_hub::{
    api::sync::{ApiBuilder, ApiRepo},
    Cache,
};
use ndarray::{Array3, ArrayView3};
use ort::{ExecutionProviderDispatch, GraphOptimizationLevel, Session, Value};
use rayon::{iter::ParallelIterator, slice::ParallelSlice};
use std::{
    fmt::Display,
    path::{Path, PathBuf},
    thread::available_parallelism,
};
const DEFAULT_BATCH_SIZE: usize = 256;
const DEFAULT_EMBEDDING_MODEL: ImageEmbeddingModel = ImageEmbeddingModel::ClipVitB32;

/// Options for initializing the ImageEmbedding model
#[derive(Debug, Clone)]
pub struct ImageInitOptions {
    pub model_name: ImageEmbeddingModel,
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub cache_dir: PathBuf,
    pub show_download_progress: bool,
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
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct ImageInitOptionsUserDefined {
    pub execution_providers: Vec<ExecutionProviderDispatch>,
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
pub struct UserDefinedImageEmbeddingModel {
    pub onnx_file: Vec<u8>,
    pub preprocessor_file: Vec<u8>,
}

/// Rust representation of the ImageEmbedding model
pub struct ImageEmbedding {
    preprocessor: Compose,
    session: Session,
}

impl Display for ImageEmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let model_info = ImageEmbedding::list_supported_models()
            .into_iter()
            .find(|model| model.model == *self)
            .unwrap();
        write!(f, "{}", model_info.model_code)
    }
}

impl ImageEmbedding {
    /// Try to generate a new ImageEmbedding Instance
    ///
    /// Uses the highest level of Graph optimization
    ///
    /// Uses the total number of CPUs available as the number of intra-threads
    pub fn try_new(options: ImageInitOptions) -> Result<Self> {
        let ImageInitOptions {
            model_name,
            execution_providers,
            cache_dir,
            show_download_progress,
        } = options;

        let threads = available_parallelism()?.get();

        let model_repo = ImageEmbedding::retrieve_model(
            model_name.clone(),
            cache_dir.clone(),
            show_download_progress,
        )?;

        let preprocessor_file = model_repo
            .get("preprocessor_config.json")
            .unwrap_or_else(|_| panic!("Failed to retrieve preprocessor_config.json"));
        let preprocessor = Compose::from_file(preprocessor_file)?;

        let model_file_name = ImageEmbedding::get_model_info(&model_name).model_file;
        let model_file_reference = model_repo
            .get(&model_file_name)
            .unwrap_or_else(|_| panic!("Failed to retrieve {} ", model_file_name));

        let session = Session::builder()?
            .with_execution_providers(execution_providers)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(model_file_reference)?;

        Ok(Self::new(preprocessor, session))
    }

    /// Create a ImageEmbedding instance from model files provided by the user.
    ///
    /// This can be used for 'bring your own' embedding models
    pub fn try_new_from_user_defined(
        model: UserDefinedImageEmbeddingModel,
        options: ImageInitOptionsUserDefined,
    ) -> Result<Self> {
        let ImageInitOptionsUserDefined {
            execution_providers,
        } = options;

        let threads = available_parallelism()?.get();

        let preprocessor = Compose::from_bytes(model.preprocessor_file)?;

        let session = Session::builder()?
            .with_execution_providers(execution_providers)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_memory(&model.onnx_file)?;

        Ok(Self::new(preprocessor, session))
    }

    /// Private method to return an instance
    fn new(preprocessor: Compose, session: Session) -> Self {
        Self {
            preprocessor,
            session,
        }
    }

    /// Return the ImageEmbedding model's directory from cache or remote retrieval
    fn retrieve_model(
        model: ImageEmbeddingModel,
        cache_dir: PathBuf,
        show_download_progress: bool,
    ) -> Result<ApiRepo> {
        let cache = Cache::new(cache_dir);
        let api = ApiBuilder::from_cache(cache)
            .with_progress(show_download_progress)
            .build()
            .unwrap();

        let repo = api.model(model.to_string());
        Ok(repo)
    }

    /// Retrieve a list of supported models
    pub fn list_supported_models() -> Vec<ModelInfo<ImageEmbeddingModel>> {
        models_list()
    }

    /// Get ModelInfo from ImageEmbeddingModel
    pub fn get_model_info(model: &ImageEmbeddingModel) -> ModelInfo<ImageEmbeddingModel> {
        ImageEmbedding::list_supported_models()
            .into_iter()
            .find(|m| &m.model == model)
            .expect("Model not found.")
    }

    /// Method to generate image embeddings for a Vec of image path
    // Generic type to accept String, &str, OsString, &OsStr
    pub fn embed<S: AsRef<Path> + Send + Sync>(
        &self,
        images: Vec<S>,
        batch_size: Option<usize>,
    ) -> Result<Vec<Embedding>> {
        // Determine the batch size, default if not specified
        let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE);

        let output = images
            .par_chunks(batch_size)
            .map(|batch| {
                // Encode the texts in the batch
                let inputs = batch
                    .iter()
                    .map(|img| {
                        let img = image::ImageReader::open(img)?
                            .decode()
                            .map_err(|err| anyhow!("image decode: {}", err))?;
                        let pixels = self.preprocessor.transform(TransformData::Image(img))?;
                        match pixels {
                            TransformData::NdArray(array) => Ok(array),
                            _ => Err(anyhow!("Preprocessor configuration error!")),
                        }
                    })
                    .collect::<Result<Vec<Array3<f32>>>>()?;

                // Extract the batch size
                let inputs_view: Vec<ArrayView3<f32>> =
                    inputs.iter().map(|img| img.view()).collect();
                let pixel_values_array = ndarray::stack(ndarray::Axis(0), &inputs_view)?;

                let input_name = self.session.inputs[0].name.clone();
                let session_inputs = ort::inputs![
                    input_name => Value::from_array(pixel_values_array)?,
                ]?;

                let outputs = self.session.run(session_inputs)?;

                // Try to get the only output key
                // If multiple, then default to `image_embeds`
                let last_hidden_state_key = match outputs.len() {
                    1 => outputs.keys().next().unwrap(),
                    _ => "image_embeds",
                };

                // Extract and normalize embeddings
                let output_data = outputs[last_hidden_state_key].try_extract_tensor::<f32>()?;

                let embeddings: Vec<Vec<f32>> = output_data
                    .rows()
                    .into_iter()
                    .map(|row| normalize(row.as_slice().unwrap()))
                    .collect();

                Ok(embeddings)
            })
            .flat_map(|result: Result<Vec<Vec<f32>>, anyhow::Error>| result.unwrap())
            .collect();

        Ok(output)
    }
}
