#[cfg(feature = "online")]
use crate::{
    common::{normalize, DEFAULT_CACHE_DIR},
    models::text_embedding::models_list,
    pooling::{self, Pooling},
    Embedding, EmbeddingModel, ModelInfo,
};
use anyhow::{anyhow, Result};
#[cfg(feature = "online")]
use hf_hub::{
    api::sync::{ApiBuilder, ApiRepo},
    Cache,
};
use ndarray::Array;
use ort::{ExecutionProviderDispatch, GraphOptimizationLevel, Session, Value};
use rayon::{iter::ParallelIterator, slice::ParallelSlice};
use std::{
    fmt::Display,
    path::{Path, PathBuf},
    thread::available_parallelism,
};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};
const DEFAULT_BATCH_SIZE: usize = 256;
const DEFAULT_EMBEDDING_MODEL: EmbeddingModel = EmbeddingModel::BGESmallENV15;

/// Options for initializing the TextEmbedding model
#[derive(Debug, Clone)]
pub struct InitOptions {
    pub model_name: EmbeddingModel,
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub cache_dir: PathBuf,
    pub show_download_progress: bool,
}

impl Default for InitOptions {
    fn default() -> Self {
        Self {
            model_name: DEFAULT_EMBEDDING_MODEL,
            execution_providers: Default::default(),
            cache_dir: Path::new(DEFAULT_CACHE_DIR).to_path_buf(),
            show_download_progress: true,
        }
    }
}

/// Options for initializing UserDefinedEmbeddingModel
///
/// Model files are held by the UserDefinedEmbeddingModel struct
#[derive(Debug, Clone, Default)]
pub struct InitOptionsUserDefined {
    pub execution_providers: Vec<ExecutionProviderDispatch>,
}

/// Convert InitOptions to InitOptionsUserDefined
///
/// This is useful for when the user wants to use the same options for both the default and user-defined models
impl From<InitOptions> for InitOptionsUserDefined {
    fn from(options: InitOptions) -> Self {
        InitOptionsUserDefined {
            execution_providers: options.execution_providers,
        }
    }
}

/// Struct for "bring your own" embedding models
///
/// The onnx_file and tokenizer_files are expecting the files' bytes
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserDefinedEmbeddingModel {
    pub onnx_file: Vec<u8>,
    pub tokenizer_file: Vec<u8>,
    pub pooling: Option<Pooling>,
}

/// Rust representation of the TextEmbedding model
pub struct TextEmbedding {
    pub tokenizer: Tokenizer,
    pub pooling: Option<Pooling>,
    session: Session,
    need_token_type_ids: bool,
}

impl Display for EmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let model_info = TextEmbedding::list_supported_models()
            .into_iter()
            .find(|model| model.model == *self)
            .unwrap();
        write!(f, "{}", model_info.model_code)
    }
}

impl TextEmbedding {
    /// Try to generate a new TextEmbedding Instance
    ///
    /// Uses the highest level of Graph optimization
    ///
    /// Uses the total number of CPUs available as the number of intra-threads
    #[cfg(feature = "online")]
    pub fn try_new(options: InitOptions) -> Result<Self> {
        let InitOptions {
            model_name,
            execution_providers,
            cache_dir,
            show_download_progress,
        } = options;

        let threads = available_parallelism()?.get();

        let model_repo = TextEmbedding::retrieve_model(
            model_name.clone(),
            cache_dir.clone(),
            show_download_progress,
        )?;

        let tokenizer_file_reference = model_repo.get("tokenizer.json")?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_file_reference)
            .map_err(|err| anyhow!("Failed to load tokenizer: {}", err))?;
        if tokenizer.get_padding().is_none() {
            tokenizer.with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                ..Default::default()
            }));
        }

        let model_file_name = TextEmbedding::get_model_info(&model_name).model_file;
        let model_file_reference = model_repo
            .get(&model_file_name)
            .unwrap_or_else(|_| panic!("Failed to retrieve {} {} ", model_name, model_file_name));

        // TODO: If more models need .onnx_data, implement a better way to handle this
        // Probably by adding `additional_files` field in the `ModelInfo` struct
        if model_name == EmbeddingModel::MultilingualE5Large {
            model_repo
                .get("model.onnx_data")
                .expect("Failed to retrieve model.onnx_data.");
        }

        // prioritise loading pooling config if available, if not (thanks qdrant!), look for it in hardcoded
        let post_processing = model_name.get_default_pooling_method();

        let session = Session::builder()?
            .with_execution_providers(execution_providers)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(model_file_reference)?;

        Ok(Self::new(tokenizer, session, post_processing))
    }

    /// Create a TextEmbedding instance from model files provided by the user.
    ///
    /// This can be used for 'bring your own' embedding models
    pub fn try_new_from_user_defined(
        model: UserDefinedEmbeddingModel,
        options: InitOptionsUserDefined,
    ) -> Result<Self> {
        let InitOptionsUserDefined {
            execution_providers,
        } = options;

        let threads = available_parallelism()?.get();

        let session = Session::builder()?
            .with_execution_providers(execution_providers)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_memory(&model.onnx_file)?;

        let tokenizer = Tokenizer::from_bytes(model.tokenizer_file)
            .map_err(|err| anyhow!("Failed to load tokenizer: {}", err))?;
        Ok(Self::new(tokenizer, session, model.pooling))
    }

    /// Private method to return an instance
    fn new(tokenizer: Tokenizer, session: Session, post_process: Option<Pooling>) -> Self {
        let need_token_type_ids = session
            .inputs
            .iter()
            .any(|input| input.name == "token_type_ids");
        Self {
            tokenizer,
            session,
            need_token_type_ids,
            pooling: post_process,
        }
    }
    /// Return the TextEmbedding model's directory from cache or remote retrieval
    #[cfg(feature = "online")]
    fn retrieve_model(
        model: EmbeddingModel,
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
    pub fn list_supported_models() -> Vec<ModelInfo<EmbeddingModel>> {
        models_list()
    }

    /// Get ModelInfo from EmbeddingModel
    pub fn get_model_info(model: &EmbeddingModel) -> ModelInfo<EmbeddingModel> {
        TextEmbedding::list_supported_models()
            .into_iter()
            .find(|m| &m.model == model)
            .expect("Model not found.")
    }

    /// Method to generate sentence embeddings for a Vec of texts
    // Generic type to accept String, &str, OsString, &OsStr
    pub fn embed<S: AsRef<str> + Send + Sync>(
        &self,
        texts: Vec<S>,
        batch_size: Option<usize>,
    ) -> Result<Vec<Embedding>> {
        // Determine the batch size, default if not specified
        let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE);

        let output = texts
            .par_chunks(batch_size)
            .map(|batch| {
                // Encode the texts in the batch
                let inputs = batch.iter().map(|text| text.as_ref()).collect();
                let encodings = self.tokenizer.encode_batch(inputs, true).unwrap();

                // Extract the encoding length and batch size
                let encoding_length = encodings[0].len();
                let batch_size = batch.len();

                let max_size = encoding_length * batch_size;

                // Preallocate arrays with the maximum size
                let mut ids_array = Vec::with_capacity(max_size);
                let mut mask_array = Vec::with_capacity(max_size);
                let mut typeids_array = Vec::with_capacity(max_size);

                // Not using par_iter because the closure needs to be FnMut
                encodings.iter().for_each(|encoding| {
                    let ids = encoding.get_ids();
                    let mask = encoding.get_attention_mask();
                    let typeids = encoding.get_type_ids();

                    // Extend the preallocated arrays with the current encoding
                    // Requires the closure to be FnMut
                    ids_array.extend(ids.iter().map(|x| *x as i64));
                    mask_array.extend(mask.iter().map(|x| *x as i64));
                    typeids_array.extend(typeids.iter().map(|x| *x as i64));
                });

                // Create CowArrays from vectors
                let inputs_ids_array =
                    Array::from_shape_vec((batch_size, encoding_length), ids_array)?;

                let attention_mask_array =
                    Array::from_shape_vec((batch_size, encoding_length), mask_array)?;

                let token_type_ids_array =
                    Array::from_shape_vec((batch_size, encoding_length), typeids_array)?;

                let mut session_inputs = ort::inputs![
                    "input_ids" => Value::from_array(inputs_ids_array)?,
                    "attention_mask" => Value::from_array(attention_mask_array.clone())?,
                ]?;

                if self.need_token_type_ids {
                    session_inputs.push((
                        "token_type_ids".into(),
                        Value::from_array(token_type_ids_array)?.into(),
                    ));
                }

                let outputs = self.session.run(session_inputs)?;

                // Try to get the only output key
                // If multiple, then default to `last_hidden_state`
                let last_hidden_state_key = match outputs.len() {
                    1 => outputs.keys().next().unwrap(),
                    _ => "last_hidden_state",
                };

                // Extract as tensor
                let output_data = outputs[last_hidden_state_key].try_extract_tensor::<f32>()?;

                // Pre compute attention mask for post processing
                let attention_mask = attention_mask_array.insert_axis(ndarray::Axis(2));
                let attention_mask = attention_mask
                    .broadcast(output_data.dim())
                    .expect("Resize attention mask to match output successfull")
                    .mapv(|x| x as f32);

                let embeddings: Vec<Vec<f32>> = match self.pooling {
                    // default to cls so as not to break the existing implementations
                    // TODO: Consider return output as is to support custom model that has built-in pooling layer.
                    None => pooling::cls(&output_data)
                        .rows()
                        .into_iter()
                        .map(|row| normalize(row.as_slice().expect("success")))
                        .collect(),
                    Some(Pooling::Cls) => pooling::cls(&output_data)
                        .rows()
                        .into_iter()
                        .map(|row| normalize(row.as_slice().expect("success")))
                        .collect(),
                    Some(Pooling::Mean) => pooling::mean(&output_data, &attention_mask)
                        .rows()
                        .into_iter()
                        .map(|row| normalize(row.as_slice().expect("success")))
                        .collect(),
                };
                Ok(embeddings)
            })
            .flat_map(|result: Result<Vec<Vec<f32>>, anyhow::Error>| result.unwrap())
            .collect();

        Ok(output)
    }
}
