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

use std::{
    fmt::Display,
    fs::{read_dir, File},
    path::{self, Path, PathBuf},
    thread::available_parallelism,
};

use anyhow::{Error, Ok, Result};
use hf_hub::api::{sync::ApiRepo, RepoInfo};
use hf_hub::{api::sync::ApiBuilder, Cache};
use ndarray::Array;
pub use ort::{ExecutionProvider, ExecutionProviderDispatch};
use ort::{GraphOptimizationLevel, Session, Value};
use rayon::{
    prelude::{IntoParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};
use tokenizers::{AddedToken, PaddingParams, PaddingStrategy, TruncationParams};
use variant_count::VariantCount;

const DEFAULT_BATCH_SIZE: usize = 256;
const DEFAULT_MAX_LENGTH: usize = 512;
const DEFAULT_CACHE_DIR: &str = ".fastembed_cache";
const DEFAULT_EMBEDDING_MODEL: EmbeddingModel = EmbeddingModel::BGESmallENV15;

/// Type alias for the embedding vector
pub type Embedding = Vec<f32>;

/// Enum for the available models
#[derive(Debug, Clone, PartialEq, Eq, VariantCount)]
pub enum EmbeddingModel {
    /// Sentence Transformer model, MiniLM-L6-v2
    AllMiniLML6V2,
    /// v1.5 release of the base English model
    BGEBaseENV15,
    /// v1.5 release of the large English model
    BGELargeENV15,
    /// Fast and Default English model
    BGESmallENV15,
    /// 8192 context length english model
    NomicEmbedTextV1,
    /// Multi-lingual model
    ParaphraseMLMiniLML12V2,
    /// v1.5 release of the small Chinese model
    BGESmallZHV15,
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

/// Options for initializing the TextEmbedding model
#[derive(Debug, Clone)]
pub struct InitOptions {
    pub model_name: EmbeddingModel,
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub max_length: usize,
    pub cache_dir: PathBuf,
    pub show_download_progress: bool,
}

impl Default for InitOptions {
    fn default() -> Self {
        Self {
            model_name: DEFAULT_EMBEDDING_MODEL,
            execution_providers: Default::default(),
            max_length: DEFAULT_MAX_LENGTH,
            cache_dir: Path::new(DEFAULT_CACHE_DIR).to_path_buf(),
            show_download_progress: true,
        }
    }
}
/// Options for initializing UserDefinedEmbeddingModel
/// Model files are held by the UserDefinedEmbeddingModel struct
#[derive(Debug, Clone)]
pub struct InitOptionsUserDefined {
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub max_length: usize,
}

impl Default for InitOptionsUserDefined {
    fn default() -> Self {
        Self {
            execution_providers: Default::default(),
            max_length: DEFAULT_MAX_LENGTH,
        }
    }
}

/// Data struct about the available models
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model: EmbeddingModel,
    pub dim: usize,
    pub description: String,
    pub model_code: String,
}

// Struct for "bring your own" embedding models
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserDefinedEmbeddingModel {
    pub dim: usize,
    pub description: String,
    pub model_code: String,
    pub onnx_file: LocalOrRemoteFile,
    pub tokenizer_files: TokenizerFiles,
}

// Tokenizer files for "bring your own" embedding models
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerFiles {
    pub tokenizer_file: LocalOrRemoteFile,
    pub config_file: LocalOrRemoteFile,
    pub special_tokens_map_file: LocalOrRemoteFile,
    pub tokenizer_config_file: LocalOrRemoteFile,
}

// This enum allows users to specify local or remote file paths (eg, cloud storage)
// for the onnx file and tokenizer files for a UserDefinedEmbeddingModel
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalOrRemoteFile {
    Local(PathBuf),
    Remote(String),
}

/// Rust representation of the TextEmbedding model
pub struct TextEmbedding {
    tokenizer: Tokenizer,
    session: Session,
}

impl TextEmbedding {
    /// Try to generate a new TextEmbedding Instance
    ///
    /// Uses the highest level of Graph optimization
    ///
    /// Uses the total number of CPUs available as the number of intra-threads
    pub fn try_new(options: InitOptions) -> Result<Self> {
        let InitOptions {
            model_name,
            execution_providers,
            max_length,
            cache_dir,
            show_download_progress,
        } = options;

        let threads = available_parallelism()?.get() as i16;

        let model_repo = TextEmbedding::retrieve_model(
            model_name.clone(),
            cache_dir.clone(),
            show_download_progress,
        )?;

        let model_file_info_result = model_repo.info();

        // If the attempt fails (likely no connection), fall back on cached onnx file
        let model_file_reference = match model_file_info_result {
            std::result::Result::Ok(info) => {
                TextEmbedding::retrieve_remote_model_file(info, &model_repo)
            }
            Err(ref _e) => {
                eprintln!("Falling back on cached model.");
                TextEmbedding::retrieve_cached_model_file(&model_name, &cache_dir).expect(
                    "Could not find any locally cached .onnx file for this model. Please try again with a web connection.",
                )
            }
        };

        let session = Session::builder()?
            .with_execution_providers(execution_providers)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .with_model_from_file(model_file_reference)?;

        let tokenizer = TextEmbedding::load_tokenizer_hf_hub(model_repo, max_length)?;
        Ok(Self::new(tokenizer, session))
    }

    /// Create a TextEmbedding instance from model files provided by the user.
    /// This can be used for 'bring your own' embedding models
    /// It also facilitates remote hosting of the model files (eg, on cloud storage)
    pub fn try_new_from_user_defined(
        model: UserDefinedEmbeddingModel,
        options: InitOptionsUserDefined,
    ) -> Result<Self> {
        let InitOptionsUserDefined {
            execution_providers,
            max_length,
        } = options;

        let threads = available_parallelism()?.get() as i16;
        let session: Session;
        match model.onnx_file {
            LocalOrRemoteFile::Local(path) => {
                session = Session::builder()?
                    .with_execution_providers(execution_providers)?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .with_intra_threads(threads)?
                    .with_model_from_file(path)?;
            }
            LocalOrRemoteFile::Remote(url) => {
                // Load the model from the remote file to memory
                let model = reqwest::blocking::get(url)?.bytes()?;
                session = Session::builder()?
                    .with_execution_providers(execution_providers)?
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .with_intra_threads(threads)?
                    .with_model_from_memory(&model)?;
            }
        }

        let tokenizer = TextEmbedding::load_tokenizer(model.tokenizer_files, max_length)?;
        Ok(Self::new(tokenizer, session))
    }

    /// Private method to return an instance
    fn new(tokenizer: Tokenizer, session: Session) -> Self {
        Self { tokenizer, session }
    }
    /// Return the TextEmbedding model's directory from cache or remote retrieval
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

    /// Look for the model in the hf remote repository
    /// This will download the .onnx if not already cached
    fn retrieve_remote_model_file(model_file_info: RepoInfo, model_repo: &ApiRepo) -> PathBuf {
        let model_file = model_file_info
            .siblings
            .into_iter()
            .find(|f| {
                f.rfilename.ends_with("model.onnx") || f.rfilename.ends_with("model_optimized.onnx")
            })
            .expect("Can't retrieve .onnx model from remote. Try again with a connection.");
        model_repo
            .get(&model_file.rfilename)
            .expect(".onnx file is not available in cache. This shouldn't happen - try again.")
    }

    /// Look for the model file path in the local cache only - no call to hf remote
    fn retrieve_cached_model_file(
        embedding_model: &EmbeddingModel,
        cache_dir: &PathBuf,
    ) -> Result<PathBuf> {
        let model_info = TextEmbedding::get_model_info(embedding_model);
        get_cached_onnx_file(model_info?, cache_dir)
    }

    fn load_tokenizer_hf_hub(model_repo: ApiRepo, max_length: usize) -> Result<Tokenizer> {
        let tokenizer_files: TokenizerFiles = TokenizerFiles {
            tokenizer_file: LocalOrRemoteFile::Local(model_repo.get("tokenizer.json")?),
            config_file: LocalOrRemoteFile::Local(model_repo.get("config.json")?),
            special_tokens_map_file: LocalOrRemoteFile::Local(
                model_repo.get("special_tokens_map.json")?,
            ),
            tokenizer_config_file: LocalOrRemoteFile::Local(
                model_repo.get("tokenizer_config.json")?,
            ),
        };
        TextEmbedding::load_tokenizer(tokenizer_files, max_length)
    }

    fn load_tokenizer(tokenizer_files: TokenizerFiles, max_length: usize) -> Result<Tokenizer> {
        let config: serde_json::Value;
        match tokenizer_files.tokenizer_file {
            LocalOrRemoteFile::Local(ref path) => {
                let config_path: PathBuf = path.to_path_buf();
                let file = File::open(config_path)?;
                config = serde_json::from_reader(file)?;
            }
            LocalOrRemoteFile::Remote(ref url) => {
                let response = reqwest::blocking::get(url)?;
                if response.status().is_success() {
                    config = serde_json::from_str(&response.text()?)?;
                } else {
                    return Err(Error::msg(format!(
                        "Failed to fetch URL: {}",
                        response.status()
                    )));
                }
            }
        }

        let special_tokens_map: serde_json::Value;
        match tokenizer_files.special_tokens_map_file {
            LocalOrRemoteFile::Local(ref path) => {
                let config_path: PathBuf = path.to_path_buf();
                let file = File::open(config_path)?;
                special_tokens_map = serde_json::from_reader(file)?;
            }
            LocalOrRemoteFile::Remote(ref url) => {
                let response = reqwest::blocking::get(url)?;
                if response.status().is_success() {
                    special_tokens_map = serde_json::from_str(&response.text()?)?;
                } else {
                    return Err(Error::msg(format!(
                        "Failed to fetch URL: {}",
                        response.status()
                    )));
                }
            }
        }

        let tokenizer_config: serde_json::Value;
        match tokenizer_files.tokenizer_config_file {
            LocalOrRemoteFile::Local(ref path) => {
                let config_path: PathBuf = path.to_path_buf();
                let file = File::open(config_path)?;
                tokenizer_config = serde_json::from_reader(file)?;
            }
            LocalOrRemoteFile::Remote(ref url) => {
                let response = reqwest::blocking::get(url)?;
                if response.status().is_success() {
                    tokenizer_config = serde_json::from_str(&response.text()?)?;
                } else {
                    return Err(Error::msg(format!(
                        "Failed to fetch URL: {}",
                        response.status()
                    )));
                }
            }
        }

        let mut tokenizer: tokenizers::Tokenizer;

        match tokenizer_files.tokenizer_file {
            LocalOrRemoteFile::Local(path) => {
                tokenizer = tokenizers::Tokenizer::from_file(path).map_err(anyhow::Error::msg)?;
            }
            LocalOrRemoteFile::Remote(url) => {
                let response = reqwest::blocking::get(url)?;
                if response.status().is_success() {
                    let tokenizer_bytes = response.bytes()?;
                    tokenizer = tokenizers::Tokenizer::from_bytes(tokenizer_bytes)
                        .map_err(anyhow::Error::msg)?;
                } else {
                    return Err(Error::msg(format!(
                        "Failed to fetch URL: {}",
                        response.status()
                    )));
                }
            }
        }

        //For BGEBaseSmall, the model_max_length value is set to 1000000000000000019884624838656. Which fits in a f64
        let model_max_length = tokenizer_config["model_max_length"].as_f64().unwrap();
        let max_length = max_length.min(model_max_length as usize);
        let pad_id = config["pad_token_id"].as_u64().unwrap_or(0) as u32;
        let pad_token = tokenizer_config["pad_token"].as_str().unwrap().into();

        let mut tokenizer = tokenizer
            .with_padding(Some(PaddingParams {
                // TODO: the user should able to choose the padding strategy
                strategy: PaddingStrategy::BatchLongest,
                pad_token,
                pad_id,
                ..Default::default()
            }))
            .with_truncation(Some(TruncationParams {
                max_length,
                ..Default::default()
            }))
            .map_err(anyhow::Error::msg)?
            .clone();
        if let serde_json::Value::Object(root_object) = special_tokens_map {
            for (_, value) in root_object.iter() {
                if value.is_string() {
                    tokenizer.add_special_tokens(&[AddedToken {
                        content: value.as_str().unwrap().into(),
                        special: true,
                        ..Default::default()
                    }]);
                } else if value.is_object() {
                    tokenizer.add_special_tokens(&[AddedToken {
                        content: value["content"].as_str().unwrap().into(),
                        special: true,
                        single_word: value["single_word"].as_bool().unwrap(),
                        lstrip: value["lstrip"].as_bool().unwrap(),
                        rstrip: value["rstrip"].as_bool().unwrap(),
                        normalized: value["normalized"].as_bool().unwrap(),
                    }]);
                }
            }
        }
        Ok(tokenizer)
    }

    /// Retrieve a list of supported models
    pub fn list_supported_models() -> Vec<ModelInfo> {
        let models = vec![
            ModelInfo {
                model: EmbeddingModel::AllMiniLML6V2,
                dim: 384,
                description: String::from("Sentence Transformer model, MiniLM-L6-v2"),
                model_code: String::from("Qdrant/all-MiniLM-L6-v2-onnx"),
            },
            ModelInfo {
                model: EmbeddingModel::BGEBaseENV15,
                dim: 768,
                description: String::from("v1.5 release of the base English model"),
                model_code: String::from("Qdrant/bge-base-en-v1.5-onnx-Q"),
            },
            ModelInfo {
                model: EmbeddingModel::BGELargeENV15,
                dim: 1024,
                description: String::from("v1.5 release of the large English model"),
                model_code: String::from("Qdrant/bge-large-en-v1.5-onnx-Q"),
            },
            ModelInfo {
                model: EmbeddingModel::BGESmallENV15,
                dim: 384,
                description: String::from("v1.5 release of the fast and default English model"),
                model_code: String::from("Qdrant/bge-small-en-v1.5-onnx-Q"),
            },
            ModelInfo {
                model: EmbeddingModel::NomicEmbedTextV1,
                dim: 768,
                description: String::from("8192 context length english model"),
                model_code: String::from("nomic-ai/nomic-embed-text-v1"),
            },
            ModelInfo {
                model: EmbeddingModel::ParaphraseMLMiniLML12V2,
                dim: 384,
                description: String::from("Multi-lingual model"),
                model_code: String::from("Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"),
            },
            ModelInfo {
                model: EmbeddingModel::BGESmallZHV15,
                dim: 512,
                description: String::from("v1.5 release of the small Chinese model"),
                model_code: String::from("Xenova/bge-small-zh-v1.5"),
            },
        ];

        // TODO: Use when out in stable
        // assert_eq!(std::mem::variant_count::<EmbeddingModel>(), models.len());

        assert_eq!(
            EmbeddingModel::VARIANT_COUNT,
            models.len(),
            "list_supported_models() is not exhaustive"
        );
        models
    }

    /// Get ModelInfo from EmbeddingModel
    pub fn get_model_info(model: &EmbeddingModel) -> Result<ModelInfo> {
        let model_info_option = TextEmbedding::list_supported_models()
            .into_iter()
            .find(|m| &m.model == model);
        match model_info_option {
            Some(model_info) => Ok(model_info),
            None => Err(Error::msg("Model not found")),
        }
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

                let outputs = self.session.run(ort::inputs![
                    "input_ids" => Value::from_array(inputs_ids_array)?,
                    "attention_mask" => Value::from_array(attention_mask_array)?,
                    "token_type_ids" => Value::from_array(token_type_ids_array)?,
                ]?)?;

                // Extract and normalize embeddings
                let output_data = outputs["last_hidden_state"].extract_tensor::<f32>()?;
                let view = output_data.view();
                let shape = view.shape();
                let flattened = view.as_slice().unwrap();
                let data = get_embeddings(flattened, shape);
                let embeddings: Vec<Embedding> = data
                    .into_par_iter()
                    .map(|mut d| normalize(&mut d))
                    .collect();

                Ok(embeddings)
            })
            .flat_map(|result| result.unwrap())
            .collect();

        Ok(output)
    }
}

type Tokenizer = tokenizers::TokenizerImpl<
    tokenizers::ModelWrapper,
    tokenizers::NormalizerWrapper,
    tokenizers::PreTokenizerWrapper,
    tokenizers::PostProcessorWrapper,
    tokenizers::DecoderWrapper,
>;

fn normalize(v: &mut [f32]) -> Vec<f32> {
    let norm = (v.iter().map(|val| val * val).sum::<f32>()).sqrt();
    let epsilon = 1e-12;

    // We add the super-small epsilon to avoid dividing by zero
    v.iter().map(|&val| val / (norm + epsilon)).collect()
}

fn get_embeddings(data: &[f32], dimensions: &[usize]) -> Vec<Embedding> {
    let x = dimensions[0];
    let y = dimensions[1];
    let z = dimensions[2];
    let mut embeddings: Vec<Embedding> = Vec::with_capacity(x);

    for index in 0..x {
        let start_index = index * y * z;
        let end_index = start_index + z;
        let embedding = data[start_index..end_index].to_vec();
        embeddings.push(embedding);
    }

    embeddings
}

/// Get the cached onnx file from the model directory
fn get_cached_onnx_file(model: ModelInfo, cache_dir: &PathBuf) -> Result<PathBuf> {
    // Get relevant model directory
    let conformed_model_name = format!("models--{}", model.model_code.replace('/', "--"));
    let model_dir = Path::new(cache_dir).join(conformed_model_name);
    // Walk the directory and find the onnx file
    let onnx_file = visit_dirs(&model_dir);
    onnx_file
}

fn visit_dirs(dir: &Path) -> Result<PathBuf> {
    if dir.is_dir() {
        for entry in read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                if let std::result::Result::Ok(path_buf) = visit_dirs(&path) {
                    return Ok(path_buf);
                }
            }
            if path.ends_with("model_optimized.onnx") || path.ends_with("model.onnx") {
                return Ok(path.to_path_buf());
            }
        }
    }
    Err(Error::msg("Can't locate .onnx file in local cache."))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embeddings() {
        for supported_model in TextEmbedding::list_supported_models() {
            let model: TextEmbedding = TextEmbedding::try_new(InitOptions {
                model_name: supported_model.model,
                ..Default::default()
            })
            .unwrap();

            let documents = vec![
                "Hello, World!",
                "This is an example passage.",
                "fastembed-rs is licensed under Apache-2.0",
                "Some other short text here blah blah blah",
            ];

            // Generate embeddings with the default batch size, 256
            let embeddings = model.embed(documents.clone(), None).unwrap();

            assert_eq!(embeddings.len(), documents.len());
            for embedding in embeddings {
                assert_eq!(embedding.len(), supported_model.dim);
            }
        }
    }
}
