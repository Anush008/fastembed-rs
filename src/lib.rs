//! [FastEmbed](https://github.com/Anush008/fastembed-rs) - Fast, light, accurate library built for retrieval embedding generation.
//!
//! The library provides the FlagEmbedding struct to interface with the Flag embedding models.
//!
//! ### Instantiating [FlagEmbedding](crate::FlagEmbedding)
//! ```
//! use fastembed::{FlagEmbedding, InitOptions, EmbeddingModel, EmbeddingBase};
//!
//!# fn model_demo() -> anyhow::Result<()> {
//! // With default InitOptions
//! let model: FlagEmbedding = FlagEmbedding::try_new(Default::default())?;
//!
//! // List all supported models
//! dbg!(FlagEmbedding::list_supported_models());
//!
//! // With custom InitOptions
//! let model: FlagEmbedding = FlagEmbedding::try_new(InitOptions {
//!     model_name: EmbeddingModel::BGEBaseENV15,
//!     show_download_message: false,
//!     ..Default::default()
//! })?;
//! # Ok(())
//! # }
//! ```
//! Find more info about the available options in the [InitOptions](crate::InitOptions) documentation.
//!
//! ### Embeddings generation
//!```
//!# use fastembed::{FlagEmbedding, InitOptions, EmbeddingModel, EmbeddingBase};
//!# fn embedding_demo() -> anyhow::Result<()> {
//!# let model: FlagEmbedding = FlagEmbedding::try_new(Default::default())?;
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
//! ### Generate query and passage embeddings
//!```
//!# use fastembed::{FlagEmbedding, InitOptions, EmbeddingModel, EmbeddingBase};
//!# fn query_passage_demo() -> anyhow::Result<()> {
//!# let model: FlagEmbedding = FlagEmbedding::try_new(Default::default())?;
//! let passages = vec![
//!     "This is the first passage. It contains provides more context for retrieval.",
//!     "Here's the second passage, which is longer than the first one. It includes additional information.",
//!     "And this is the third passage, the longest of all. It contains several sentences and is meant for more extensive testing."
//!    ];
//!
//! // Generate embeddings for the passages
//! // The texts are prefixed with "passage" for better results
//! // The batch size is set to 1 for demonstration purposes
//! let embeddings = model.passage_embed(passages, Some(1))?;
//!
//! println!("Passage embeddings length: {}", embeddings.len()); // -> Embeddings length: 3
//!
//! let query = "What is the answer to this generic question?";
//!
//! // Generate embeddings for the query
//! // The text is prefixed with "query" for better retrieval
//! let query_embedding = model.query_embed(query)?;
//!
//! println!("Query embedding dimension: {}", query_embedding.len()); // -> Query embedding dimension: 768
//! # Ok(())
//! # }
//! ```
//!
use std::{
    fmt::Display,
    fs::File,
    path::{Path, PathBuf},
    thread::available_parallelism,
};

use anyhow::{Ok, Result};
use hf_hub::api::sync::ApiRepo;
use hf_hub::{api::sync::ApiBuilder, Cache};
use ndarray::Array;
pub use ort::{ExecutionProvider, ExecutionProviderDispatch};
use ort::{GraphOptimizationLevel, Session, Value};
use rayon::{
    prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSlice,
};
use tokenizers::{AddedToken, PaddingParams, PaddingStrategy, TruncationParams};

const DEFAULT_BATCH_SIZE: usize = 256;
const DEFAULT_MAX_LENGTH: usize = 512;
const DEFAULT_CACHE_DIR: &str = "local_cache";
const DEFAULT_EMBEDDING_MODEL: EmbeddingModel = EmbeddingModel::BGESmallENV15;

/// Type alias for the embedding vector
pub type Embedding = Vec<f32>;

type Tokenizer = tokenizers::TokenizerImpl<
    tokenizers::ModelWrapper,
    tokenizers::NormalizerWrapper,
    tokenizers::PreTokenizerWrapper,
    tokenizers::PostProcessorWrapper,
    tokenizers::DecoderWrapper,
>;

/// Enum for the available models
#[derive(Debug, Clone, PartialEq, Eq)]
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

}

impl Display for EmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let model_info = FlagEmbedding::list_supported_models()
            .into_iter()
            .find(|model| model.model == *self)
            .unwrap();
        write!(f, "{}", model_info.model_code)
    }
}

/// Options for initializing the FlagEmbedding model
#[derive(Debug, Clone)]
pub struct InitOptions {
    pub model_name: EmbeddingModel,
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub max_length: usize,
    pub cache_dir: PathBuf,
    pub show_download_message: bool,
}

impl Default for InitOptions {
    fn default() -> Self {
        Self {
            model_name: DEFAULT_EMBEDDING_MODEL,
            execution_providers: Default::default(),
            max_length: DEFAULT_MAX_LENGTH,
            cache_dir: Path::new(DEFAULT_CACHE_DIR).to_path_buf(),
            show_download_message: true,
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

/// Base for implementing an embedding model
pub trait EmbeddingBase<S: AsRef<str>> {
    /// The base embedding method for generating sentence embeddings
    fn embed(&self, texts: Vec<S>, batch_size: Option<usize>) -> Result<Vec<Embedding>>;

    /// Generate sentence embeddings for passages, pre-fixed with "passage"
    fn passage_embed(&self, texts: Vec<S>, batch_size: Option<usize>) -> Result<Vec<Embedding>>;

    /// Generate embeddings for user queries pre-fixed with "query"
    fn query_embed(&self, query: S) -> Result<Embedding>;
}

/// Rust representation of the FlagEmbedding model
pub struct FlagEmbedding {
    tokenizer: Tokenizer,
    session: Session,
}

impl FlagEmbedding {
    /// Try to generate a new FlagEmbedding Instance
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
            show_download_message,
        } = options;

        let threads = available_parallelism()?.get() as i16;

        let model_repo =
            FlagEmbedding::retrieve_model(model_name.clone(), cache_dir, show_download_message)?;

        // The model files could be placed in subdirectories
        let model_file = model_repo
            .info()?
            .siblings
            .into_iter()
            .find(|f| {
                f.rfilename.ends_with("model.onnx") || f.rfilename.ends_with("model_optimized.onnx")
            })
            .unwrap();

        let model_file = model_repo.get(&model_file.rfilename)?;

        let session = Session::builder()?
            .with_execution_providers(execution_providers)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .with_model_from_file(model_file)?;

        let tokenizer = FlagEmbedding::load_tokenizer(model_repo, max_length)?;
        Ok(Self::new(tokenizer, session))
    }

    /// Private method to return an instance
    fn new(tokenizer: Tokenizer, session: Session) -> Self {
        Self { tokenizer, session }
    }
    /// Return the FlagEmbedding model's directory from cache or remote retrieval
    fn retrieve_model(
        model: EmbeddingModel,
        cache_dir: PathBuf,
        show_download_message: bool,
    ) -> Result<ApiRepo> {
        let cache = Cache::new(cache_dir);
        let api = ApiBuilder::from_cache(cache)
            .with_progress(show_download_message)
            .build()
            .unwrap();

        let repo = api.model(model.to_string());
        Ok(repo)
    }

    fn load_tokenizer(model_repo: ApiRepo, max_length: usize) -> Result<Tokenizer> {
        let config_path = model_repo.get("config.json")?;
        let file = File::open(config_path)?;
        let config: serde_json::Value = serde_json::from_reader(file)?;

        let tokenizer_config_path = model_repo.get("tokenizer_config.json")?;
        let file = File::open(tokenizer_config_path)?;
        let tokenizer_config: serde_json::Value = serde_json::from_reader(file)?;

        let special_tokens_map_path = model_repo.get("special_tokens_map.json")?;
        let file = File::open(special_tokens_map_path)?;
        let special_tokens_map: serde_json::Value = serde_json::from_reader(file)?;

        let tokenizer_path = model_repo.get("tokenizer.json")?;
        let mut tokenizer =
            tokenizers::Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;

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

    /// Retrieve a list of supported modelsc
    pub fn list_supported_models() -> Vec<ModelInfo> {
        vec![
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
            }
        ]
    }
}

/// EmbeddingBase implementation for FlagEmbedding
///
/// Generic type to accept String, &str, OsString, &OsStr
impl<S: AsRef<str> + Send + Sync> EmbeddingBase<S> for FlagEmbedding {
    // Method to generate sentence embeddings for a Vec of str refs
    fn embed(&self, texts: Vec<S>, batch_size: Option<usize>) -> Result<Vec<Embedding>> {
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

    // Method implememtation to generate passage embeddings prefixed with "passage"
    fn passage_embed(&self, texts: Vec<S>, batch_size: Option<usize>) -> Result<Vec<Embedding>> {
        let passages: Vec<String> = texts
            .par_iter()
            .map(|text| format!("passage: {}", text.as_ref()))
            .collect();
        self.embed(passages, batch_size)
    }

    // Method implementation for query embeddings prefixed with "query".
    fn query_embed(&self, query: S) -> Result<Embedding> {
        let query = format!("query: {}", query.as_ref());

        let embeddings = self.embed(vec![query], None)?[0].clone();

        Ok(embeddings)
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    const EPSILON: f32 = 1e-4;

    #[test]
    fn test_embeddings() {
        let models_and_expected_values = vec![
            (
                EmbeddingModel::AllMiniLML6V2,
                vec![0.02591, 0.00573, 0.01147, 0.03796, -0.0232],
            ),
            (
                EmbeddingModel::BGEBaseENV15,
                vec![0.01129394, 0.05493144, 0.02615099, 0.00328772, 0.02996045],
            ),
            (
                EmbeddingModel::BGESmallENV15,
                vec![0.01522374, -0.02271799, 0.00860278, -0.07424029, 0.00386434],
            ),
        ];

        for (model_name, expected) in models_and_expected_values {
            let model: FlagEmbedding = FlagEmbedding::try_new(InitOptions {
                model_name: model_name.clone(),
                ..Default::default()
            })
            .unwrap();

            let documents = vec!["hello world"];

            // Generate embeddings with the default batch size, 256
            let embeddings = model.embed(documents, None).unwrap();

            for (i, v) in expected.into_iter().enumerate() {
                let difference = (v - embeddings[0][i]).abs();
                assert!(
                    difference < EPSILON,
                    "Difference for {}: {}",
                    model_name,
                    difference
                )
            }
        }
    }
}
