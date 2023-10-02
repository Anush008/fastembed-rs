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
//!     model_name: EmbeddingModel::BGEBaseEN,
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
    path::{Path, PathBuf},
    thread::available_parallelism,
};

use anyhow::{Ok, Result};
use flate2::read::GzDecoder;
use ndarray::{Array, CowArray};
pub use ort::ExecutionProvider;
use ort::{Environment, GraphOptimizationLevel, Session, SessionBuilder, Value};
use rayon::{
    prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSlice,
};
use tar::Archive;
use tokenizers::{Encoding, PaddingParams, PaddingStrategy, TruncationParams};

const DEFAULT_BATCH_SIZE: usize = 256;
const DEFAULT_MAX_LENGTH: usize = 512;
const DEFAULT_CACHE_DIR: &str = "local_cache";
const DEFAULT_EMBEDDING_MODEL: EmbeddingModel = EmbeddingModel::BGESmallEN;

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
#[derive(Debug, Clone)]
pub enum EmbeddingModel {
    /// Sentence Transformer model, MiniLM-L6-v2
    AllMiniLML6V2,
    /// Base English model
    BGEBaseEN,
    /// Fast and Default English model
    BGESmallEN,
    /// Multilingual model, e5-large. Recommend using this model for non-English languages.
    MLE5Large,
}

impl ToString for EmbeddingModel {
    fn to_string(&self) -> String {
        match self {
            EmbeddingModel::AllMiniLML6V2 => String::from("fast-all-MiniLM-L6-v2"),
            EmbeddingModel::BGEBaseEN => String::from("fast-bge-base-en"),
            EmbeddingModel::BGESmallEN => String::from("fast-bge-small-en"),
            EmbeddingModel::MLE5Large => String::from("intfloat-multilingual-e5-large"),
        }
    }
}

/// Options for initializing the FlagEmbedding model
#[derive(Debug, Clone)]
pub struct InitOptions {
    pub model_name: EmbeddingModel,
    pub execution_providers: Vec<ExecutionProvider>,
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
}

/// Base class for implemnting an embedding model
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
    model: Session,
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

        let model_path =
            FlagEmbedding::retrieve_model(model_name, &cache_dir, show_download_message)?;

        let environment = Environment::builder()
            .with_name("Fastembed")
            .with_execution_providers(execution_providers)
            .build()?;
        let model = SessionBuilder::new(&environment.into())?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .with_model_from_file(model_path.join("model_optimized.onnx"))?;

        let mut tokenizer =
            tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json")).unwrap();
        let tokenizer: Tokenizer = tokenizer
            .with_truncation(Some(TruncationParams {
                max_length,
                ..Default::default()
            }))
            .unwrap()
            .with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::Fixed(max_length),
                pad_token: "[PAD]".into(),
                ..Default::default()
            }))
            .clone();
        Ok(Self::new(tokenizer, model))
    }

    /// Private method to return an instance
    fn new(tokenizer: Tokenizer, model: Session) -> Self {
        Self { tokenizer, model }
    }

    /// Download and unpack the model from Google Cloud Storage
    fn download_from_gcs(fast_model_name: &str, output_directory: &PathBuf) -> Result<()> {
        let download_url =
            format!("https://storage.googleapis.com/qdrant-fastembed/{fast_model_name}.tar.gz");

        let response = minreq::get(download_url).send()?;

        let data = match response.status_code {
            200..=299 => response.as_bytes(),
            _ => anyhow::bail!(
                "{} {}: {}",
                response.status_code,
                response.reason_phrase,
                response.as_str()?
            ),
        };

        let tar = GzDecoder::new(data);
        let mut archive = Archive::new(tar);
        archive.unpack(output_directory)?;
        Ok(())
    }

    /// Return the FlagEmbedding model's directory from cache or remote retrieval
    fn retrieve_model(
        model: EmbeddingModel,
        cache_dir: &PathBuf,
        show_download_message: bool,
    ) -> Result<PathBuf> {
        let fast_model_name = model.to_string();
        let output_path = Path::new(&cache_dir).join(&fast_model_name);

        if output_path.exists() {
            return Ok(output_path);
        }

        // A progress indicator hasn't been implemented as it doesn't seem possible with a synchronous flow and minreq
        // I've kept the lib sync as I'd love users to be able to try it out quickly without a dependency on tokio or some heavy async runtime
        if show_download_message {
            println!("Downloading {} model", fast_model_name);
        }

        FlagEmbedding::download_from_gcs(&fast_model_name, cache_dir)?;

        Ok(output_path)
    }

    /// Retrieve a list of supported modelsc
    pub fn list_supported_models() -> Vec<ModelInfo> {
        vec![ModelInfo {
            model: EmbeddingModel::AllMiniLML6V2,
            dim: 384,
            description: String::from("Sentence Transformer model, MiniLM-L6-v2"),
        },
        ModelInfo {
            model: EmbeddingModel::BGEBaseEN,
            dim: 768,
            description: String::from("Base English model"),
        },
        ModelInfo {
            model: EmbeddingModel::BGESmallEN,
            dim: 384,
            description: String::from("Fast and Default English model"),
        },
        ModelInfo {
            model: EmbeddingModel::MLE5Large,
            dim: 1024,
            description: String::from("Multilingual model, e5-large. Recommend using this model for non-English languages."),
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
                let encodings: Vec<Encoding> = batch
                    .iter()
                    .map(|text| self.tokenizer.encode(text.as_ref(), true).unwrap())
                    .collect();

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
                let inputs_ids_array = CowArray::from(Array::from_shape_vec(
                    (batch_size, encoding_length),
                    ids_array,
                )?)
                .into_dyn();

                let attention_mask_array = CowArray::from(Array::from_shape_vec(
                    (batch_size, encoding_length),
                    mask_array,
                )?)
                .into_dyn();

                let token_type_ids_array = CowArray::from(Array::from_shape_vec(
                    (batch_size, encoding_length),
                    typeids_array,
                )?)
                .into_dyn();

                // Run the model with inputs
                let outputs = self.model.run(vec![
                    Value::from_array(self.model.allocator(), &inputs_ids_array)?,
                    Value::from_array(self.model.allocator(), &attention_mask_array)?,
                    Value::from_array(self.model.allocator(), &token_type_ids_array)?,
                ])?;

                // Extract and normalize embeddings
                let output_data = outputs[0].try_extract::<f32>()?;
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

    // Method implementation for query embeddings. Prefixed with "query"
    fn query_embed(&self, query: S) -> Result<Embedding> {
        let query = format!("query: {}", query.as_ref());
        let query_embedding = self.embed(vec![&query], None);
        Ok(query_embedding?[0].to_owned())
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
