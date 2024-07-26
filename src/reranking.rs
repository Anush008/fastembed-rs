use anyhow::Result;
use std::{
    fmt::Display,
    path::{Path, PathBuf},
    thread::available_parallelism,
};

#[cfg(feature = "online")]
use crate::common::load_tokenizer_hf_hub;
use crate::common::{load_tokenizer, Tokenizer, TokenizerFiles, DEFAULT_CACHE_DIR};
#[cfg(feature = "online")]
use hf_hub::{api::sync::ApiBuilder, Cache};
use ndarray::{s, Array};
use ort::{ExecutionProviderDispatch, GraphOptimizationLevel, Session, Value};
use rayon::{iter::ParallelIterator, slice::ParallelSlice};

use crate::{models::reranking::reranker_model_list, RerankerModel, RerankerModelInfo};

const DEFAULT_RE_RANKER_MODEL: RerankerModel = RerankerModel::BGERerankerBase;
const DEFAULT_BATCH_SIZE: usize = 256;
const DEFAULT_MAX_LENGTH: usize = 512;

pub struct TextRerank {
    pub tokenizer: Tokenizer,
    session: Session,
    need_token_type_ids: bool,
}

/// Options for initializing the reranking model
#[derive(Debug, Clone)]
pub struct RerankInitOptions {
    pub model_name: RerankerModel,
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub max_length: usize,
    pub cache_dir: PathBuf,
    pub show_download_progress: bool,
}

impl Default for RerankInitOptions {
    fn default() -> Self {
        Self {
            model_name: DEFAULT_RE_RANKER_MODEL,
            execution_providers: Default::default(),
            max_length: DEFAULT_MAX_LENGTH,
            cache_dir: Path::new(DEFAULT_CACHE_DIR).to_path_buf(),
            show_download_progress: true,
        }
    }
}

/// Options for initializing UserDefinedRerankerModel
///
/// Model files are held by the UserDefinedRerankerModel struct
/// #[derive(Debug, Clone)]
pub struct RerankInitOptionsUserDefined {
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub max_length: usize,
}

impl Default for RerankInitOptionsUserDefined {
    fn default() -> Self {
        Self {
            execution_providers: Default::default(),
            max_length: DEFAULT_MAX_LENGTH,
        }
    }
}

/// Convert RerankInitOptions to RerankInitOptionsUserDefined
///
/// This is useful for when the user wants to use the same options for both the default and user-defined models
impl From<RerankInitOptions> for RerankInitOptionsUserDefined {
    fn from(options: RerankInitOptions) -> Self {
        RerankInitOptionsUserDefined {
            execution_providers: options.execution_providers,
            max_length: options.max_length,
        }
    }
}

/// Struct for "bring your own" reranking models
///
/// The onnx_file and tokenizer_files are expecting the files' bytes
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserDefinedRerankingModel {
    pub onnx_file: Vec<u8>,
    pub tokenizer_files: TokenizerFiles,
}

impl Display for RerankerModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let model_info = TextRerank::list_supported_models()
            .into_iter()
            .find(|model| model.model == *self)
            .expect("Model not found in supported models list.");
        write!(f, "{}", model_info.model_code)
    }
}

impl TextRerank {
    fn new(tokenizer: Tokenizer, session: Session) -> Self {
        let need_token_type_ids = session
            .inputs
            .iter()
            .any(|input| input.name == "token_type_ids");
        Self {
            tokenizer,
            session,
            need_token_type_ids,
        }
    }

    pub fn get_model_info(model: &RerankerModel) -> RerankerModelInfo {
        TextRerank::list_supported_models()
            .into_iter()
            .find(|m| &m.model == model)
            .expect("Model not found.")
    }

    pub fn list_supported_models() -> Vec<RerankerModelInfo> {
        reranker_model_list()
    }

    #[cfg(feature = "online")]
    pub fn try_new(options: RerankInitOptions) -> Result<TextRerank> {
        let RerankInitOptions {
            model_name,
            execution_providers,
            max_length,
            cache_dir,
            show_download_progress,
        } = options;

        let threads = available_parallelism()?.get();

        let cache = Cache::new(cache_dir);
        let api = ApiBuilder::from_cache(cache)
            .with_progress(show_download_progress)
            .build()
            .expect("Failed to build API from cache");
        let model_repo = api.model(model_name.to_string());

        let model_file_name = TextRerank::get_model_info(&model_name).model_file;
        let model_file_reference = model_repo
            .get(&model_file_name)
            .unwrap_or_else(|_| panic!("Failed to retrieve model file: {}", model_file_name));

        let session = Session::builder()?
            .with_execution_providers(execution_providers)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(model_file_reference)?;

        let tokenizer = load_tokenizer_hf_hub(model_repo, max_length)?;
        Ok(Self::new(tokenizer, session))
    }

    /// Create a TextRerank instance from model files provided by the user.
    ///
    /// This can be used for 'bring your own' reranking models
    pub fn try_new_from_user_defined(
        model: UserDefinedRerankingModel,
        options: RerankInitOptionsUserDefined,
    ) -> Result<Self> {
        let RerankInitOptionsUserDefined {
            execution_providers,
            max_length,
        } = options;

        let threads = available_parallelism()?.get();

        let session = Session::builder()?
            .with_execution_providers(execution_providers)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_memory(&model.onnx_file)?;

        let tokenizer = load_tokenizer(model.tokenizer_files, max_length)?;
        Ok(Self::new(tokenizer, session))
    }

    /// Reranks documents using the reranker model and returns the results sorted by score in descending order.
    pub fn rerank<S: AsRef<str> + Send + Sync>(
        &self,
        query: S,
        documents: Vec<S>,
        return_documents: bool,
        batch_size: Option<usize>,
    ) -> Result<Vec<RerankResult>> {
        let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE);

        let q = query.as_ref();

        let scores: Vec<f32> = documents
            .par_chunks(batch_size)
            .map(|batch| {
                let inputs = batch.iter().map(|d| (q, d.as_ref())).collect();

                let encodings = self
                    .tokenizer
                    .encode_batch(inputs, true)
                    .expect("Failed to encode batch");

                let encoding_length = encodings[0].len();
                let batch_size = batch.len();

                let max_size = encoding_length * batch_size;

                let mut ids_array = Vec::with_capacity(max_size);
                let mut mask_array = Vec::with_capacity(max_size);
                let mut typeids_array = Vec::with_capacity(max_size);

                encodings.iter().for_each(|encoding| {
                    let ids = encoding.get_ids();
                    let mask = encoding.get_attention_mask();
                    let typeids = encoding.get_type_ids();

                    ids_array.extend(ids.iter().map(|x| *x as i64));
                    mask_array.extend(mask.iter().map(|x| *x as i64));
                    typeids_array.extend(typeids.iter().map(|x| *x as i64));
                });

                let inputs_ids_array =
                    Array::from_shape_vec((batch_size, encoding_length), ids_array)?;

                let attention_mask_array =
                    Array::from_shape_vec((batch_size, encoding_length), mask_array)?;

                let token_type_ids_array =
                    Array::from_shape_vec((batch_size, encoding_length), typeids_array)?;

                let mut session_inputs = ort::inputs![
                    "input_ids" => Value::from_array(inputs_ids_array)?,
                    "attention_mask" => Value::from_array(attention_mask_array)?,
                ]?;

                if self.need_token_type_ids {
                    session_inputs.push((
                        "token_type_ids".into(),
                        Value::from_array(token_type_ids_array)?.into(),
                    ));
                }

                let outputs = self.session.run(session_inputs)?;

                let outputs = outputs["logits"]
                    .try_extract_tensor::<f32>()
                    .expect("Failed to extract logits tensor");

                let scores: Vec<f32> = outputs
                    .slice(s![.., 0])
                    .rows()
                    .into_iter()
                    .flat_map(|row| row.to_vec())
                    .collect();

                Ok(scores)
            })
            .flat_map(|result: Result<Vec<f32>, anyhow::Error>| result.unwrap())
            .collect();

        // Return top_n_result of type Vec<RerankResult> ordered by score in descending order, don't use binary heap
        let mut top_n_result: Vec<RerankResult> = scores
            .into_iter()
            .enumerate()
            .map(|(index, score)| RerankResult {
                document: return_documents.then(|| documents[index].as_ref().to_string()),
                score,
                index,
            })
            .collect();

        top_n_result.sort_by(|a, b| a.score.total_cmp(&b.score).reverse());

        Ok(top_n_result.to_vec())
    }
}

/// Rerank result.
#[derive(Debug, PartialEq, Clone)]
pub struct RerankResult {
    pub document: Option<String>,
    pub score: f32,
    pub index: usize,
}
