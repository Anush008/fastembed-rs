#[cfg(feature = "online")]
use crate::common::load_tokenizer_hf_hub;
use crate::{
    models::sparse::{models_list, SparseModel},
    ModelInfo, SparseEmbedding,
};
#[cfg(feature = "online")]
use anyhow::Context;
use anyhow::Result;
#[cfg(feature = "online")]
use hf_hub::{
    api::sync::{ApiBuilder, ApiRepo},
    Cache,
};
use ndarray::{Array, CowArray};
use ort::{session::Session, value::Value};
#[cfg_attr(not(feature = "online"), allow(unused_imports))]
use rayon::{iter::ParallelIterator, slice::ParallelSlice};
#[cfg(feature = "online")]
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[cfg_attr(not(feature = "online"), allow(unused_imports))]
use std::thread::available_parallelism;

#[cfg(feature = "online")]
use super::SparseInitOptions;
use super::{SparseTextEmbedding, DEFAULT_BATCH_SIZE};

impl SparseTextEmbedding {
    /// Try to generate a new SparseTextEmbedding Instance
    ///
    /// Uses the highest level of Graph optimization
    ///
    /// Uses the total number of CPUs available as the number of intra-threads
    #[cfg(feature = "online")]
    pub fn try_new(options: SparseInitOptions) -> Result<Self> {
        use super::SparseInitOptions;
        use ort::{session::builder::GraphOptimizationLevel, session::Session};

        let SparseInitOptions {
            model_name,
            execution_providers,
            max_length,
            cache_dir,
            show_download_progress,
        } = options;

        let threads = available_parallelism()?.get();

        let model_repo = SparseTextEmbedding::retrieve_model(
            model_name.clone(),
            cache_dir.clone(),
            show_download_progress,
        )?;

        let model_file_name = SparseTextEmbedding::get_model_info(&model_name).model_file;
        let model_file_reference = model_repo
            .get(&model_file_name)
            .context(format!("Failed to retrieve {} ", model_file_name))?;

        let session = Session::builder()?
            .with_execution_providers(execution_providers)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(model_file_reference)?;

        let tokenizer = load_tokenizer_hf_hub(model_repo, max_length)?;
        Ok(Self::new(tokenizer, session, model_name))
    }

    /// Private method to return an instance
    #[cfg_attr(not(feature = "online"), allow(dead_code))]
    fn new(tokenizer: Tokenizer, session: Session, model: SparseModel) -> Self {
        let need_token_type_ids = session
            .inputs
            .iter()
            .any(|input| input.name == "token_type_ids");
        Self {
            tokenizer,
            session,
            need_token_type_ids,
            model,
        }
    }
    /// Return the SparseTextEmbedding model's directory from cache or remote retrieval
    #[cfg(feature = "online")]
    fn retrieve_model(
        model: SparseModel,
        cache_dir: PathBuf,
        show_download_progress: bool,
    ) -> Result<ApiRepo> {
        let cache = Cache::new(cache_dir);
        let api = ApiBuilder::from_cache(cache)
            .with_progress(show_download_progress)
            .build()?;

        let repo = api.model(model.to_string());
        Ok(repo)
    }

    /// Retrieve a list of supported models
    pub fn list_supported_models() -> Vec<ModelInfo<SparseModel>> {
        models_list()
    }

    /// Get ModelInfo from SparseModel
    pub fn get_model_info(model: &SparseModel) -> ModelInfo<SparseModel> {
        SparseTextEmbedding::list_supported_models()
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
    ) -> Result<Vec<SparseEmbedding>> {
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
                let owned_attention_mask =
                    Array::from_shape_vec((batch_size, encoding_length), mask_array)?;
                let attention_mask_array = CowArray::from(&owned_attention_mask);

                let token_type_ids_array =
                    Array::from_shape_vec((batch_size, encoding_length), typeids_array)?;

                let mut session_inputs = ort::inputs![
                    "input_ids" => Value::from_array(inputs_ids_array)?,
                    "attention_mask" => Value::from_array(&attention_mask_array)?,
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

                let output_data = outputs[last_hidden_state_key].try_extract_tensor::<f32>()?;

                let embeddings = self.model.post_process(&output_data, &attention_mask_array);

                Ok(embeddings)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();

        Ok(output)
    }
}
