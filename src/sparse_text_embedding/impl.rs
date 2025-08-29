#[cfg(feature = "hf-hub")]
use crate::common::load_tokenizer_hf_hub;
use crate::{
    models::sparse::{models_list, SparseModel},
    ModelInfo, SparseEmbedding,
};
#[cfg(feature = "hf-hub")]
use anyhow::Context;
use anyhow::Result;
#[cfg(feature = "hf-hub")]
use hf_hub::api::sync::ApiRepo;
use ndarray::{Array, ArrayViewD, Axis, CowArray, Dim};
use ort::{session::Session, value::Value};
#[cfg_attr(not(feature = "hf-hub"), allow(unused_imports))]
#[cfg(feature = "hf-hub")]
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[cfg_attr(not(feature = "hf-hub"), allow(unused_imports))]
use std::thread::available_parallelism;

#[cfg(feature = "hf-hub")]
use super::SparseInitOptions;
use super::{SparseTextEmbedding, DEFAULT_BATCH_SIZE};

impl SparseTextEmbedding {
    /// Try to generate a new SparseTextEmbedding Instance
    ///
    /// Uses the highest level of Graph optimization
    ///
    /// Uses the total number of CPUs available as the number of intra-threads
    #[cfg(feature = "hf-hub")]
    pub fn try_new(options: SparseInitOptions) -> Result<Self> {
        use super::SparseInitOptions;
        use ort::{session::builder::GraphOptimizationLevel, session::Session};

        let SparseInitOptions {
            max_length,
            model_name,
            cache_dir,
            show_download_progress,
            execution_providers,
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
    #[cfg_attr(not(feature = "hf-hub"), allow(dead_code))]
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
    #[cfg(feature = "hf-hub")]
    fn retrieve_model(
        model: SparseModel,
        cache_dir: PathBuf,
        show_download_progress: bool,
    ) -> Result<ApiRepo> {
        use crate::common::pull_from_hf;

        pull_from_hf(model.to_string(), cache_dir, show_download_progress)
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
        &mut self,
        texts: Vec<S>,
        batch_size: Option<usize>,
    ) -> Result<Vec<SparseEmbedding>> {
        // Determine the batch size, default if not specified
        let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE);

        let output = texts
            .chunks(batch_size)
            .map(|batch| {
                // Encode the texts in the batch
                let inputs = batch.iter().map(|text| text.as_ref()).collect();
                let encodings = self.tokenizer.encode_batch(inputs, true).map_err(|e| {
                    anyhow::Error::msg(e.to_string()).context("Failed to encode the batch.")
                })?;

                // Extract the encoding length and batch size
                let encoding_length = encodings[0].len();
                let batch_size = batch.len();

                let max_size = encoding_length * batch_size;

                // Preallocate arrays with the maximum size
                let mut ids_array = Vec::with_capacity(max_size);
                let mut mask_array = Vec::with_capacity(max_size);
                let mut type_ids_array = Vec::with_capacity(max_size);

                encodings.iter().for_each(|encoding| {
                    let ids = encoding.get_ids();
                    let mask = encoding.get_attention_mask();
                    let type_ids = encoding.get_type_ids();

                    ids_array.extend(ids.iter().map(|x| *x as i64));
                    mask_array.extend(mask.iter().map(|x| *x as i64));
                    type_ids_array.extend(type_ids.iter().map(|x| *x as i64));
                });

                let inputs_ids_array =
                    Array::from_shape_vec((batch_size, encoding_length), ids_array)?;
                let attention_mask_array =
                    Array::from_shape_vec((batch_size, encoding_length), mask_array)?;
                // removed CowArray usage, use owned array

                let token_type_ids_array =
                    Array::from_shape_vec((batch_size, encoding_length), type_ids_array)?;

                let mut session_inputs = ort::inputs![
                    "input_ids" => Value::from_array(inputs_ids_array)?,
                    "attention_mask" => Value::from_array(attention_mask_array.clone())?,
                ];

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

                let (shape, data) = outputs[last_hidden_state_key].try_extract_tensor::<f32>()?;
                let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                let output_array = ndarray::ArrayViewD::from_shape(shape.as_slice(), data)?;
                let attention_mask_cow = ndarray::CowArray::from(&attention_mask_array);

                let embeddings = SparseTextEmbedding::post_process(
                    &self.model,
                    &output_array,
                    &attention_mask_cow,
                );

                Ok(embeddings)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();

        Ok(output)
    }

    fn post_process(
        model_name: &SparseModel,
        model_output: &ArrayViewD<f32>,
        attention_mask: &CowArray<i64, Dim<[usize; 2]>>,
    ) -> Vec<SparseEmbedding> {
        match model_name {
            SparseModel::SPLADEPPV1 => {
                // Apply ReLU and logarithm transformation
                let relu_log = model_output.mapv(|x| (1.0 + x.max(0.0)).ln());

                // Convert to f32 and expand the dimensions
                let attention_mask = attention_mask.mapv(|x| x as f32).insert_axis(Axis(2));

                // Weight the transformed values by the attention mask
                let weighted_log = relu_log * attention_mask;

                // Get the max scores
                let scores = weighted_log.fold_axis(Axis(1), f32::NEG_INFINITY, |r, &v| r.max(v));

                scores
                    .rows()
                    .into_iter()
                    .map(|row_scores| {
                        let mut values: Vec<f32> = Vec::with_capacity(scores.len());
                        let mut indices: Vec<usize> = Vec::with_capacity(scores.len());

                        row_scores.into_iter().enumerate().for_each(|(idx, f)| {
                            if *f > 0.0 {
                                values.push(*f);
                                indices.push(idx);
                            }
                        });

                        SparseEmbedding { values, indices }
                    })
                    .collect()
            }
        }
    }
}
