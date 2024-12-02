//! The definition of the main struct for text embeddings - [`TextEmbedding`].

#[cfg(feature = "online")]
use crate::common::load_tokenizer_hf_hub;
use crate::{
    common::load_tokenizer,
    models::text_embedding::{get_model_info, models_list},
    pooling::Pooling,
    Embedding, EmbeddingModel, EmbeddingOutput, ModelInfo, QuantizationMode, SingleBatchOutput,
};
#[cfg(feature = "online")]
use anyhow::Context;
use anyhow::Result;
#[cfg(feature = "online")]
use hf_hub::{
    api::sync::{ApiBuilder, ApiRepo},
    Cache,
};
use ndarray::Array;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use rayon::{
    iter::{FromParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};
#[cfg(feature = "online")]
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[cfg(feature = "online")]
use super::InitOptions;
use super::{
    output, InitOptionsUserDefined, TextEmbedding, UserDefinedEmbeddingModel, DEFAULT_BATCH_SIZE,
};

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
            max_length,
            cache_dir,
            show_download_progress,
            node_thread_nums,
            graph_thread_nums,
            parallel_execution,
        } = options;

        let model_repo = TextEmbedding::retrieve_model(
            model_name.clone(),
            cache_dir.clone(),
            show_download_progress,
        )?;

        let model_info = TextEmbedding::get_model_info(&model_name)?;
        let model_file_name = &model_info.model_file;
        let model_file_reference = model_repo
            .get(model_file_name)
            .context(format!("Failed to retrieve {}", model_file_name))?;

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
            .with_intra_threads(node_thread_nums.get())?
            .with_inter_threads(graph_thread_nums.get())?
            .with_parallel_execution(parallel_execution)?
            .commit_from_file(model_file_reference)?;

        let tokenizer = load_tokenizer_hf_hub(model_repo, max_length)?;
        Ok(Self::new(
            tokenizer,
            session,
            post_processing,
            model_name.get_quantization_mode(),
        ))
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
            max_length,
            node_thread_nums,
            graph_thread_nums,
            parallel_execution,
        } = options;

        let session = Session::builder()?
            .with_execution_providers(execution_providers)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(node_thread_nums.get())?
            .with_inter_threads(graph_thread_nums.get())?
            .with_parallel_execution(parallel_execution)?
            .commit_from_memory(&model.onnx_file)?;

        let tokenizer = load_tokenizer(model.tokenizer_files, max_length)?;
        Ok(Self::new(
            tokenizer,
            session,
            model.pooling,
            model.quantization,
        ))
    }

    /// Private method to return an instance
    fn new(
        tokenizer: Tokenizer,
        session: Session,
        post_process: Option<Pooling>,
        quantization: QuantizationMode,
    ) -> Self {
        let need_token_type_ids = session
            .inputs
            .iter()
            .any(|input| input.name == "token_type_ids");
        Self {
            tokenizer,
            session,
            need_token_type_ids,
            pooling: post_process,
            quantization,
        }
    }
    /// Return the TextEmbedding model's directory from cache or remote retrieval
    #[cfg(feature = "online")]
    fn retrieve_model(
        model: EmbeddingModel,
        cache_dir: PathBuf,
        show_download_progress: bool,
    ) -> anyhow::Result<ApiRepo> {
        let cache = Cache::new(cache_dir);
        let api = ApiBuilder::from_cache(cache)
            .with_progress(show_download_progress)
            .build()?;

        let repo = api.model(model.to_string());
        Ok(repo)
    }

    /// Retrieve a list of supported models
    pub fn list_supported_models() -> Vec<ModelInfo<EmbeddingModel>> {
        models_list()
    }

    /// Get ModelInfo from EmbeddingModel
    pub fn get_model_info(model: &EmbeddingModel) -> Result<&ModelInfo<EmbeddingModel>> {
        get_model_info(model).ok_or_else(|| {
            anyhow::Error::msg(format!(
                "Model {model:?} not found. Please check if the model is supported \
                by the current version."
            ))
        })
    }

    /// Method to generate an [`ort::SessionOutputs`] wrapped in a [`EmbeddingOutput`]
    /// instance, which can be used to extract the embeddings with default or custom
    /// methods as well as output key precedence.
    ///
    /// Metadata that could be useful for creating the array transformer is
    /// returned alongside the [`EmbeddingOutput`] instance, such as pooling methods
    /// etc.
    ///
    /// # Note
    ///
    /// This is a lower level method than [`TextEmbedding::embed`], and is useful
    /// when you need to extract the session outputs in a custom way.
    ///
    /// If you want to extract the embeddings directly, use [`TextEmbedding::embed`].
    ///
    /// If you want to use the raw session outputs, use [`EmbeddingOutput::into_raw`]
    /// on the output of this method.
    ///
    /// If you want to choose a different export key or customise the way the batch
    /// arrays are aggregated, you can define your own array transformer
    /// and use it on [`EmbeddingOutput::export_with_transformer`] to extract the
    /// embeddings with your custom output type.
    pub fn transform<'e, 'r, 's, S: AsRef<str> + Send + Sync>(
        &'e self,
        texts: Vec<S>,
        batch_size: Option<usize>,
    ) -> Result<EmbeddingOutput<'r, 's>>
    where
        'e: 'r,
        'e: 's,
    {
        // Determine the batch size according to the quantization method used.
        // Default if not specified
        let batch_size = match self.quantization {
            QuantizationMode::Dynamic => {
                if let Some(batch_size) = batch_size {
                    if batch_size < texts.len() {
                        Err(anyhow::Error::msg(
                            "Dynamic quantization cannot be used with batching. \
                            This is due to the dynamic quantization process adjusting \
                            the data range to fit each batch, making the embeddings \
                            incompatible across batches. Try specifying a batch size \
                            of `None`, or use a model with static or no quantization.",
                        ))
                    } else {
                        Ok(texts.len())
                    }
                } else {
                    Ok(texts.len())
                }
            }
            _ => Ok(batch_size.unwrap_or(DEFAULT_BATCH_SIZE)),
        }?;

        let batches =
            // anyhow::Result::<Vec<_>>::from_par_iter(texts.par_chunks(batch_size).map(|batch| {
                anyhow::Result::<Vec<_>>::from_iter(texts.chunks(batch_size).map(|batch| {
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
            let inputs_ids_array = Array::from_shape_vec((batch_size, encoding_length), ids_array)?;

            let attention_mask_array =
                Array::from_shape_vec((batch_size, encoding_length), mask_array)?;

            let token_type_ids_array =
                Array::from_shape_vec((batch_size, encoding_length), typeids_array)?;

            let mut session_inputs = ort::inputs![
                "input_ids" => Value::from_array(inputs_ids_array)?,
                "attention_mask" => Value::from_array(attention_mask_array.view())?,
            ]?;

            if self.need_token_type_ids {
                session_inputs.push((
                    "token_type_ids".into(),
                    Value::from_array(token_type_ids_array)?.into(),
                ));
            }

            Ok(
                // Package all the data required for post-processing (e.g. pooling)
                // into a SingleBatchOutput struct.
                SingleBatchOutput {
                    session_outputs: self
                        .session
                        .run(session_inputs)
                        .map_err(anyhow::Error::new)?,
                    attention_mask_array,
                },
            )
        }))?;

        Ok(EmbeddingOutput::new(batches))
    }

    /// Method to generate sentence embeddings for a Vec of texts.
    ///
    /// Accepts a [`Vec`] consisting of elements of either [`String`], &[`str`],
    /// [`std::ffi::OsString`], &[`std::ffi::OsStr`].
    ///
    /// The output is a [`Vec`] of [`Embedding`]s.
    ///
    /// # Note
    ///
    /// This method is a higher level method than [`TextEmbedding::transform`] by utilizing
    /// the default output precedence and array transformer for the [`TextEmbedding`] model.
    pub fn embed<S: AsRef<str> + Send + Sync>(
        &self,
        texts: Vec<S>,
        batch_size: Option<usize>,
    ) -> Result<Vec<Embedding>> {
        let batches = self.transform(texts, batch_size)?;

        batches.export_with_transformer(output::transformer_with_precedence(
            output::OUTPUT_TYPE_PRECENDENCE,
            self.pooling.clone(),
        ))
    }
}
