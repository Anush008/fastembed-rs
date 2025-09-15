//! The definition of the main struct for text embeddings - [`TextEmbedding`].

#[cfg(feature = "hf-hub")]
use crate::common::load_tokenizer_hf_hub;
use crate::{
    common::load_tokenizer,
    models::{text_embedding::models_list, ModelTrait},
    pooling::Pooling,
    Embedding, EmbeddingModel, EmbeddingOutput, ModelInfo, OutputKey, QuantizationMode,
    SingleBatchOutput,
};
#[cfg(feature = "hf-hub")]
use anyhow::Context;
use anyhow::Result;
#[cfg(feature = "hf-hub")]
use hf_hub::api::sync::ApiRepo;
use ndarray::Array;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
#[cfg(feature = "hf-hub")]
use std::path::PathBuf;
use std::thread::available_parallelism;
use tokenizers::Tokenizer;

#[cfg(feature = "hf-hub")]
use super::TextInitOptions;
use super::{
    output, InitOptionsUserDefined, TextEmbedding, UserDefinedEmbeddingModel, DEFAULT_BATCH_SIZE,
};

impl TextEmbedding {
    /// Try to generate a new TextEmbedding Instance
    ///
    /// Uses the highest level of Graph optimization
    ///
    /// Uses the total number of CPUs available as the number of intra-threads
    #[cfg(feature = "hf-hub")]
    pub fn try_new(options: TextInitOptions) -> Result<Self> {
        let TextInitOptions {
            max_length,
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

        let model_info = TextEmbedding::get_model_info(&model_name)?;
        let model_file_name = &model_info.model_file;
        let model_file_reference = model_repo
            .get(model_file_name)
            .context(format!("Failed to retrieve {}", model_file_name))?;

        if !model_info.additional_files.is_empty() {
            for file in &model_info.additional_files {
                model_repo
                    .get(file)
                    .context(format!("Failed to retrieve {}", file))?;
            }
        }

        // prioritise loading pooling config if available, if not (thanks qdrant!), look for it in hardcoded
        let post_processing = TextEmbedding::get_default_pooling_method(&model_name);

        let session = Session::builder()?
            .with_execution_providers(execution_providers)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(model_file_reference)?;

        let tokenizer = load_tokenizer_hf_hub(model_repo, max_length)?;
        Ok(Self::new(
            tokenizer,
            session,
            post_processing,
            TextEmbedding::get_quantization_mode(&model_name),
            model_info.output_key.clone(),
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
        } = options;

        let threads = available_parallelism()?.get();

        let session = Session::builder()?
            .with_execution_providers(execution_providers)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_memory(&model.onnx_file)?;

        let tokenizer = load_tokenizer(model.tokenizer_files, max_length)?;
        Ok(Self::new(
            tokenizer,
            session,
            model.pooling,
            model.quantization,
            model.output_key,
        ))
    }

    /// Private method to return an instance
    fn new(
        tokenizer: Tokenizer,
        session: Session,
        post_process: Option<Pooling>,
        quantization: QuantizationMode,
        output_key: Option<OutputKey>,
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
            output_key,
        }
    }
    /// Return the TextEmbedding model's directory from cache or remote retrieval
    #[cfg(feature = "hf-hub")]
    fn retrieve_model(
        model: EmbeddingModel,
        cache_dir: PathBuf,
        show_download_progress: bool,
    ) -> anyhow::Result<ApiRepo> {
        use crate::common::pull_from_hf;

        pull_from_hf(model.to_string(), cache_dir, show_download_progress)
    }

    pub fn get_default_pooling_method(model_name: &EmbeddingModel) -> Option<Pooling> {
        match model_name {
            EmbeddingModel::AllMiniLML6V2 => Some(Pooling::Mean),
            EmbeddingModel::AllMiniLML6V2Q => Some(Pooling::Mean),
            EmbeddingModel::AllMiniLML12V2 => Some(Pooling::Mean),
            EmbeddingModel::AllMiniLML12V2Q => Some(Pooling::Mean),

            EmbeddingModel::BGEBaseENV15 => Some(Pooling::Cls),
            EmbeddingModel::BGEBaseENV15Q => Some(Pooling::Cls),
            EmbeddingModel::BGELargeENV15 => Some(Pooling::Cls),
            EmbeddingModel::BGELargeENV15Q => Some(Pooling::Cls),
            EmbeddingModel::BGESmallENV15 => Some(Pooling::Cls),
            EmbeddingModel::BGESmallENV15Q => Some(Pooling::Cls),
            EmbeddingModel::BGESmallZHV15 => Some(Pooling::Cls),
            EmbeddingModel::BGELargeZHV15 => Some(Pooling::Cls),

            EmbeddingModel::NomicEmbedTextV1 => Some(Pooling::Mean),
            EmbeddingModel::NomicEmbedTextV15 => Some(Pooling::Mean),
            EmbeddingModel::NomicEmbedTextV15Q => Some(Pooling::Mean),

            EmbeddingModel::ParaphraseMLMiniLML12V2 => Some(Pooling::Mean),
            EmbeddingModel::ParaphraseMLMiniLML12V2Q => Some(Pooling::Mean),
            EmbeddingModel::ParaphraseMLMpnetBaseV2 => Some(Pooling::Mean),

            EmbeddingModel::ModernBertEmbedLarge => Some(Pooling::Mean),

            EmbeddingModel::MultilingualE5Base => Some(Pooling::Mean),
            EmbeddingModel::MultilingualE5Small => Some(Pooling::Mean),
            EmbeddingModel::MultilingualE5Large => Some(Pooling::Mean),

            EmbeddingModel::MxbaiEmbedLargeV1 => Some(Pooling::Cls),
            EmbeddingModel::MxbaiEmbedLargeV1Q => Some(Pooling::Cls),

            EmbeddingModel::GTEBaseENV15 => Some(Pooling::Cls),
            EmbeddingModel::GTEBaseENV15Q => Some(Pooling::Cls),
            EmbeddingModel::GTELargeENV15 => Some(Pooling::Cls),
            EmbeddingModel::GTELargeENV15Q => Some(Pooling::Cls),

            EmbeddingModel::ClipVitB32 => Some(Pooling::Mean),

            EmbeddingModel::JinaEmbeddingsV2BaseCode => Some(Pooling::Mean),

            EmbeddingModel::EmbeddingGemma300M => Some(Pooling::Mean),
        }
    }

    /// Get the quantization mode of the model.
    ///
    /// Any models with a `Q` suffix in their name are quantized models.
    ///
    /// Currently only 6 supported models have dynamic quantization:
    /// - Alibaba-NLP/gte-base-en-v1.5
    /// - Alibaba-NLP/gte-large-en-v1.5
    /// - mixedbread-ai/mxbai-embed-large-v1
    /// - nomic-ai/nomic-embed-text-v1.5
    /// - Xenova/all-MiniLM-L12-v2
    /// - Xenova/all-MiniLM-L6-v2
    ///
    // TODO: Update this list when more models are added
    pub fn get_quantization_mode(model_name: &EmbeddingModel) -> QuantizationMode {
        match model_name {
            EmbeddingModel::AllMiniLML6V2Q => QuantizationMode::Dynamic,
            EmbeddingModel::AllMiniLML12V2Q => QuantizationMode::Dynamic,
            EmbeddingModel::BGEBaseENV15Q => QuantizationMode::Static,
            EmbeddingModel::BGELargeENV15Q => QuantizationMode::Static,
            EmbeddingModel::BGESmallENV15Q => QuantizationMode::Static,
            EmbeddingModel::NomicEmbedTextV15Q => QuantizationMode::Dynamic,
            EmbeddingModel::ParaphraseMLMiniLML12V2Q => QuantizationMode::Static,
            EmbeddingModel::MxbaiEmbedLargeV1Q => QuantizationMode::Dynamic,
            EmbeddingModel::GTEBaseENV15Q => QuantizationMode::Dynamic,
            EmbeddingModel::GTELargeENV15Q => QuantizationMode::Dynamic,
            _ => QuantizationMode::None,
        }
    }

    /// Retrieve a list of supported models
    pub fn list_supported_models() -> Vec<ModelInfo<EmbeddingModel>> {
        models_list()
    }

    /// Get ModelInfo from EmbeddingModel
    pub fn get_model_info(model: &EmbeddingModel) -> Result<&ModelInfo<EmbeddingModel>> {
        EmbeddingModel::get_model_info(model).ok_or_else(|| {
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
    /// If you want to choose a different export key or customize the way the batch
    /// arrays are aggregated, you can define your own array transformer
    /// and use it on [`EmbeddingOutput::export_with_transformer`] to extract the
    /// embeddings with your custom output type.
    pub fn transform<S: AsRef<str> + Send + Sync>(
        &mut self,
        texts: Vec<S>,
        batch_size: Option<usize>,
    ) -> Result<EmbeddingOutput> {
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

        let batches = texts
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

                let outputs_map = self
                    .session
                    .run(session_inputs)
                    .map_err(anyhow::Error::new)?
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect();
                Ok(SingleBatchOutput {
                    outputs: outputs_map,
                    attention_mask_array,
                })
            })
            .collect::<Result<Vec<_>>>()?;

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
        &mut self,
        texts: Vec<S>,
        batch_size: Option<usize>,
    ) -> Result<Vec<Embedding>> {
        let batches = self.transform(texts, batch_size)?;
        if let Some(output_key) = &self.output_key {
            batches.export_with_transformer(output::transformer_with_precedence(
                output_key,
                self.pooling.clone(),
            ))
        } else {
            batches.export_with_transformer(output::transformer_with_precedence(
                output::OUTPUT_TYPE_PRECEDENCE,
                self.pooling.clone(),
            ))
        }
    }
}
