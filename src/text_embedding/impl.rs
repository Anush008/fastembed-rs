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
        let prefix = TextEmbedding::get_prefix(&model_name);

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
            prefix,
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

        let session = {
            let mut session_builder = Session::builder()?
                .with_execution_providers(execution_providers)?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(threads)?;

            for external_initializer_file in model.external_initializers {
                session_builder = session_builder.with_external_initializer_file_in_memory(
                    external_initializer_file.file_name,
                    external_initializer_file.buffer.into(),
                )?;
            }

            session_builder.commit_from_memory(&model.onnx_file)?
        };

        let tokenizer = load_tokenizer(model.tokenizer_files, max_length)?;
        Ok(Self::new(
            tokenizer,
            session,
            model.pooling,
            model.quantization,
            model.output_key,
            None,
        ))
    }

    /// Private method to return an instance
    fn new(
        tokenizer: Tokenizer,
        session: Session,
        post_process: Option<Pooling>,
        quantization: QuantizationMode,
        output_key: Option<OutputKey>,
        prefix: Option<&'static str>,
    ) -> Self {
        let need_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");

        let need_position_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "position_ids");

        let need_task_id = session
            .inputs()
            .iter()
            .any(|input| input.name() == "task_id");

        // Count KV-cache layer pairs by counting `past_key_values.N.key` inputs.
        let kv_cache_layers = session
            .inputs()
            .iter()
            .filter(|i| i.name().starts_with("past_key_values.") && i.name().ends_with(".key"))
            .count();

        // Auto-detect kv_heads and head_dim from the first key tensor's shape.
        // Shape convention: [batch, kv_heads, seq, head_dim] — dims 1 and 3 are fixed.
        let (kv_cache_kv_heads, kv_cache_head_dim) = if kv_cache_layers > 0 {
            session
                .inputs()
                .iter()
                .find(|i| i.name() == "past_key_values.0.key")
                .and_then(|outlet| outlet.dtype().tensor_shape())
                .and_then(|shape| {
                    let s: &[i64] = shape;
                    if s.len() == 4 && s[1] > 0 && s[3] > 0 {
                        Some((s[1] as usize, s[3] as usize))
                    } else {
                        None
                    }
                })
                .unwrap_or((0, 0))
        } else {
            (0, 0)
        };

        // Detect static batch dimension from `input_ids` shape.
        // A positive (non-−1) batch dim means the model was exported with a fixed batch size.
        let max_batch_size = session
            .inputs()
            .iter()
            .find(|i| i.name() == "input_ids")
            .and_then(|outlet| outlet.dtype().tensor_shape())
            .and_then(|shape| {
                let s: &[i64] = shape;
                if !s.is_empty() && s[0] > 0 {
                    Some(s[0] as usize)
                } else {
                    None // dynamic batch dimension
                }
            });

        Self {
            tokenizer,
            session,
            need_token_type_ids,
            need_position_ids,
            need_task_id,
            kv_cache_layers,
            kv_cache_kv_heads,
            kv_cache_head_dim,
            pooling: post_process,
            quantization,
            output_key,
            max_batch_size,
            prefix,
        }
    }
    /// Return the TextEmbedding model's directory from cache or remote retrieval.
    ///
    /// Searches all directories listed in `FASTEMBED_CACHE_DIR` (colon-separated)
    /// before falling back to `cache_dir` for downloading.
    #[cfg(feature = "hf-hub")]
    fn retrieve_model(
        model: EmbeddingModel,
        cache_dir: PathBuf,
        show_download_progress: bool,
    ) -> anyhow::Result<ApiRepo> {
        use crate::common::{find_model_cache_dir, get_cache_dirs, pull_from_hf};

        let model_code = TextEmbedding::get_model_info(&model)?.model_code.clone();
        let all_dirs = get_cache_dirs();
        let effective_dir = find_model_cache_dir(&model_code, &all_dirs).unwrap_or(cache_dir);
        pull_from_hf(model_code, effective_dir, show_download_progress)
    }

    /// Return the static text prefix to prepend to every input for this model, if any.
    ///
    /// Models that require a fixed task prefix on all inputs (e.g. Jina-embeddings-v5
    /// variants that need `"Document: "` for symmetric retrieval) should be listed here.
    ///
    /// For asymmetric retrieval with separate query and document prefixes, callers should
    /// prepend the appropriate prefix themselves before passing text to `embed()`.
    pub fn get_prefix(model_name: &EmbeddingModel) -> Option<&'static str> {
        match model_name {
            // Jina-v5 nano (and other task-specific variants) require a task prefix on every
            // input.  "Document: " is used as the default for symmetric embedding (indexing).
            // For query encoding, prepend "Query: " to your input strings before embed().
            EmbeddingModel::JinaEmbeddingsV5Nano => Some("Document: "),
            _ => None,
        }
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
            EmbeddingModel::BGEM3 => Some(Pooling::Cls),

            EmbeddingModel::NomicEmbedTextV1 => Some(Pooling::Mean),
            EmbeddingModel::NomicEmbedTextV15 => Some(Pooling::Mean),
            EmbeddingModel::NomicEmbedTextV15Q => Some(Pooling::Mean),

            EmbeddingModel::ParaphraseMLMiniLML12V2 => Some(Pooling::Mean),
            EmbeddingModel::ParaphraseMLMiniLML12V2Q => Some(Pooling::Mean),
            EmbeddingModel::ParaphraseMLMpnetBaseV2 => Some(Pooling::Mean),
            EmbeddingModel::AllMpnetBaseV2 => Some(Pooling::Mean),

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
            EmbeddingModel::JinaEmbeddingsV2BaseEN => Some(Pooling::Mean),

            EmbeddingModel::EmbeddingGemma300M => Some(Pooling::Mean),

            EmbeddingModel::SnowflakeArcticEmbedXS => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedXSQ => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedS => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedSQ => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedM => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedMQ => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedMLong => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedMLongQ => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedL => Some(Pooling::Cls),
            EmbeddingModel::SnowflakeArcticEmbedLQ => Some(Pooling::Cls),

            // Calibrated uint8: affine dequant f32 = (u8 - zp) × scale
            // Parameters read from the QuantizeLinear initializers in dynamic_uint8.onnx
            // (quant_scale = 0.0027450979687273502, quant_zero_point = 109)
            EmbeddingModel::Qwen3Embedding0_6BUint8 => Some(Pooling::PrePooledU8 {
                scale: 0.002_745_098,
                zero_point: 109,
            }),
            EmbeddingModel::SnowflakeArcticEmbedLV2 => Some(Pooling::Cls),
            // Snowflake Arctic Embed M v2: same GTE architecture as L v2, uses CLS pooling
            EmbeddingModel::SnowflakeArcticEmbedMV2 => Some(Pooling::Cls),
            // PIXIE-Rune uses CLS pooling (pooling_mode_cls_token: true in 1_Pooling/config.json)
            EmbeddingModel::PixieRuneV1 => Some(Pooling::Cls),
            EmbeddingModel::PixieRuneV1Q => Some(Pooling::Cls),
            EmbeddingModel::PixieRuneV1Int4 => Some(Pooling::Cls),
            EmbeddingModel::PixieRuneV1Int4Full => Some(Pooling::Cls),
            // Jina v3: XLM-R + LoRA adapters. task_id=1 (retrieval.passage) is injected
            // automatically (need_task_id auto-detected from ONNX inputs). Mean pooling
            // over the 3D `text_embeds` output [batch, seq, 1024].
            EmbeddingModel::JinaEmbeddingsV3 => Some(Pooling::Mean),
            // Jina v5 Nano ships a pre-pooled 'sentence_embedding' output [batch, dim].
            // Cls on a 2D tensor is a no-op pass-through.
            EmbeddingModel::JinaEmbeddingsV5Nano => Some(Pooling::Cls),
            // Decoder-style models: take the last non-padding token
            EmbeddingModel::OctenEmbedding0_6BFp32 => Some(Pooling::LastToken),
            EmbeddingModel::OctenEmbedding0_6BInt4 => Some(Pooling::LastToken),
            EmbeddingModel::OctenEmbedding0_6BInt8Full => Some(Pooling::LastToken),
            // F2LLM-v2-0.6B: same Qwen3 decoder architecture, last-token pooling
            EmbeddingModel::F2LlmV2_0_6BFp32 => Some(Pooling::LastToken),
            EmbeddingModel::F2LlmV2_0_6BInt8 => Some(Pooling::LastToken),
            EmbeddingModel::F2LlmV2_0_6BInt4 => Some(Pooling::LastToken),
            EmbeddingModel::F2LlmV2_0_6BInt8Full => Some(Pooling::LastToken),
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
            EmbeddingModel::SnowflakeArcticEmbedXSQ => QuantizationMode::Dynamic,
            EmbeddingModel::SnowflakeArcticEmbedSQ => QuantizationMode::Dynamic,
            EmbeddingModel::SnowflakeArcticEmbedMQ => QuantizationMode::Dynamic,
            EmbeddingModel::SnowflakeArcticEmbedMLongQ => QuantizationMode::Dynamic,
            EmbeddingModel::SnowflakeArcticEmbedLQ => QuantizationMode::Dynamic,
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
        texts: impl AsRef<[S]>,
        batch_size: Option<usize>,
    ) -> Result<EmbeddingOutput> {
        let texts = texts.as_ref();
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

        // Enforce static batch dimension if detected from the ONNX graph.
        let batch_size = if let Some(max) = self.max_batch_size {
            if batch_size > max {
                return Err(anyhow::anyhow!(
                    "This model was exported with a static batch size of {max}. \
                     Pass `batch_size = Some({max})` (or embed fewer texts at a time)."
                ));
            }
            batch_size.min(max)
        } else {
            batch_size
        };

        let batches = texts
            .chunks(batch_size)
            .map(|batch| {
                // Encode the texts in the batch, prepending prefix if set
                let prefixed: Vec<String>;
                let inputs: Vec<&str> = if let Some(pfx) = self.prefix {
                    prefixed = batch
                        .iter()
                        .map(|text| format!("{}{}", pfx, text.as_ref()))
                        .collect();
                    prefixed.iter().map(|s| s.as_str()).collect()
                } else {
                    batch.iter().map(|text| text.as_ref()).collect()
                };
                let encodings = self.tokenizer.encode_batch(inputs, true).map_err(|e| {
                    anyhow::Error::msg(e.to_string()).context("Failed to encode the batch.")
                })?;

                // Extract the encoding length and batch size
                let encoding_length = encodings
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("Tokenizer returned empty encodings"))?
                    .len();
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

                if self.need_position_ids {
                    // Build position_ids [[0, 1, ..., seq-1], ...] for each batch item.
                    let pos_ids: Vec<i64> = (0..batch_size)
                        .flat_map(|_| 0..encoding_length as i64)
                        .collect();
                    let position_ids_array =
                        Array::from_shape_vec((batch_size, encoding_length), pos_ids)?;
                    session_inputs.push((
                        "position_ids".into(),
                        Value::from_array(position_ids_array)?.into(),
                    ));
                }

                if self.need_task_id {
                    // task_id=1 selects the retrieval.passage LoRA adapter.
                    // Jina-embeddings-v3 expects a scalar (0-D) int64 tensor, not [batch].
                    let task_id_scalar = ndarray::arr0(1i64);
                    session_inputs
                        .push(("task_id".into(), Value::from_array(task_id_scalar)?.into()));
                }

                if self.kv_cache_layers > 0 {
                    // Inject empty KV-cache tensors [batch, kv_heads, 0, head_dim] for each layer.
                    // Required by onnx-community-style decoder models that expect past_key_values.
                    for layer in 0..self.kv_cache_layers {
                        let kv_shape = (
                            batch_size,
                            self.kv_cache_kv_heads,
                            0usize,
                            self.kv_cache_head_dim,
                        );
                        let k_empty = ndarray::Array4::<f32>::zeros(kv_shape);
                        let v_empty = ndarray::Array4::<f32>::zeros(kv_shape);
                        session_inputs.push((
                            format!("past_key_values.{}.key", layer).into(),
                            Value::from_array(k_empty)?.into(),
                        ));
                        session_inputs.push((
                            format!("past_key_values.{}.value", layer).into(),
                            Value::from_array(v_empty)?.into(),
                        ));
                    }
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

    /// Method to generate sentence embeddings for a collection of texts.
    ///
    /// Accepts anything that can be referenced as a slice of elements implementing
    /// [`AsRef<str>`], such as `Vec<String>`, `Vec<&str>`, `&[String]`, or `&[&str]`.
    ///
    /// The output is a [`Vec`] of [`Embedding`]s.
    ///
    /// # Note
    ///
    /// This method is a higher level method than [`TextEmbedding::transform`] by utilizing
    /// the default output precedence and array transformer for the [`TextEmbedding`] model.
    pub fn embed<S: AsRef<str> + Send + Sync>(
        &mut self,
        texts: impl AsRef<[S]>,
        batch_size: Option<usize>,
    ) -> Result<Vec<Embedding>> {
        let batches = self.transform(texts.as_ref(), batch_size)?;
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
