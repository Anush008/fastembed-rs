use std::{collections::HashSet, fmt::Display, str::FromStr, thread::available_parallelism};

use anyhow::{Context, Result};
#[cfg(feature = "hf-hub")]
use hf_hub::api::sync::ApiRepo;
use ndarray::Array;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
#[cfg(feature = "hf-hub")]
use std::path::PathBuf;
use tokenizers::Tokenizer;

use crate::common::load_tokenizer;
#[cfg(feature = "hf-hub")]
use crate::common::load_tokenizer_hf_hub;

use super::{
    LateInteractionEmbedding, LateInteractionInitOptions, LateInteractionInitOptionsUserDefined,
    LateInteractionModel, LateInteractionModelInfo, LateInteractionTextEmbedding,
    UserDefinedLateInteractionModel, DEFAULT_BATCH_SIZE,
};

/// All punctuation characters for skip list
fn get_punctuation_chars() -> Vec<char> {
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~".chars().collect()
}

fn models_list() -> Vec<LateInteractionModelInfo> {
    vec![
        LateInteractionModelInfo {
            model: LateInteractionModel::ColBERTV2,
            dim: 128,
            description: String::from(
                "Text embeddings, Unimodal (text), English, 512 input tokens truncation, 2023 year",
            ),
            model_code: String::from("colbert-ir/colbertv2.0"),
            model_file: String::from("model.onnx"),
            additional_files: Vec::new(),
            query_marker_token_id: 1,
            document_marker_token_id: 2,
            mask_token: String::from("[MASK]"),
            min_query_length: 31,
        },
        LateInteractionModelInfo {
            model: LateInteractionModel::AnswerAIColBERTSmallV1,
            dim: 96,
            description: String::from(
                "Text embeddings, Unimodal (text), English, 512 input tokens truncation, 2024 year",
            ),
            model_code: String::from("answerdotai/answerai-colbert-small-v1"),
            model_file: String::from("vespa_colbert.onnx"),
            additional_files: Vec::new(),
            query_marker_token_id: 1,
            document_marker_token_id: 2,
            mask_token: String::from("[MASK]"),
            min_query_length: 31,
        },
    ]
}

impl Display for LateInteractionModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let model_info = models_list()
            .into_iter()
            .find(|m| m.model == *self)
            .ok_or(std::fmt::Error)?;
        write!(f, "{}", model_info.model_code)
    }
}

impl FromStr for LateInteractionModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        models_list()
            .into_iter()
            .find(|m| m.model_code.eq_ignore_ascii_case(s))
            .map(|m| m.model)
            .ok_or_else(|| format!("Unknown late interaction model: {s}"))
    }
}

impl TryFrom<String> for LateInteractionModel {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

impl LateInteractionTextEmbedding {
    /// Try to create a new LateInteractionTextEmbedding instance
    #[cfg(feature = "hf-hub")]
    pub fn try_new(options: LateInteractionInitOptions) -> Result<Self> {
        let LateInteractionInitOptions {
            max_length,
            model_name,
            execution_providers,
            cache_dir,
            show_download_progress,
        } = options;

        let threads = available_parallelism()?.get();
        let model_info = Self::get_model_info(&model_name);

        let model_repo = Self::retrieve_model(
            model_name.clone(),
            cache_dir.clone(),
            show_download_progress,
        )?;

        let model_file_reference = model_repo
            .get(&model_info.model_file)
            .context(format!("Failed to retrieve {}", model_info.model_file))?;

        for file in &model_info.additional_files {
            model_repo
                .get(file)
                .context(format!("Failed to retrieve {}", file))?;
        }

        let session = Session::builder()?
            .with_execution_providers(execution_providers)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(model_file_reference)?;

        // Load tokenizer for documents (truncates at max_length - 1 to leave room for marker)
        let tokenizer = load_tokenizer_hf_hub(model_repo, max_length - 1)?;

        // Load query tokenizer with MASK padding
        let mut query_tokenizer = tokenizer.clone();

        // Get mask token id
        let mask_token_id = query_tokenizer
            .token_to_id(&model_info.mask_token)
            .ok_or_else(|| anyhow::anyhow!("Could not find mask token in vocabulary"))?;

        // Get pad token id
        let pad_token_id = tokenizer.get_padding().map(|p| p.pad_id).unwrap_or(0);

        // Enable padding for query tokenizer with MASK token
        // with_padding takes &mut self and returns &mut Self, mutating in place
        query_tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(model_info.min_query_length),
            pad_token: model_info.mask_token.clone(),
            pad_id: mask_token_id,
            ..Default::default()
        }));

        // Build skip list from punctuation
        let skip_list = Self::build_skip_list(&tokenizer);

        Ok(Self::new(
            tokenizer,
            query_tokenizer,
            session,
            model_info.query_marker_token_id,
            model_info.document_marker_token_id,
            mask_token_id,
            pad_token_id,
            skip_list,
            model_info.min_query_length,
            model_info.dim,
        ))
    }

    /// Create from user-defined model
    pub fn try_new_from_user_defined(
        model: UserDefinedLateInteractionModel,
        options: LateInteractionInitOptionsUserDefined,
    ) -> Result<Self> {
        let LateInteractionInitOptionsUserDefined {
            execution_providers,
            max_length,
        } = options;

        let threads = available_parallelism()?.get();

        let session = Session::builder()?
            .with_execution_providers(execution_providers)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_memory(&model.onnx_file)?;

        let tokenizer = load_tokenizer(model.tokenizer_files.clone(), max_length - 1)?;
        let mut query_tokenizer = load_tokenizer(model.tokenizer_files, max_length - 1)?;

        let mask_token_id = query_tokenizer
            .token_to_id(&model.mask_token)
            .ok_or_else(|| anyhow::anyhow!("Could not find mask token in vocabulary"))?;

        let pad_token_id = tokenizer.get_padding().map(|p| p.pad_id).unwrap_or(0);

        query_tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(model.min_query_length),
            pad_token: model.mask_token.clone(),
            pad_id: mask_token_id,
            ..Default::default()
        }));

        let skip_list = Self::build_skip_list(&tokenizer);

        Ok(Self::new(
            tokenizer,
            query_tokenizer,
            session,
            model.query_marker_token_id,
            model.document_marker_token_id,
            mask_token_id,
            pad_token_id,
            skip_list,
            model.min_query_length,
            model.dim,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        tokenizer: Tokenizer,
        query_tokenizer: Tokenizer,
        session: Session,
        query_marker_token_id: u32,
        document_marker_token_id: u32,
        mask_token_id: u32,
        pad_token_id: u32,
        skip_list: HashSet<u32>,
        min_query_length: usize,
        dim: usize,
    ) -> Self {
        let need_token_type_ids = session
            .inputs
            .iter()
            .any(|input| input.name == "token_type_ids");

        Self {
            tokenizer,
            query_tokenizer,
            session,
            need_token_type_ids,
            query_marker_token_id,
            document_marker_token_id,
            mask_token_id,
            pad_token_id,
            skip_list,
            min_query_length,
            dim,
        }
    }

    fn build_skip_list(tokenizer: &Tokenizer) -> HashSet<u32> {
        let mut skip_list = HashSet::new();
        for c in get_punctuation_chars() {
            if let Some(id) = tokenizer.token_to_id(&c.to_string()) {
                skip_list.insert(id);
            }
        }
        skip_list
    }

    #[cfg(feature = "hf-hub")]
    fn retrieve_model(
        model: LateInteractionModel,
        cache_dir: PathBuf,
        show_download_progress: bool,
    ) -> Result<ApiRepo> {
        use crate::common::pull_from_hf;
        pull_from_hf(model.to_string(), cache_dir, show_download_progress)
    }

    pub fn list_supported_models() -> Vec<LateInteractionModelInfo> {
        models_list()
    }

    pub fn get_model_info(model: &LateInteractionModel) -> LateInteractionModelInfo {
        Self::list_supported_models()
            .into_iter()
            .find(|m| &m.model == model)
            .expect("Model not found in supported models list")
    }

    /// Get the embedding dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Embed documents (passages)
    pub fn embed<S: AsRef<str> + Send + Sync>(
        &mut self,
        documents: impl AsRef<[S]>,
        batch_size: Option<usize>,
    ) -> Result<Vec<LateInteractionEmbedding>> {
        self.embed_internal(documents, batch_size, false)
    }

    /// Embed queries
    pub fn query_embed<S: AsRef<str> + Send + Sync>(
        &mut self,
        queries: impl AsRef<[S]>,
        batch_size: Option<usize>,
    ) -> Result<Vec<LateInteractionEmbedding>> {
        self.embed_internal(queries, batch_size, true)
    }

    fn embed_internal<S: AsRef<str> + Send + Sync>(
        &mut self,
        texts: impl AsRef<[S]>,
        batch_size: Option<usize>,
        is_query: bool,
    ) -> Result<Vec<LateInteractionEmbedding>> {
        let texts = texts.as_ref();
        let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE);

        let mut all_embeddings = Vec::with_capacity(texts.len());

        for batch in texts.chunks(batch_size) {
            let tokenizer = if is_query {
                &self.query_tokenizer
            } else {
                &self.tokenizer
            };

            let inputs: Vec<&str> = batch.iter().map(|t| t.as_ref()).collect();
            let encodings = tokenizer
                .encode_batch(inputs, true)
                .map_err(|e| anyhow::anyhow!("Failed to encode batch: {}", e))?;

            let encoding_length = encodings
                .first()
                .ok_or_else(|| anyhow::anyhow!("Empty encodings"))?
                .len();

            let batch_size_actual = batch.len();
            // +1 for the marker token we'll insert
            let final_length = encoding_length + 1;

            let mut ids_array = Vec::with_capacity(batch_size_actual * final_length);
            let mut mask_array = Vec::with_capacity(batch_size_actual * final_length);
            let mut type_ids_array = Vec::with_capacity(batch_size_actual * final_length);

            let marker_token = if is_query {
                self.query_marker_token_id
            } else {
                self.document_marker_token_id
            };

            for encoding in &encodings {
                let ids = encoding.get_ids();
                let mask = encoding.get_attention_mask();
                let type_ids = encoding.get_type_ids();

                // Insert marker token after first token (position 1)
                // [CLS, marker, token1, token2, ...]
                ids_array.push(ids[0] as i64);
                ids_array.push(marker_token as i64);
                ids_array.extend(ids[1..].iter().map(|&x| x as i64));

                mask_array.push(mask[0] as i64);
                mask_array.push(1i64); // Marker always attended
                mask_array.extend(mask[1..].iter().map(|&x| x as i64));

                type_ids_array.push(type_ids[0] as i64);
                type_ids_array.push(0i64);
                type_ids_array.extend(type_ids[1..].iter().map(|&x| x as i64));
            }

            let input_ids_array =
                Array::from_shape_vec((batch_size_actual, final_length), ids_array)?;
            let attention_mask_array =
                Array::from_shape_vec((batch_size_actual, final_length), mask_array)?;
            let token_type_ids_array =
                Array::from_shape_vec((batch_size_actual, final_length), type_ids_array)?;

            let mut session_inputs = ort::inputs![
                "input_ids" => Value::from_array(input_ids_array.clone())?,
                "attention_mask" => Value::from_array(attention_mask_array.clone())?,
            ];

            if self.need_token_type_ids {
                session_inputs.push((
                    "token_type_ids".into(),
                    Value::from_array(token_type_ids_array)?.into(),
                ));
            }

            let outputs = self.session.run(session_inputs)?;

            // Get the first output
            let (_, output_value) = outputs
                .iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("No output from model"))?;

            // Extract as ndarray ArrayView
            let (shape, data) = output_value.try_extract_tensor::<f32>()?;
            // Shape is (batch_size, seq_len, dim)
            let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            let seq_len = shape_vec[1];
            let dim = shape_vec[2];

            // Process each item in batch
            for batch_idx in 0..batch_size_actual {
                let mut embeddings: Vec<Vec<f32>> = Vec::new();

                if is_query {
                    // For queries: return ALL token embeddings (including MASK padding)
                    for seq_idx in 0..seq_len {
                        let start = batch_idx * seq_len * dim + seq_idx * dim;
                        let end = start + dim;
                        let token_embedding: Vec<f32> = data[start..end].to_vec();

                        // L2 normalize
                        let norm: f32 = token_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                        let norm = norm.max(1e-12);
                        let normalized: Vec<f32> =
                            token_embedding.iter().map(|x| x / norm).collect();

                        embeddings.push(normalized);
                    }
                } else {
                    // For documents: mask out punctuation and pad tokens, filter by attention mask
                    let mut attention_mask_vec: Vec<i64> = attention_mask_array
                        .row(batch_idx)
                        .iter()
                        .copied()
                        .collect();

                    let input_ids_row = input_ids_array.row(batch_idx);
                    for (j, &token_id) in input_ids_row.iter().enumerate() {
                        let token_id_u32 = token_id as u32;
                        if self.skip_list.contains(&token_id_u32)
                            || token_id_u32 == self.pad_token_id
                        {
                            attention_mask_vec[j] = 0;
                        }
                    }

                    for (seq_idx, &mask_val) in attention_mask_vec.iter().enumerate().take(seq_len) {
                        if mask_val == 1 {
                            let start = batch_idx * seq_len * dim + seq_idx * dim;
                            let end = start + dim;
                            let token_embedding: Vec<f32> = data[start..end].to_vec();

                            // L2 normalize
                            let norm: f32 =
                                token_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                            let norm = norm.max(1e-12);
                            let normalized: Vec<f32> =
                                token_embedding.iter().map(|x| x / norm).collect();

                            embeddings.push(normalized);
                        }
                    }
                }

                all_embeddings.push(embeddings);
            }
        }

        Ok(all_embeddings)
    }
}
