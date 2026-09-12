#[cfg(feature = "hf-hub")]
use crate::common::{init_session_builder, load_tokenizer_hf_hub};
#[cfg(feature = "hf-hub")]
use crate::models::sparse::IDF_FILE;
use crate::{
    common::{Error, Result},
    models::sparse::{models_list, SparseModel},
    ModelInfo, SparseEmbedding,
};
#[cfg(feature = "hf-hub")]
use hf_hub::api::sync::ApiRepo;
use ndarray::{Array, ArrayViewD, Axis, CowArray, Dim};
use ort::{session::Session, value::Value};
use std::collections::{HashMap, HashSet};
#[cfg_attr(not(feature = "hf-hub"), allow(unused_imports))]
#[cfg(feature = "hf-hub")]
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

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

        let SparseInitOptions {
            max_length,
            model_name,
            cache_dir,
            show_download_progress,
            execution_providers,
            intra_threads,
        } = options;

        let model_repo = SparseTextEmbedding::retrieve_model(
            model_name.clone(),
            cache_dir.clone(),
            show_download_progress,
        )?;

        let model_info = SparseTextEmbedding::get_model_info(&model_name);
        let model_file_name = &model_info.model_file;
        let model_file_reference =
            model_repo
                .get(model_file_name)
                .map_err(|e| Error::ModelRetrieval {
                    file: model_file_name.clone(),
                    source: Box::new(e),
                })?;

        // Download additional files if needed (e.g., model.onnx.data for large models)
        let mut idf_file_reference: Option<PathBuf> = None;
        if !model_info.additional_files.is_empty() {
            for file in &model_info.additional_files {
                let reference = model_repo.get(file).map_err(|e| Error::ModelRetrieval {
                    file: file.clone(),
                    source: Box::new(e),
                })?;
                if file == IDF_FILE {
                    idf_file_reference = Some(reference);
                }
            }
        }

        let session = init_session_builder(execution_providers, intra_threads)?
            .commit_from_file(model_file_reference)?;

        let tokenizer = load_tokenizer_hf_hub(model_repo, max_length)?;
        // Models declaring an `idf.json` embed queries from a lookup table instead of
        // running inference, so the table is loaded up front alongside the tokenizer.
        let token_id_to_idf = idf_file_reference
            .map(|reference| Self::load_idf(&reference, &tokenizer))
            .transpose()?;

        Ok(Self::new(tokenizer, session, model_name, token_id_to_idf))
    }

    /// Private method to return an instance
    #[cfg_attr(not(feature = "hf-hub"), allow(dead_code))]
    fn new(
        tokenizer: Tokenizer,
        session: Session,
        model: SparseModel,
        token_id_to_idf: Option<HashMap<usize, f32>>,
    ) -> Self {
        let need_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");
        let special_token_ids = tokenizer
            .get_added_tokens_decoder()
            .iter()
            .filter(|(_, token)| token.special)
            .map(|(id, _)| *id as usize)
            .collect();
        Self {
            tokenizer,
            session,
            need_token_type_ids,
            model,
            special_token_ids,
            token_id_to_idf,
        }
    }

    /// Read the `idf.json` sidecar, resolving its token strings to token ids via the
    /// tokenizer's vocabulary. Tokens the tokenizer does not know about are dropped.
    #[cfg(feature = "hf-hub")]
    fn load_idf(idf_file: &Path, tokenizer: &Tokenizer) -> Result<HashMap<usize, f32>> {
        let token_to_idf: HashMap<String, f32> = serde_json::from_slice(&std::fs::read(idf_file)?)
            .map_err(|e| {
                Error::Other(format!("Failed to parse the {IDF_FILE} of the model: {e}"))
            })?;

        let vocab = tokenizer.get_vocab(true);
        Ok(token_to_idf
            .into_iter()
            .filter_map(|(token, idf)| vocab.get(&token).map(|&id| (id as usize, idf)))
            .collect())
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
            .expect("Model not found in supported models list. This is a bug - please report it.")
    }

    /// Method to generate sentence embeddings for a collection of texts.
    ///
    /// Accepts anything that can be referenced as a slice of elements implementing
    /// [`AsRef<str>`], such as `Vec<String>`, `Vec<&str>`, `&[String]`, or `&[&str]`.
    pub fn embed<S: AsRef<str> + Send + Sync>(
        &mut self,
        texts: impl AsRef<[S]>,
        batch_size: Option<usize>,
    ) -> Result<Vec<SparseEmbedding>> {
        let texts = texts.as_ref();
        // Determine the batch size, default if not specified
        let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE);
        if batch_size == 0 {
            return Err(Error::InvalidArgument(
                "batch_size must be greater than 0".into(),
            ));
        }

        let output = texts
            .chunks(batch_size)
            .map(|batch| {
                // Encode the texts in the batch
                let inputs = batch.iter().map(|text| text.as_ref()).collect();
                let encodings = self
                    .tokenizer
                    .encode_batch(inputs, true)
                    .map_err(|e| Error::Tokenization(format!("Failed to encode the batch: {e}")))?;

                // Extract the encoding length and batch size
                let encoding_length = encodings.first().ok_or(Error::EmptyTokenizations)?.len();
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
                    Array::from_shape_vec((batch_size, encoding_length), ids_array)
                        .map_err(|e| Error::InvalidShape(e.to_string()))?;
                let attention_mask_array =
                    Array::from_shape_vec((batch_size, encoding_length), mask_array)
                        .map_err(|e| Error::InvalidShape(e.to_string()))?;

                let token_type_ids_array =
                    Array::from_shape_vec((batch_size, encoding_length), type_ids_array)
                        .map_err(|e| Error::InvalidShape(e.to_string()))?;

                let mut session_inputs = ort::inputs![
                    "input_ids" => Value::from_array(inputs_ids_array.clone())
                        .map_err(|e| Error::OrtSession(e.to_string()))?,
                    "attention_mask" => Value::from_array(attention_mask_array.clone())
                        .map_err(|e| Error::OrtSession(e.to_string()))?,
                ];

                if self.need_token_type_ids {
                    session_inputs.push((
                        "token_type_ids".into(),
                        Value::from_array(token_type_ids_array)
                            .map_err(|e| Error::OrtSession(e.to_string()))?
                            .into(),
                    ));
                }

                let outputs = self
                    .session
                    .run(session_inputs)
                    .map_err(|e| Error::OrtSession(e.to_string()))?;

                let embeddings = match self.model {
                    SparseModel::SPLADEPPV1 => {
                        let last_hidden_state_key = match outputs.len() {
                            1 => outputs
                                .keys()
                                .next()
                                .ok_or_else(|| Error::OutputKeyMissing {
                                    key: "<only output>".into(),
                                })?,
                            _ => "last_hidden_state",
                        };

                        let (shape, data) = outputs[last_hidden_state_key]
                            .try_extract_tensor::<f32>()
                            .map_err(|e| Error::TensorExtraction(e.to_string()))?;
                        let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                        let output_array = ndarray::ArrayViewD::from_shape(shape.as_slice(), data)
                            .map_err(|e| Error::InvalidShape(e.to_string()))?;
                        let attention_mask_cow = ndarray::CowArray::from(&attention_mask_array);

                        Self::post_process_splade(&output_array, &attention_mask_cow)
                    }
                    SparseModel::BGEM3 => {
                        let output_key =
                            outputs
                                .keys()
                                .next()
                                .ok_or_else(|| Error::OutputKeyMissing {
                                    key: "<first output>".into(),
                                })?;

                        let (shape, data) = outputs[output_key]
                            .try_extract_tensor::<f32>()
                            .map_err(|e| Error::TensorExtraction(e.to_string()))?;
                        let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                        let hidden_states = ndarray::ArrayViewD::from_shape(shape.as_slice(), data)
                            .map_err(|e| Error::InvalidShape(e.to_string()))?;

                        Self::post_process_bgem3(
                            &hidden_states,
                            &inputs_ids_array,
                            &attention_mask_array,
                        )
                    }
                    SparseModel::OpenSearchNeuralSparseDocV3Gte => {
                        let logits_key = match outputs.len() {
                            1 => outputs
                                .keys()
                                .next()
                                .ok_or_else(|| Error::OutputKeyMissing {
                                    key: "<only output>".into(),
                                })?,
                            _ => "logits",
                        };

                        let (shape, data) = outputs[logits_key]
                            .try_extract_tensor::<f32>()
                            .map_err(|e| Error::TensorExtraction(e.to_string()))?;
                        let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                        let logits = ndarray::ArrayViewD::from_shape(shape.as_slice(), data)
                            .map_err(|e| Error::InvalidShape(e.to_string()))?;

                        Self::post_process_if_splade(
                            &logits,
                            &attention_mask_array,
                            &self.special_token_ids,
                        )
                    }
                };

                Ok(embeddings)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();

        Ok(output)
    }

    /// Method to generate sparse query embeddings without running any model inference.
    ///
    /// Only available for the inference-free (asymmetric) models, such as
    /// [`SparseModel::OpenSearchNeuralSparseDocV3Gte`].
    ///
    /// Accepts anything that can be referenced as a slice of elements implementing
    /// [`AsRef<str>`], such as `Vec<String>`, `Vec<&str>`, `&[String]`, or `&[&str]`.
    pub fn query_embed<S: AsRef<str> + Send + Sync>(
        &self,
        texts: impl AsRef<[S]>,
    ) -> Result<Vec<SparseEmbedding>> {
        let token_id_to_idf = self.token_id_to_idf.as_ref().ok_or_else(|| {
            Error::InvalidArgument(format!(
                "{} has no IDF table and no separate query representation, use `embed` instead",
                self.model
            ))
        })?;

        texts
            .as_ref()
            .iter()
            .map(|text| {
                let encoding = self
                    .tokenizer
                    .encode(text.as_ref(), true)
                    .map_err(|e| Error::Tokenization(format!("Failed to encode the query: {e}")))?;

                // Every unique token contributes its IDF weight exactly once, ordered by token id
                let mut token_ids: Vec<usize> = encoding
                    .get_ids()
                    .iter()
                    .map(|&id| id as usize)
                    .filter(|id| !self.special_token_ids.contains(id))
                    .collect();
                token_ids.sort_unstable();
                token_ids.dedup();

                let mut indices = Vec::with_capacity(token_ids.len());
                let mut values = Vec::with_capacity(token_ids.len());
                for token_id in token_ids {
                    if let Some(&idf) = token_id_to_idf.get(&token_id) {
                        indices.push(token_id);
                        values.push(idf);
                    }
                }

                Ok(SparseEmbedding { values, indices })
            })
            .collect()
    }

    fn post_process_splade(
        model_output: &ArrayViewD<f32>,
        attention_mask: &CowArray<i64, Dim<[usize; 2]>>,
    ) -> Vec<SparseEmbedding> {
        let relu_log = model_output.mapv(|x| (1.0 + x.max(0.0)).ln());

        let attention_mask = attention_mask.mapv(|x| x as f32).insert_axis(Axis(2));

        let weighted_log = relu_log * attention_mask;

        let scores = weighted_log.fold_axis(Axis(1), f32::NEG_INFINITY, |r, &v| r.max(v));

        scores
            .rows()
            .into_iter()
            .map(|row_scores| {
                let mut values: Vec<f32> = Vec::with_capacity(row_scores.len());
                let mut indices: Vec<usize> = Vec::with_capacity(row_scores.len());

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

    fn post_process_bgem3(
        hidden_states: &ArrayViewD<f32>,
        input_ids: &Array<i64, Dim<[usize; 2]>>,
        attention_mask: &Array<i64, Dim<[usize; 2]>>,
    ) -> Vec<SparseEmbedding> {
        use ndarray::ArrayView1;

        // Special tokens to skip (XLM-RoBERTa: CLS=0, PAD=1, EOS=2, UNK=3)
        const SPECIAL_TOKENS: [i64; 4] = [0, 1, 2, 3];

        let sparse_weights = super::bgem3_weights::get_weights();
        let weights = ArrayView1::from(&sparse_weights.weight[..]);
        let bias = sparse_weights.bias;
        let batch_size = input_ids.shape()[0];
        let seq_len = input_ids.shape()[1];

        (0..batch_size)
            .map(|batch_idx| {
                let mut token_weights: HashMap<usize, f32> = HashMap::new();

                for seq_idx in 0..seq_len {
                    if attention_mask[[batch_idx, seq_idx]] == 0 {
                        continue;
                    }

                    let token_id = input_ids[[batch_idx, seq_idx]];
                    if SPECIAL_TOKENS.contains(&token_id) {
                        continue;
                    }

                    let hidden = hidden_states.slice(ndarray::s![batch_idx, seq_idx, ..]);
                    let weight = (hidden.dot(&weights) + bias).max(0.0);

                    if weight > 0.0 {
                        token_weights
                            .entry(token_id as usize)
                            .and_modify(|w| *w = w.max(weight))
                            .or_insert(weight);
                    }
                }

                let mut indices: Vec<_> = token_weights.keys().copied().collect();
                indices.sort_unstable();
                let values: Vec<_> = indices.iter().map(|i| token_weights[i]).collect();

                SparseEmbedding { values, indices }
            })
            .collect()
    }

    /// Post-processing for the inference-free SPLADE document encoder.
    ///
    /// The token logits are max-pooled over the unmasked positions and squashed with a double
    /// log activation, `log(1 + log(1 + relu(x)))`, which the v3 models of the
    /// opensearch-neural-sparse family use to make document embeddings sparser than the single
    /// `log(1 + relu(x))` of SPLADE++.
    fn post_process_if_splade(
        logits: &ArrayViewD<f32>,
        attention_mask: &Array<i64, Dim<[usize; 2]>>,
        special_token_ids: &HashSet<usize>,
    ) -> Vec<SparseEmbedding> {
        let batch_size = attention_mask.shape()[0];
        let seq_len = attention_mask.shape()[1];
        let vocab_size = logits.shape()[2];

        (0..batch_size)
            .map(|batch_idx| {
                // Starting the accumulator at `0.0` encodes both the ReLU floor and the zero
                // contribution of the padded positions, which are skipped altogether.
                let mut pooled = vec![0.0f32; vocab_size];

                for seq_idx in 0..seq_len {
                    if attention_mask[[batch_idx, seq_idx]] == 0 {
                        continue;
                    }

                    let token_logits = logits.slice(ndarray::s![batch_idx, seq_idx, ..]);
                    for (score, &logit) in pooled.iter_mut().zip(token_logits.iter()) {
                        *score = score.max(logit);
                    }
                }

                let mut values: Vec<f32> = Vec::new();
                let mut indices: Vec<usize> = Vec::new();

                for (token_id, &score) in pooled.iter().enumerate() {
                    // Special tokens are dropped from the document side as well, otherwise they
                    // would match every query
                    if score <= 0.0 || special_token_ids.contains(&token_id) {
                        continue;
                    }
                    values.push((1.0 + (1.0 + score).ln()).ln());
                    indices.push(token_id);
                }

                SparseEmbedding { values, indices }
            })
            .collect()
    }
}
