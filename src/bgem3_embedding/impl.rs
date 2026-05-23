#[cfg(feature = "hf-hub")]
use crate::common::load_tokenizer_hf_hub;
use crate::{
    common::load_tokenizer,
    models::bgem3::{models_list, Bgem3Model},
    ModelInfo, SparseEmbedding, text_embedding::InitOptionsUserDefined,
};
#[cfg(feature = "hf-hub")]
use anyhow::Context;
use anyhow::Result;
#[cfg(feature = "hf-hub")]
use hf_hub::api::sync::ApiRepo;
use ndarray::Array;
use ort::{session::Session, value::Value};
use std::collections::HashMap;
#[cfg_attr(not(feature = "hf-hub"), allow(unused_imports))]
#[cfg(feature = "hf-hub")]
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[cfg_attr(not(feature = "hf-hub"), allow(unused_imports))]
use std::thread::available_parallelism;

#[cfg(feature = "hf-hub")]
use super::Bgem3InitOptions;
use super::{Bgem3Embedding, Bgem3EmbeddingOutput, DEFAULT_BATCH_SIZE, UserDefinedBgem3Model};

impl Bgem3Embedding {
    fn builder_error(err: ort::Error<ort::session::builder::SessionBuilder>) -> anyhow::Error {
        anyhow::Error::msg(err.to_string())
    }

    /// Try to generate a new Bgem3Embedding Instance
    ///
    /// Uses the highest level of Graph optimization
    ///
    /// Uses the total number of CPUs available as the number of intra-threads
    #[cfg(feature = "hf-hub")]
    pub fn try_new(options: Bgem3InitOptions) -> Result<Self> {
        use ort::session::builder::GraphOptimizationLevel;

        let Bgem3InitOptions {
            max_length,
            model_name,
            cache_dir,
            show_download_progress,
            execution_providers,
        } = options;

        let threads = available_parallelism()?.get();

        let model_repo = Bgem3Embedding::retrieve_model(
            model_name.clone(),
            cache_dir.clone(),
            show_download_progress,
        )?;

        let model_info = Bgem3Embedding::get_model_info(&model_name);
        let model_file_name = &model_info.model_file;
        let model_file_reference = model_repo
            .get(model_file_name)
            .context(format!("Failed to retrieve {} ", model_file_name))?;

        // Download additional files if needed
        if !model_info.additional_files.is_empty() {
            for file in &model_info.additional_files {
                model_repo
                    .get(file)
                    .context(format!("Failed to retrieve {}", file))?;
            }
        }

        let session = Session::builder()?
            .with_execution_providers(execution_providers)
            .map_err(Self::builder_error)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(Self::builder_error)?
            .with_intra_threads(threads)
            .map_err(Self::builder_error)?
            .commit_from_file(model_file_reference)?;

        let tokenizer = load_tokenizer_hf_hub(model_repo, max_length)?;
        Ok(Self::new(tokenizer, session, model_name))
    }

    /// Create a Bgem3Embedding instance from model files provided by the user.
    pub fn try_new_from_user_defined(
        model: UserDefinedBgem3Model,
        options: InitOptionsUserDefined,
    ) -> Result<Self> {
        use ort::session::builder::GraphOptimizationLevel;

        let InitOptionsUserDefined {
            execution_providers,
            max_length,
        } = options;

        let threads = available_parallelism()?.get();

        let session = Session::builder()?
            .with_execution_providers(execution_providers)
            .map_err(Self::builder_error)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(Self::builder_error)?
            .with_intra_threads(threads)
            .map_err(Self::builder_error)?
            .commit_from_memory(&model.onnx_file)?;

        let tokenizer = load_tokenizer(model.tokenizer_files, max_length)?;
        Ok(Self::new(tokenizer, session, Bgem3Model::default()))
    }

    /// Private method to return an instance
    fn new(tokenizer: Tokenizer, session: Session, model: Bgem3Model) -> Self {
        let need_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");
        Self {
            tokenizer,
            session,
            need_token_type_ids,
            model,
        }
    }

    /// Return the Bgem3Embedding model's directory from cache or remote retrieval
    #[cfg(feature = "hf-hub")]
    fn retrieve_model(
        model: Bgem3Model,
        cache_dir: PathBuf,
        show_download_progress: bool,
    ) -> Result<ApiRepo> {
        use crate::common::pull_from_hf;

        pull_from_hf(model.to_string(), cache_dir, show_download_progress)
    }

    /// Retrieve a list of supported models
    pub fn list_supported_models() -> Vec<ModelInfo<Bgem3Model>> {
        models_list()
    }

    /// Get ModelInfo from Bgem3Model
    pub fn get_model_info(model: &Bgem3Model) -> ModelInfo<Bgem3Model> {
        Bgem3Embedding::list_supported_models()
            .into_iter()
            .find(|m| &m.model == model)
            .expect("Model not found in supported models list. This is a bug - please report it.")
    }

    /// Method to generate sentence embeddings for a collection of texts.
    pub fn embed<S: AsRef<str> + Send + Sync>(
        &mut self,
        texts: impl AsRef<[S]>,
        batch_size: Option<usize>,
    ) -> Result<Bgem3EmbeddingOutput> {
        let texts = texts.as_ref();
        let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE);

        let mut all_dense = Vec::with_capacity(texts.len());
        let mut all_sparse = Vec::with_capacity(texts.len());
        let mut all_colbert = Vec::with_capacity(texts.len());

        for batch in texts.chunks(batch_size) {
            let inputs = batch.iter().map(|text| text.as_ref()).collect();
            let encodings = self.tokenizer.encode_batch(inputs, true).map_err(|e| {
                anyhow::Error::msg(e.to_string()).context("Failed to encode the batch.")
            })?;

            let encoding_length = encodings
                .first()
                .ok_or_else(|| anyhow::anyhow!("Tokenizer returned empty encodings"))?
                .len();
            let current_batch_size = batch.len();
            let max_size = encoding_length * current_batch_size;

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
                Array::from_shape_vec((current_batch_size, encoding_length), ids_array)?;
            let attention_mask_array =
                Array::from_shape_vec((current_batch_size, encoding_length), mask_array)?;
            let token_type_ids_array =
                Array::from_shape_vec((current_batch_size, encoding_length), type_ids_array)?;

            let mut session_inputs = ort::inputs![
                "input_ids" => Value::from_array(inputs_ids_array.clone())?,
                "attention_mask" => Value::from_array(attention_mask_array.clone())?,
            ];

            if self.need_token_type_ids {
                session_inputs.push((
                    "token_type_ids".into(),
                    Value::from_array(token_type_ids_array)?.into(),
                ));
            }

            let outputs = self.session.run(session_inputs)?;

            // gpahal/bge-m3-onnx-int8 returns outputs in this exact order:
            // outputs[0] -> dense_vecs: [batch_size, 1024]
            // outputs[1] -> sparse_vecs: [batch_size, seq_len, 1]
            // outputs[2] -> colbert_vecs: [batch_size, seq_len - 1, 1024]
            
            // Dense vecs
            let dense_output = &outputs[0];
            let (dense_shape, dense_data) = dense_output.try_extract_tensor::<f32>()?;
            let dense_shape: Vec<usize> = dense_shape.iter().map(|&d| d as usize).collect();
            let dense_view = ndarray::ArrayViewD::from_shape(dense_shape.as_slice(), dense_data)?;
            
            for row in dense_view.rows() {
                all_dense.push(row.to_vec());
            }

            // Sparse vecs
            let sparse_output = &outputs[1];
            let (sparse_shape, sparse_data) = sparse_output.try_extract_tensor::<f32>()?;
            let sparse_shape: Vec<usize> = sparse_shape.iter().map(|&d| d as usize).collect();
            let sparse_view = ndarray::ArrayViewD::from_shape(sparse_shape.as_slice(), sparse_data)?;
            
            // Special tokens to skip: XLM-RoBERTa: CLS=0, PAD=1, EOS=2, UNK=3
            const SPECIAL_TOKENS: [i64; 4] = [0, 1, 2, 3];

            for batch_idx in 0..current_batch_size {
                let mut token_weights: HashMap<usize, f32> = HashMap::new();
                for seq_idx in 0..encoding_length {
                    if attention_mask_array[[batch_idx, seq_idx]] == 0 {
                        continue;
                    }

                    let token_id = inputs_ids_array[[batch_idx, seq_idx]];
                    if SPECIAL_TOKENS.contains(&token_id) {
                        continue;
                    }

                    let weight = sparse_view[[batch_idx, seq_idx, 0]];
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

                all_sparse.push(SparseEmbedding { values, indices });
            }

            // ColBERT vecs
            let colbert_output = &outputs[2];
            let (colbert_shape, colbert_data) = colbert_output.try_extract_tensor::<f32>()?;
            let colbert_shape: Vec<usize> = colbert_shape.iter().map(|&d| d as usize).collect();
            let colbert_view = ndarray::ArrayViewD::from_shape(colbert_shape.as_slice(), colbert_data)?;

            // Shape of colbert_view is [batch_size, seq_len - 1, 1024]
            let colbert_seq_len = colbert_shape[1]; // seq_len - 1

            for batch_idx in 0..current_batch_size {
                let mut doc_colbert = Vec::new();
                for seq_idx in 0..colbert_seq_len {
                    if attention_mask_array[[batch_idx, seq_idx + 1]] == 1 {
                        let token_vector = colbert_view.slice(ndarray::s![batch_idx, seq_idx, ..]);
                        doc_colbert.push(token_vector.to_vec());
                    }
                }
                all_colbert.push(doc_colbert);
            }
        }

        Ok(Bgem3EmbeddingOutput {
            dense: all_dense,
            sparse: all_sparse,
            colbert: all_colbert,
        })
    }
}
