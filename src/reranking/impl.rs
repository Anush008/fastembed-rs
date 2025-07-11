#[cfg(feature = "hf-hub")]
use anyhow::Context;
use anyhow::Result;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};

#[cfg(feature = "hf-hub")]
use crate::common::load_tokenizer_hf_hub;
use crate::{
    common::load_tokenizer, models::reranking::reranker_model_list, RerankerModel,
    RerankerModelInfo,
};
#[cfg(feature = "hf-hub")]
use hf_hub::{api::sync::ApiBuilder, Cache};
use ndarray::{s, Array};
use tokenizers::Tokenizer;

#[cfg(feature = "hf-hub")]
use super::RerankInitOptions;
use super::{
    OnnxSource, RerankInitOptionsUserDefined, RerankResult, TextRerank, UserDefinedRerankingModel,
    DEFAULT_BATCH_SIZE,
};

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

    #[cfg(feature = "hf-hub")]
    pub fn try_new(options: RerankInitOptions) -> Result<TextRerank> {
        use super::RerankInitOptions;

        let RerankInitOptions {
            model_name,
            execution_providers,
            max_length,
            cache_dir,
            show_download_progress,
        } = options;

        let cache = Cache::new(cache_dir);
        let api = ApiBuilder::from_cache(cache)
            .with_progress(show_download_progress)
            .build()
            .expect("Failed to build API from cache");
        let model_repo = api.model(model_name.to_string());

        let model_file_name = TextRerank::get_model_info(&model_name).model_file;
        let model_file_reference = model_repo.get(&model_file_name).context(format!(
            "Failed to retrieve model file: {}",
            model_file_name
        ))?;
        let additional_files = TextRerank::get_model_info(&model_name).additional_files;
        for additional_file in additional_files {
            let _additional_file_reference = model_repo.get(&additional_file).context(format!(
                "Failed to retrieve additional file: {}",
                additional_file
            ))?;
        }

        // Create ONNX Runtime session with deterministic configuration
        // Use single thread execution for consistent/deterministic results
        let session = Session::builder()?
            .with_execution_providers(execution_providers)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?  // Use single thread for deterministic results
            .with_inter_threads(1)?  // Use single thread for inter-op parallelism
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

        // Create ONNX Runtime session with deterministic configuration
        // Use single thread execution for consistent/deterministic results
        let session = Session::builder()?
            .with_execution_providers(execution_providers)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?  // Use single thread for deterministic results
            .with_inter_threads(1)?;  // Use single thread for inter-op parallelism

        let session = match &model.onnx_source {
            OnnxSource::Memory(bytes) => session.commit_from_memory(bytes)?,
            OnnxSource::File(path) => session.commit_from_file(path)?,
        };

        let tokenizer = load_tokenizer(model.tokenizer_files, max_length)?;
        Ok(Self::new(tokenizer, session))
    }

    /// Rerank documents using the reranker model and returns the results sorted by score in descending order.
    pub fn rerank<S: AsRef<str> + Send + Sync>(
        &mut self,
        query: S,
        documents: Vec<S>,
        return_documents: bool,
        batch_size: Option<usize>,
    ) -> Result<Vec<RerankResult>> {
        let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE);
        let q = query.as_ref();

        let mut scores: Vec<f32> = Vec::with_capacity(documents.len());
        for batch in documents.chunks(batch_size) {
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
            let mut type_ids_array = Vec::with_capacity(max_size);

            encodings.iter().for_each(|encoding| {
                let ids = encoding.get_ids();
                let mask = encoding.get_attention_mask();
                let type_ids = encoding.get_type_ids();

                ids_array.extend(ids.iter().map(|x| *x as i64));
                mask_array.extend(mask.iter().map(|x| *x as i64));
                type_ids_array.extend(type_ids.iter().map(|x| *x as i64));
            });

            let inputs_ids_array = Array::from_shape_vec((batch_size, encoding_length), ids_array)?;
            let attention_mask_array =
                Array::from_shape_vec((batch_size, encoding_length), mask_array)?;
            let token_type_ids_array =
                Array::from_shape_vec((batch_size, encoding_length), type_ids_array)?;

            let mut session_inputs = ort::inputs![
                "input_ids" => Value::from_array(inputs_ids_array)?,
                "attention_mask" => Value::from_array(attention_mask_array)?,
            ];
            if self.need_token_type_ids {
                session_inputs.push((
                    "token_type_ids".into(),
                    Value::from_array(token_type_ids_array)?.into(),
                ));
            }

            let outputs = self.session.run(session_inputs)?;
            let outputs = outputs["logits"]
                .try_extract_array()
                .expect("Failed to extract logits tensor");
            let batch_scores: Vec<f32> = outputs
                .slice(s![.., 0])
                .rows()
                .into_iter()
                .flat_map(|row| row.to_vec())
                .collect();
            scores.extend(batch_scores);
        }

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
