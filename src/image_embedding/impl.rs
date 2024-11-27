#[cfg(feature = "online")]
use hf_hub::{
    api::sync::{ApiBuilder, ApiRepo},
    Cache,
};
use ndarray::{Array3, ArrayView3};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
#[cfg(feature = "online")]
use std::path::PathBuf;
use std::{path::Path, thread::available_parallelism};

use crate::{
    common::normalize, models::image_embedding::models_list, Embedding, ImageEmbeddingModel,
    ModelInfo,
};
use anyhow::anyhow;
#[cfg(feature = "online")]
use anyhow::Context;

#[cfg(feature = "online")]
use super::ImageInitOptions;
use super::{
    init::{ImageInitOptionsUserDefined, UserDefinedImageEmbeddingModel},
    utils::{Compose, Transform, TransformData},
    ImageEmbedding, DEFAULT_BATCH_SIZE,
};
use rayon::prelude::*;

impl ImageEmbedding {
    /// Try to generate a new ImageEmbedding Instance
    ///
    /// Uses the highest level of Graph optimization
    ///
    /// Uses the total number of CPUs available as the number of intra-threads
    #[cfg(feature = "online")]
    pub fn try_new(options: ImageInitOptions) -> anyhow::Result<Self> {
        let ImageInitOptions {
            model_name,
            execution_providers,
            cache_dir,
            show_download_progress,
        } = options;

        let threads = available_parallelism()?.get();

        let model_repo = ImageEmbedding::retrieve_model(
            model_name.clone(),
            cache_dir.clone(),
            show_download_progress,
        )?;

        let preprocessor_file = model_repo
            .get("preprocessor_config.json")
            .context("Failed to retrieve preprocessor_config.json")?;
        let preprocessor = Compose::from_file(preprocessor_file)?;

        let model_file_name = ImageEmbedding::get_model_info(&model_name).model_file;
        let model_file_reference = model_repo
            .get(&model_file_name)
            .context(format!("Failed to retrieve {}", model_file_name))?;

        let session = Session::builder()?
            .with_execution_providers(execution_providers)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(model_file_reference)?;

        Ok(Self::new(preprocessor, session))
    }

    /// Create a ImageEmbedding instance from model files provided by the user.
    ///
    /// This can be used for 'bring your own' embedding models
    pub fn try_new_from_user_defined(
        model: UserDefinedImageEmbeddingModel,
        options: ImageInitOptionsUserDefined,
    ) -> anyhow::Result<Self> {
        let ImageInitOptionsUserDefined {
            execution_providers,
        } = options;

        let threads = available_parallelism()?.get();

        let preprocessor = Compose::from_bytes(model.preprocessor_file)?;

        let session = Session::builder()?
            .with_execution_providers(execution_providers)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_memory(&model.onnx_file)?;

        Ok(Self::new(preprocessor, session))
    }

    /// Private method to return an instance
    fn new(preprocessor: Compose, session: Session) -> Self {
        Self {
            preprocessor,
            session,
        }
    }

    /// Return the ImageEmbedding model's directory from cache or remote retrieval
    #[cfg(feature = "online")]
    fn retrieve_model(
        model: ImageEmbeddingModel,
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
    pub fn list_supported_models() -> Vec<ModelInfo<ImageEmbeddingModel>> {
        models_list()
    }

    /// Get ModelInfo from ImageEmbeddingModel
    pub fn get_model_info(model: &ImageEmbeddingModel) -> ModelInfo<ImageEmbeddingModel> {
        ImageEmbedding::list_supported_models()
            .into_iter()
            .find(|m| &m.model == model)
            .expect("Model not found.")
    }

    /// Method to generate image embeddings for a Vec of image path
    // Generic type to accept String, &str, OsString, &OsStr
    pub fn embed<S: AsRef<Path> + Send + Sync>(
        &self,
        images: Vec<S>,
        batch_size: Option<usize>,
    ) -> anyhow::Result<Vec<Embedding>> {
        // Determine the batch size, default if not specified
        let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE);

        let output = images
            .par_chunks(batch_size)
            .map(|batch| {
                // Encode the texts in the batch
                let inputs = batch
                    .iter()
                    .map(|img| {
                        let img = image::ImageReader::open(img)?
                            .decode()
                            .map_err(|err| anyhow!("image decode: {}", err))?;
                        let pixels = self.preprocessor.transform(TransformData::Image(img))?;
                        match pixels {
                            TransformData::NdArray(array) => Ok(array),
                            _ => Err(anyhow!("Preprocessor configuration error!")),
                        }
                    })
                    .collect::<anyhow::Result<Vec<Array3<f32>>>>()?;

                // Extract the batch size
                let inputs_view: Vec<ArrayView3<f32>> =
                    inputs.iter().map(|img| img.view()).collect();
                let pixel_values_array = ndarray::stack(ndarray::Axis(0), &inputs_view)?;

                let input_name = self.session.inputs[0].name.clone();
                let session_inputs = ort::inputs![
                    input_name => Value::from_array(pixel_values_array)?,
                ]?;

                let outputs = self.session.run(session_inputs)?;

                // Try to get the only output key
                // If multiple, then default to few known keys `image_embeds` and `last_hidden_state`
                let last_hidden_state_key = match outputs.len() {
                    1 => vec![outputs.keys().next().unwrap()],
                    _ => vec!["image_embeds", "last_hidden_state"],
                };

                // Extract tensor and handle different dimensionalities
                let output_data = last_hidden_state_key
                    .iter()
                    .find_map(|&key| {
                        outputs
                            .get(key)
                            .and_then(|v| v.try_extract_tensor::<f32>().ok())
                    })
                    .ok_or_else(|| anyhow!("Could not extract tensor from any known output key"))?;
                let shape = output_data.shape();

                let embeddings: Vec<Vec<f32>> = match shape.len() {
                    3 => {
                        // For 3D output [batch_size, sequence_length, hidden_size]
                        // Take only the first token, sequence_length[0] (CLS token), embedding
                        // and return [batch_size, hidden_size]
                        (0..shape[0])
                            .map(|batch_idx| {
                                let cls_embedding =
                                    output_data.slice(ndarray::s![batch_idx, 0, ..]).to_vec();
                                normalize(&cls_embedding)
                            })
                            .collect()
                    }
                    2 => {
                        // For 2D output [batch_size, hidden_size]
                        output_data
                            .rows()
                            .into_iter()
                            .map(|row| normalize(row.as_slice().unwrap()))
                            .collect()
                    }
                    _ => return Err(anyhow!("Unexpected output tensor shape: {:?}", shape)),
                };

                Ok(embeddings)
            })
            .collect::<anyhow::Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();

        Ok(output)
    }
}
