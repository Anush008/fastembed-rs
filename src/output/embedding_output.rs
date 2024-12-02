use ndarray::{Array2, ArrayView, Dim, IxDynImpl};
use ort::session::SessionOutputs;

use crate::pooling;

use super::{OutputKey, OutputPrecedence};

/// [`SingleBatchOutput`] contains the output of a single batch of inference.
///
/// In the future, each batch will need to deal with its own post-processing, such as
/// pooling etc. This struct should contain all the necessary information for the
/// post-processing to be performed.
pub struct SingleBatchOutput<'r, 's> {
    pub session_outputs: SessionOutputs<'r, 's>,
    pub attention_mask_array: Array2<i64>,
}

impl SingleBatchOutput<'_, '_> {
    /// Select the output from the session outputs based on the given precedence.
    ///
    /// This returns a view into the tensor, which can be used to perform further
    /// operations.
    pub fn select_output<'a>(
        &'a self,
        precedence: &impl OutputPrecedence,
    ) -> anyhow::Result<ArrayView<f32, Dim<IxDynImpl>>> {
        let ort_output: &ort::value::Value = precedence
            .key_precedence()
            .find_map(|key| match key {
                OutputKey::OnlyOne => self
                    .session_outputs
                    .get(self.session_outputs.keys().nth(0)?),
                OutputKey::ByOrder(idx) => {
                    let x = self
                        .session_outputs
                        .get(self.session_outputs.keys().nth(*idx)?);
                    x
                }
                OutputKey::ByName(name) => self.session_outputs.get(name),
            })
            .ok_or_else(|| {
                anyhow::Error::msg(format!(
                    "No suitable output found in the session outputs. Available outputs: {:?}",
                    self.session_outputs.keys().collect::<Vec<_>>()
                ))
            })?;

        ort_output
            .try_extract_tensor::<f32>()
            .map_err(anyhow::Error::new)
    }

    /// Select the output from the session outputs based on the given precedence and pool it.
    ///
    /// This function will pool the output based on the given pooling option, if any.
    pub fn select_and_pool_output(
        &self,
        precedence: &impl OutputPrecedence,
        pooling_opt: Option<pooling::Pooling>,
    ) -> anyhow::Result<Array2<f32>> {
        let tensor = self.select_output(precedence)?;

        // If there is none pooling, default to cls so as not to break the existing implementations
        // TODO: Consider return output as is to support custom model that has built-in pooling layer:
        // - [] Add model with built-in pooling to the list of supported model in ``models::text_embdding::models_list``
        // - [] Write unit test for new model
        // - [] Update ``pooling::Pooling`` to include None type
        // - [] Change the line below to return output as is
        // - [] Release major version because of breaking changes
        match pooling_opt.unwrap_or_default() {
            pooling::Pooling::Cls => pooling::cls(&tensor),
            pooling::Pooling::Mean => pooling::mean(&tensor, self.attention_mask_array.clone()),
        }
    }
}

/// Container struct with all the outputs from the embedding layer.
///
/// This will contain one [`SingleBatchOutput`] object per batch/inference call.
pub struct EmbeddingOutput<'r, 's> {
    batches: Vec<SingleBatchOutput<'r, 's>>,
}

impl<'r, 's> EmbeddingOutput<'r, 's> {
    /// Create a new [`EmbeddingOutput`] from a [`ort::SessionOutputs`] object.
    pub fn new(batches: impl IntoIterator<Item = SingleBatchOutput<'r, 's>>) -> Self {
        Self {
            batches: batches.into_iter().collect(),
        }
    }

    /// Consume this [`EmbeddingOutput`] and return the raw session outputs.
    ///
    /// This allows the user to perform their custom extractions outside of this
    /// library.
    pub fn into_raw(self) -> Vec<SingleBatchOutput<'r, 's>> {
        self.batches
    }

    /// Export the output using the given output transformer.
    ///
    /// The transformer shall be responsible for:
    /// - Selecting the output from the session outputs based on the precedence order,
    /// - Extracting the tensor from the output, then
    /// - Transform the tensor into the desired output.
    ///
    /// The transformer function should take a slice of [`SingleBatchOutput`], and return
    /// the desired output type.
    ///
    /// If any of the steps fail, this function will return an error, including
    /// the session output not containing the expected precedence keys.
    pub fn export_with_transformer<R>(
        &self,
        // TODO: Convert this to a trait alias when it's stabilized.
        // https://github.com/rust-lang/rust/issues/41517
        transformer: impl Fn(&[SingleBatchOutput]) -> anyhow::Result<R>,
    ) -> anyhow::Result<R> {
        transformer(&self.batches)
    }
}
