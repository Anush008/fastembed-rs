//! Output types and functions for the [`TextEmbedding`] model.
//!
use crate::{
    common::{normalize, Embedding},
    output::{OutputKey, OutputPrecedence, SingleBatchOutput},
    pooling::Pooling,
};

#[cfg(doc)]
use super::TextEmbedding;

/// The default output precedence for the TextEmbedding model.
pub const OUTPUT_TYPE_PRECEDENCE: &[OutputKey] = &[
    OutputKey::OnlyOne,
    OutputKey::ByName("text_embeds"),
    OutputKey::ByName("last_hidden_state"),
    OutputKey::ByName("sentence_embedding"),
    // Better not to expose this unless the user explicitly asks for it.
    // OutputKey::ByName("token_embeddings"),
];

/// Generates thea default array transformer for the [`TextEmbedding`] model using the
/// provided output precedence.
///
// TODO (denwong47): now that pooling is done in SingleBatchOutput, it is possible that
// all the models will use this same generic transformer. Move this into SingleBatchOutput?
#[allow(unused_variables)]
pub fn transformer_with_precedence(
    output_precedence: impl OutputPrecedence,
    pooling: Option<Pooling>,
) -> impl Fn(&[SingleBatchOutput]) -> anyhow::Result<Vec<Embedding>> {
    move |batches| {
        // Not using `par_iter` here: the operations here is probably not
        // computationally expensive enough to warrant spinning up costs of the threads.
        batches
            .iter()
            .map(|batch| {
                batch
                    .select_and_pool_output(&output_precedence, pooling.clone())
                    .map(|array| {
                        array
                            .rows()
                            .into_iter()
                            .map(|row| normalize(row.as_slice().unwrap()))
                            .collect::<Vec<Embedding>>()
                    })
            })
            .try_fold(Vec::new(), |mut acc, res| {
                acc.extend(res?);
                Ok(acc)
            })
    }
}
