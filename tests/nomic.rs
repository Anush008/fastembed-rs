use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

#[test]
fn nomic_matryoshka() {
    let model = TextEmbedding::try_new(InitOptions {
        model_name: EmbeddingModel::NomicEmbedTextV15,
        ..Default::default()
    })
    .unwrap();
}
