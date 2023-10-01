use fastembed::EmbeddingBase;

fn main() {
    // Create a flattened ndarray

    let model = fastembed::FlagEmbedding::try_new(Default::default()).unwrap();
    let data = vec!["Hello is the world dead"];
    let embeddings = model.embed(data, None).unwrap();
    dbg!(embeddings[0][0]);
}
