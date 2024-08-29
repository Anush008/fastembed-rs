use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = InitOptions::new(EmbeddingModel::BGEM3)
        .with_show_download_progress(true);

    let model = TextEmbedding::try_new(options)?;

    let documents = vec![
        "fastembed-rs is licensed under Apache  2.0",
        "embeddings are vectors",
        "rust is a systems programming language",
        "python is a high-level programming language",
        "javascript is a scripting language",
        "php is a server-side scripting language",
        "c# is a programming language",
        "c++ is a programming language",
        "c is a programming language",
        "c++ is a programming language",
        "c is a programming language",
    ];

    let start = Instant::now();
    let embeddings = model.embed(documents, None)?;
    let duration = start.elapsed();

    println!("Inference time: {:?}", duration);
    println!("Embeddings length: {}", embeddings.len());
    println!("Embedding dimension: {}", embeddings[0].len());
    println!("Embeddings: {:?}", embeddings[0]);

    Ok(())
}
