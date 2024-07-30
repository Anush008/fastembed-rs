use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gtebaseenv15_embed() {
        let model = TextEmbedding::try_new(Default::default()).expect("Failed to initialize model");
        let documents = vec![
            "passage: Hello, World!",
            "query: Hello, World!",
            "passage: This is an example passage.",
            "fastembed-rs is licensed under Apache 2.0",
        ];

        let embeddings = model
            .embed(documents, None)
            .expect("Failed to generate embeddings");

        assert_eq!(embeddings.len(), 4, "Expected 4 embeddings");
        assert_eq!(
            embeddings[0].len(),
            384,
            "Expected embedding dimension of 384"
        );
    }

    #[test]
    fn test_gtebaseenv15q_embed() {
        let model = TextEmbedding::try_new(InitOptions {
            model_name: EmbeddingModel::GTEBaseENV15Q,
            ..Default::default()
        })
        .expect("Failed to initialize model");
        let documents = vec![
            "passage: Hello, World!",
            "query: Hello, World!",
            "passage: This is an example passage.",
            "fastembed-rs is licensed under Apache 2.0",
        ];

        let embeddings = model
            .embed(documents, None)
            .expect("Failed to generate embeddings");

        assert_eq!(embeddings.len(), 4, "Expected 4 embeddings");
        assert_eq!(
            embeddings[0].len(),
            384,
            "Expected embedding dimension of 384"
        );
    }

    #[test]
    fn test_gtebaseenv15_embed_with_batch_size() {
        let model = TextEmbedding::try_new(Default::default()).expect("Failed to initialize model");
        let documents = vec![
            "passage: Hello, World!",
            "query: Hello, World!",
            "passage: This is an example passage.",
            "fastembed-rs is licensed under Apache 2.0",
        ];

        let embeddings = model
            .embed(documents, Some(2))
            .expect("Failed to generate embeddings");

        assert_eq!(embeddings.len(), 4, "Expected 4 embeddings");
        assert_eq!(
            embeddings[0].len(),
            384,
            "Expected embedding dimension of 384"
        );
    }
}
