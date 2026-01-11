// examples/muvera_demo.rs
use anyhow::Result;
use fastembed::{
    LateInteractionInitOptions, LateInteractionModel, LateInteractionTextEmbedding, Muvera,
};

/// Compute ColBERT MaxSim score between query and document embeddings
fn maxsim(query_emb: &[Vec<f32>], doc_emb: &[Vec<f32>]) -> f32 {
    // For each query token, find max similarity with any document token
    query_emb
        .iter()
        .map(|q| {
            doc_emb
                .iter()
                .map(|d| q.iter().zip(d.iter()).map(|(a, b)| a * b).sum::<f32>())
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

fn main() -> Result<()> {
    // 1. Initialize the ColBERT model
    println!("Loading ColBERT model...");
    let mut model = LateInteractionTextEmbedding::try_new(LateInteractionInitOptions::new(
        LateInteractionModel::ColBERTV2,
    ))?;

    // 2. Create MUVERA postprocessor from the model
    let muvera = Muvera::from_late_interaction_model(
        &model,
        Some(5),  // k_sim: 2^5 = 32 clusters
        Some(16), // dim_proj: project to 16 dimensions per cluster
        Some(20), // r_reps: 20 repetitions for robustness
        Some(42), // random_seed
    )?;

    // 3. Define documents and queries
    let documents = vec![
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
    ];
    let queries = vec!["What is machine learning?"];

    // 4. Get multi-vector embeddings
    let doc_embeddings = model.embed(&documents, None)?;
    let query_embeddings = model.query_embed(&queries, None)?;

    println!("Document 0 shape: ({}, {})", doc_embeddings[0].len(), doc_embeddings[0][0].len());
    println!("Query 0 shape: ({}, {})", query_embeddings[0].len(), query_embeddings[0][0].len());
    println!("FDE size: {}", muvera.embedding_size());

    // 5. Convert to Fixed Dimensional Encodings (FDEs)
    let doc_fdes: Vec<Vec<f32>> = doc_embeddings
        .iter()
        .map(|emb| muvera.process_document(emb))
        .collect();
    let query_fde = muvera.process_query(&query_embeddings[0]);

    println!("Doc FDE shape: ({},)", doc_fdes[0].len());

    // 6. Compute MUVERA similarities (for candidate retrieval)
    for (i, doc_fde) in doc_fdes.iter().enumerate() {
        let similarity: f32 = query_fde.iter().zip(doc_fde.iter()).map(|(a, b)| a * b).sum();
        println!("Query-Doc{} similarity (MUVERA): {:.4}", i, similarity);
    }

    // 7. Compute ColBERT MaxSim scores (for reranking)
    for (i, doc_emb) in doc_embeddings.iter().enumerate() {
        let score = maxsim(&query_embeddings[0], doc_emb);
        println!("Query-Doc{} MaxSim score: {:.4}", i, score);
    }

    Ok(())
}