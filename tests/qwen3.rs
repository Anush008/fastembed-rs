#![cfg(feature = "hf-hub")]

use candle_core::{DType, Device};
use fastembed::Qwen3TextEmbedding;

const REPO_06B: &str = "Qwen/Qwen3-Embedding-0.6B";
const REPO_4B: &str = "Qwen/Qwen3-Embedding-4B";
const REPO_8B: &str = "Qwen/Qwen3-Embedding-8B";
const MAX_LENGTH: usize = 512;

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn run_embed_test(repo_id: &str) {
    let device = Device::Cpu;
    let model =
        Qwen3TextEmbedding::from_hf(repo_id, &device, DType::F32, MAX_LENGTH).expect("load model");

    // Two queries and two documents (similar to official Qwen3 example)
    let queries = vec![
        "What is the capital of China?",
        "Explain gravity",
    ];
    let documents = vec![
        "Beijing is the capital of China.",
        "Gravity is a force that attracts objects toward each other.",
    ];

    let all_texts: Vec<&str> = queries.iter().chain(documents.iter()).copied().collect();
    let embeddings = model.embed(&all_texts).expect("embed");

    assert_eq!(embeddings.len(), all_texts.len());
    for (i, emb) in embeddings.iter().enumerate() {
        assert_eq!(emb.len(), model.config().hidden_size);

        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "embedding should be L2-normalized, got norm {}",
            norm
        );

        println!("[{}] text: {:?}", i, all_texts[i]);
        println!("    embedding (first 8): {:?}", &emb[..8.min(emb.len())]);
    }

    // Compute similarity scores
    let q0_d0 = cosine_sim(&embeddings[0], &embeddings[2]);
    let q0_d1 = cosine_sim(&embeddings[0], &embeddings[3]);
    let q1_d0 = cosine_sim(&embeddings[1], &embeddings[2]);
    let q1_d1 = cosine_sim(&embeddings[1], &embeddings[3]);

    println!("\nSimilarity matrix (queries x documents):");
    println!("         doc0     doc1");
    println!("query0   {:.4}   {:.4}", q0_d0, q0_d1);
    println!("query1   {:.4}   {:.4}", q1_d0, q1_d1);

    // Semantic check: query0 should be more similar to doc0, query1 to doc1
    assert!(
        q0_d0 > q0_d1,
        "query0 should be more similar to doc0 than doc1: {} vs {}",
        q0_d0,
        q0_d1
    );
    assert!(
        q1_d1 > q1_d0,
        "query1 should be more similar to doc1 than doc0: {} vs {}",
        q1_d1,
        q1_d0
    );
}

#[test]
fn qwen3_06b_embed() {
    run_embed_test(REPO_06B);
}

/// Test embedding consistency: single vs batch should give same results
#[test]
fn qwen3_06b_single_vs_batch() {
    let device = Device::Cpu;
    let model =
        Qwen3TextEmbedding::from_hf(REPO_06B, &device, DType::F32, 8192).expect("load model");

    let texts = vec!["Hello world", "This is a test", "Another sentence here"];

    // Embed as batch
    let batch_embeddings = model.embed(&texts).expect("batch embed");

    // Embed individually
    let single_embeddings: Vec<Vec<f32>> = texts
        .iter()
        .map(|t| model.embed(&[*t]).expect("single embed")[0].clone())
        .collect();

    // Compare
    for (i, (batch, single)) in batch_embeddings.iter().zip(single_embeddings.iter()).enumerate() {
        let diff: f32 = batch.iter().zip(single.iter()).map(|(a, b)| (a - b).abs()).sum();
        println!("Text {}: batch vs single diff = {}", i, diff);
        assert!(
            diff < 0.01,
            "Text {} embeddings differ: batch vs single diff = {}",
            i,
            diff
        );
    }
}

/// Test with the exact inputs from the official Qwen3-Embedding model card.
/// Expected scores: [[0.7646, 0.1414], [0.1355, 0.6000]]
#[test]
fn qwen3_06b_reference_scores() {
    let device = Device::Cpu;
    let model =
        Qwen3TextEmbedding::from_hf(REPO_06B, &device, DType::F32, 8192).expect("load model");

    // Exact inputs from model card (with Instruct prefix for queries)
    let task = "Given a web search query, retrieve relevant passages that answer the query";
    let queries: Vec<String> = vec![
        format!("Instruct: {}\nQuery:{}", task, "What is the capital of China?"),
        format!("Instruct: {}\nQuery:{}", task, "Explain gravity"),
    ];
    let documents = vec![
        "The capital of China is Beijing.".to_string(),
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.".to_string(),
    ];

    let input_texts: Vec<&str> = queries
        .iter()
        .map(|s| s.as_str())
        .chain(documents.iter().map(|s| s.as_str()))
        .collect();
    let embeddings = model.embed(&input_texts).expect("embed");

    let q0_d0 = cosine_sim(&embeddings[0], &embeddings[2]);
    let q0_d1 = cosine_sim(&embeddings[0], &embeddings[3]);
    let q1_d0 = cosine_sim(&embeddings[1], &embeddings[2]);
    let q1_d1 = cosine_sim(&embeddings[1], &embeddings[3]);

    println!("Reference test similarity matrix:");
    println!("         doc0     doc1");
    println!("query0   {:.4}   {:.4}", q0_d0, q0_d1);
    println!("query1   {:.4}   {:.4}", q1_d0, q1_d1);

    // Reference: [[0.7645568251609802, 0.14142508804798126], [0.13549736142158508, 0.5999549627304077]]
    let expected = [[0.7646f32, 0.1414f32], [0.1355f32, 0.6000f32]];
    let got = [[q0_d0, q0_d1], [q1_d0, q1_d1]];

    // Allow 5% tolerance for numerical differences (f32 vs f16, different frameworks)
    for i in 0..2 {
        for j in 0..2 {
            let diff = (got[i][j] - expected[i][j]).abs();
            assert!(
                diff < 0.05,
                "score[{}][{}] mismatch: got {:.4}, expected {:.4}, diff {:.4}",
                i, j, got[i][j], expected[i][j], diff
            );
        }
    }
}

#[test]
fn qwen3_4b_embed() {
    if std::env::var("RUN_QWEN3_4B").is_err() {
        return;
    }
    run_embed_test(REPO_4B);
}

#[test]
fn qwen3_8b_embed() {
    if std::env::var("RUN_QWEN3_8B").is_err() {
        return;
    }
    run_embed_test(REPO_8B);
}
