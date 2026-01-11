#![cfg(feature = "hf-hub")]
#![cfg(feature = "qwen3")]

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

    let queries = ["What is the capital of China?", "Explain gravity"];
    let documents = [
        "Beijing is the capital of China.",
        "Gravity is a force that attracts objects toward each other.",
    ];

    let all_texts: Vec<&str> = queries.iter().chain(documents.iter()).copied().collect();
    let embeddings = model.embed(&all_texts).expect("embed");

    assert_eq!(embeddings.len(), all_texts.len());
    for emb in &embeddings {
        assert_eq!(emb.len(), model.config().hidden_size);
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "expected L2-normalized, got {norm}"
        );
    }

    let q0_d0 = cosine_sim(&embeddings[0], &embeddings[2]);
    let q0_d1 = cosine_sim(&embeddings[0], &embeddings[3]);
    let q1_d0 = cosine_sim(&embeddings[1], &embeddings[2]);
    let q1_d1 = cosine_sim(&embeddings[1], &embeddings[3]);

    assert!(
        q0_d0 > q0_d1,
        "query0 should match doc0 better: {q0_d0} vs {q0_d1}"
    );
    assert!(
        q1_d1 > q1_d0,
        "query1 should match doc1 better: {q1_d1} vs {q1_d0}"
    );
}

#[test]
fn qwen3_06b_embed() {
    run_embed_test(REPO_06B);
}

/// Reference scores from official Qwen3-Embedding model card: [[0.7646, 0.1414], [0.1355, 0.6000]]
#[test]
fn qwen3_06b_reference_scores() {
    let device = Device::Cpu;
    let model =
        Qwen3TextEmbedding::from_hf(REPO_06B, &device, DType::F32, 8192).expect("load model");

    let task = "Given a web search query, retrieve relevant passages that answer the query";
    let queries = [
        format!("Instruct: {task}\nQuery:What is the capital of China?"),
        format!("Instruct: {task}\nQuery:Explain gravity"),
    ];
    let documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. \
         It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ];

    let input_texts: Vec<&str> = queries
        .iter()
        .map(|s| s.as_str())
        .chain(documents)
        .collect();
    let embeddings = model.embed(&input_texts).expect("embed");

    let scores = [
        [
            cosine_sim(&embeddings[0], &embeddings[2]),
            cosine_sim(&embeddings[0], &embeddings[3]),
        ],
        [
            cosine_sim(&embeddings[1], &embeddings[2]),
            cosine_sim(&embeddings[1], &embeddings[3]),
        ],
    ];
    let expected = [[0.7646f32, 0.1414f32], [0.1355f32, 0.6000f32]];

    for i in 0..2 {
        for j in 0..2 {
            let diff = (scores[i][j] - expected[i][j]).abs();
            assert!(
                diff < 0.05,
                "score[{i}][{j}]: got {:.4}, expected {:.4}",
                scores[i][j],
                expected[i][j]
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
