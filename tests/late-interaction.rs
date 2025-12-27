//! Tests for late interaction (ColBERT-style) embeddings and MUVERA post-processing.

#![cfg(feature = "hf-hub")]

#[cfg(feature = "muvera")]
use fastembed::Muvera;
use fastembed::{LateInteractionInitOptions, LateInteractionModel, LateInteractionTextEmbedding};

// Canonical values for "Hello World" with colbert-ir/colbertv2.0
// First 5 columns of first 5 tokens
const CANONICAL_DOC_VALUES_COLBERT: [[f32; 5]; 5] = [
    [0.0759, 0.0841, -0.0299, 0.0374, 0.0254],
    [0.0005, -0.0163, -0.0127, 0.2165, 0.1517],
    [-0.0257, -0.0575, 0.0135, 0.2202, 0.1896],
    [0.0846, 0.0122, 0.0032, -0.0109, -0.1041],
    [0.0477, 0.1078, -0.0314, 0.016, 0.0156],
];

const CANONICAL_QUERY_VALUES_COLBERT: [[f32; 5]; 5] = [
    [0.0824, 0.0872, -0.0324, 0.0418, 0.024],
    [-0.0007, -0.0154, -0.0113, 0.2277, 0.1528],
    [-0.0251, -0.0565, 0.0136, 0.2236, 0.1838],
    [0.0848, 0.0056, 0.0041, -0.0036, -0.1032],
    [0.0574, 0.1072, -0.0332, 0.0233, 0.0209],
];

const CANONICAL_DOC_VALUES_ANSWERAI: [[f32; 5]; 5] = [
    [-0.07281, 0.04632, -0.04711, 0.00762, -0.07374],
    [-0.04464, 0.04426, -0.074, 0.01801, -0.05233],
    [0.09936, -0.05123, -0.04925, -0.05276, -0.08944],
    [0.01644, 0.0203, -0.03789, 0.03165, -0.06501],
    [-0.07281, 0.04633, -0.04711, 0.00762, -0.07374],
];

fn assert_close(actual: &[f32], expected: &[f32], atol: f32) {
    assert_eq!(actual.len(), expected.len(), "Length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < atol,
            "Mismatch at index {}: actual={}, expected={}, diff={}",
            i,
            a,
            e,
            (a - e).abs()
        );
    }
}

// Late Interaction Embedding Tests

#[test]
fn test_colbert_document_embedding_canonical_values() {
    let mut model = LateInteractionTextEmbedding::try_new(LateInteractionInitOptions::new(
        LateInteractionModel::ColBERTV2,
    ))
    .unwrap();

    let embeddings = model.embed(&["Hello World"], None).unwrap();

    assert_eq!(embeddings.len(), 1);
    assert!(embeddings[0].len() >= 5, "Expected at least 5 tokens");
    assert_eq!(embeddings[0][0].len(), 128, "ColBERT dim should be 128");

    for (token_idx, expected_row) in CANONICAL_DOC_VALUES_COLBERT.iter().enumerate() {
        let actual: Vec<f32> = embeddings[0][token_idx][..5].to_vec();
        assert_close(&actual, expected_row, 2e-3);
    }
}

#[test]
fn test_colbert_query_embedding_canonical_values() {
    let mut model = LateInteractionTextEmbedding::try_new(LateInteractionInitOptions::new(
        LateInteractionModel::ColBERTV2,
    ))
    .unwrap();

    let embeddings = model.query_embed(&["Hello World"], None).unwrap();

    assert_eq!(embeddings.len(), 1);
    assert_eq!(
        embeddings[0].len(),
        32,
        "Query should be padded to 32 tokens"
    );
    assert_eq!(embeddings[0][0].len(), 128);

    for (token_idx, expected_row) in CANONICAL_QUERY_VALUES_COLBERT.iter().enumerate() {
        let actual: Vec<f32> = embeddings[0][token_idx][..5].to_vec();
        assert_close(&actual, expected_row, 2e-3);
    }
}

#[test]
fn test_answerai_document_embedding_canonical_values() {
    let mut model = LateInteractionTextEmbedding::try_new(LateInteractionInitOptions::new(
        LateInteractionModel::AnswerAIColBERTSmallV1,
    ))
    .unwrap();

    let embeddings = model.embed(&["Hello World"], None).unwrap();

    assert_eq!(embeddings.len(), 1);
    assert_eq!(embeddings[0][0].len(), 96, "AnswerAI dim should be 96");

    for (token_idx, expected_row) in CANONICAL_DOC_VALUES_ANSWERAI.iter().enumerate() {
        let actual: Vec<f32> = embeddings[0][token_idx][..5].to_vec();
        assert_close(&actual, expected_row, 2e-3);
    }
}

#[test]
fn test_embedding_dimension() {
    let colbert = LateInteractionTextEmbedding::try_new(LateInteractionInitOptions::new(
        LateInteractionModel::ColBERTV2,
    ))
    .unwrap();
    assert_eq!(colbert.dim(), 128);

    let answerai = LateInteractionTextEmbedding::try_new(LateInteractionInitOptions::new(
        LateInteractionModel::AnswerAIColBERTSmallV1,
    ))
    .unwrap();
    assert_eq!(answerai.dim(), 96);
}

#[test]
fn test_batch_size_consistency() {
    let mut model = LateInteractionTextEmbedding::try_new(LateInteractionInitOptions::new(
        LateInteractionModel::AnswerAIColBERTSmallV1,
    ))
    .unwrap();

    let documents = vec![
        "short document",
        "A bit longer document, which should not affect the size",
    ];

    let result_batch_1 = model.embed(&documents, Some(1)).unwrap();
    let result_batch_2 = model.embed(&documents, Some(2)).unwrap();

    assert_eq!(
        result_batch_1[0].len(),
        result_batch_2[0].len(),
        "Batch size should not affect token count"
    );

    for (t1, t2) in result_batch_1[0].iter().zip(result_batch_2[0].iter()) {
        assert_close(t1, t2, 1e-5);
    }
}

// MUVERA Post-Processing Tests

#[cfg(feature = "muvera")]
mod muvera_tests {
    use super::*;

    // Canonical MUVERA output for deterministic sin-based test vectors
    const MUVERA_EXPECTED_FIRST_10: [f32; 10] = [
        2.0179653,
        1.6323578,
        -1.5774617,
        -0.26919794,
        3.2250175,
        -2.0104198,
        -2.2146697,
        -1.0453973,
        -1.2936,
        2.5332289,
    ];

    #[test]
    fn test_muvera_canonical_values() {
        let muvera = Muvera::new(128, Some(5), Some(16), Some(20), Some(42)).unwrap();

        let vectors: Vec<Vec<f32>> = (0..10)
            .map(|i| (0..128).map(|j| ((i * 128 + j) as f32).sin()).collect())
            .collect();

        let fde = muvera.process_document(&vectors);

        assert_eq!(fde.len(), 10240);
        assert_close(&fde[..10], &MUVERA_EXPECTED_FIRST_10, 1e-5);
    }

    #[test]
    fn test_muvera_deterministic_same_seed() {
        let muvera1 = Muvera::new(128, Some(5), Some(16), Some(20), Some(42)).unwrap();
        let muvera2 = Muvera::new(128, Some(5), Some(16), Some(20), Some(42)).unwrap();

        let vectors: Vec<Vec<f32>> = (0..10)
            .map(|i| (0..128).map(|j| ((i * 128 + j) as f32).sin()).collect())
            .collect();

        let fde1 = muvera1.process_document(&vectors);
        let fde2 = muvera2.process_document(&vectors);

        assert_eq!(fde1, fde2);
    }

    #[test]
    fn test_muvera_different_seeds_differ() {
        let muvera1 = Muvera::new(128, Some(5), Some(16), Some(20), Some(42)).unwrap();
        let muvera2 = Muvera::new(128, Some(5), Some(16), Some(20), Some(123)).unwrap();

        let vectors: Vec<Vec<f32>> = (0..10)
            .map(|i| (0..128).map(|j| ((i * 128 + j) as f32).sin()).collect())
            .collect();

        let fde1 = muvera1.process_document(&vectors);
        let fde2 = muvera2.process_document(&vectors);

        assert_ne!(fde1, fde2);
    }

    #[test]
    fn test_muvera_document_vs_query_processing_differs() {
        let muvera = Muvera::new(128, Some(5), Some(16), Some(20), Some(42)).unwrap();

        let vectors: Vec<Vec<f32>> = (0..10)
            .map(|i| (0..128).map(|j| ((i * 128 + j) as f32).sin()).collect())
            .collect();

        let doc_fde = muvera.process_document(&vectors);
        let query_fde = muvera.process_query(&vectors);

        // Document processing: normalize_by_count=true, fill_empty_clusters=true
        // Query processing: normalize_by_count=false, fill_empty_clusters=false
        assert_ne!(doc_fde, query_fde);
    }

    #[test]
    fn test_muvera_empty_cluster_handling() {
        // With only 2 vectors and 2^3=8 clusters, most clusters will be empty
        let muvera = Muvera::new(8, Some(3), Some(4), Some(2), Some(42)).unwrap();
        let vectors: Vec<Vec<f32>> = vec![vec![1.0; 8], vec![-1.0; 8]];

        let fde = muvera.process_document(&vectors);

        assert_eq!(fde.len(), muvera.embedding_size());
        assert!(fde.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_muvera_rejects_invalid_dim_proj() {
        // dim_proj > dim should fail
        let result = Muvera::new(128, Some(5), Some(256), Some(20), Some(42));
        assert!(result.is_err());
    }

    #[test]
    fn test_muvera_embedding_size_calculation() {
        let muvera = Muvera::new(128, Some(5), Some(16), Some(20), Some(42)).unwrap();
        // r_reps * 2^k_sim * dim_proj = 20 * 32 * 16 = 10240
        assert_eq!(muvera.embedding_size(), 10240);
    }

    #[test]
    fn test_muvera_with_colbert_model() {
        let mut model = LateInteractionTextEmbedding::try_new(LateInteractionInitOptions::new(
            LateInteractionModel::ColBERTV2,
        ))
        .unwrap();

        let muvera =
            Muvera::from_late_interaction_model(&model, Some(5), Some(16), Some(20), Some(42))
                .unwrap();

        let doc_embeddings = model
            .embed(&["This is a test document about neural networks."], None)
            .unwrap();
        let query_embeddings = model
            .query_embed(&["What are neural networks?"], None)
            .unwrap();

        let doc_fde = muvera.process_document(&doc_embeddings[0]);
        let query_fde = muvera.process_query(&query_embeddings[0]);

        assert_eq!(doc_fde.len(), 10240);
        assert_eq!(query_fde.len(), 10240);

        let similarity: f32 = doc_fde
            .iter()
            .zip(query_fde.iter())
            .map(|(a, b)| a * b)
            .sum();
        assert!(
            similarity > 0.0,
            "Related query-doc should have positive similarity"
        );
    }

    #[test]
    fn test_muvera_preserves_retrieval_ranking() {
        // Verifies MUVERA ranking matches ColBERT MaxSim ranking
        let mut model = LateInteractionTextEmbedding::try_new(LateInteractionInitOptions::new(
            LateInteractionModel::ColBERTV2,
        ))
        .unwrap();

        let muvera =
            Muvera::from_late_interaction_model(&model, Some(5), Some(16), Some(20), Some(42))
                .unwrap();

        let documents = vec![
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language.",
        ];
        let query = "What is machine learning?";

        let doc_embeddings = model.embed(&documents, None).unwrap();
        let query_embeddings = model.query_embed(&[query], None).unwrap();

        // MUVERA scores
        let query_fde = muvera.process_query(&query_embeddings[0]);
        let muvera_scores: Vec<f32> = doc_embeddings
            .iter()
            .map(|d| {
                let doc_fde = muvera.process_document(d);
                query_fde
                    .iter()
                    .zip(doc_fde.iter())
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect();

        // MaxSim scores (ground truth)
        let maxsim_scores: Vec<f32> = doc_embeddings
            .iter()
            .map(|doc_emb| {
                query_embeddings[0]
                    .iter()
                    .map(|q| {
                        doc_emb
                            .iter()
                            .map(|d| q.iter().zip(d.iter()).map(|(a, b)| a * b).sum::<f32>())
                            .fold(f32::NEG_INFINITY, f32::max)
                    })
                    .sum()
            })
            .collect();

        // Both should rank doc0 (ML) higher than doc1 (Python)
        assert!(
            muvera_scores[0] > muvera_scores[1],
            "MUVERA: doc0={} should beat doc1={}",
            muvera_scores[0],
            muvera_scores[1]
        );
        assert!(
            maxsim_scores[0] > maxsim_scores[1],
            "MaxSim: doc0={} should beat doc1={}",
            maxsim_scores[0],
            maxsim_scores[1]
        );

        // MaxSim values should match Python exactly
        assert_close(&[maxsim_scores[0]], &[29.5733], 0.01);
        assert_close(&[maxsim_scores[1]], &[9.9226], 0.01);
    }
}
