#![cfg(feature = "hf-hub")]

use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};

#[test]
fn test_embedding_consistency_issue_171() {
    // This reproduces the issue described in GitHub issue #171
    // where TextEmbedding returns inconsistent results after v5.0
    
    let q = "red car";
    let mut fe = TextEmbedding::try_new(InitOptions::new(EmbeddingModel::ClipVitB32)).unwrap();
    let mut first = None;
    
    for i in 0..100 {
        let vec = fe.embed(vec![q], None).unwrap();
        if first.is_none() {
            first = Some(vec[0].clone());
        } else {
            if vec[0] != *first.as_ref().unwrap() {
                panic!("Embedding changed after {} iterations", i);
            }
        }
    }
    
    println!("All 100 embeddings were consistent");
}

// Individual model tests for easier debugging
#[test]
fn test_all_mini_lm_l6_v2_consistency() {
    test_model_consistency(EmbeddingModel::AllMiniLML6V2, "AllMiniLML6V2");
}

#[test]
fn test_bge_small_en_v15_consistency() {
    test_model_consistency(EmbeddingModel::BGESmallENV15, "BGESmallENV15");
}

#[test]
fn test_clip_vit_b32_consistency() {
    test_model_consistency(EmbeddingModel::ClipVitB32, "ClipVitB32");
}

#[test]
fn test_all_mini_lm_l12_v2_consistency() {
    test_model_consistency(EmbeddingModel::AllMiniLML12V2, "AllMiniLML12V2");
}

#[test]
fn test_bge_base_en_v15_consistency() {
    test_model_consistency(EmbeddingModel::BGEBaseENV15, "BGEBaseENV15");
}

fn test_model_consistency(model: EmbeddingModel, model_name: &str) {
    println!("Testing model: {}", model_name);
    
    let q = "hello world";
    let mut fe = TextEmbedding::try_new(InitOptions::new(model.clone())).unwrap();
    let mut first = None;
    
    for i in 0..10 {
        let vec = fe.embed(vec![q], None).unwrap();
        if first.is_none() {
            first = Some(vec[0].clone());
            println!("Model {}: First embedding captured (length: {})", model_name, vec[0].len());
        } else {
            if vec[0] != *first.as_ref().unwrap() {
                // Print some debugging info before panicking
                println!("Model {}: Embedding mismatch at iteration {}", model_name, i);
                println!("First few values of original: {:?}", &first.as_ref().unwrap()[0..std::cmp::min(5, first.as_ref().unwrap().len())]);
                println!("First few values of current:  {:?}", &vec[0][0..std::cmp::min(5, vec[0].len())]);
                panic!("Embedding changed for model {:?} after {} iterations", model, i);
            }
        }
    }
    
    println!("Model {}: All 10 embeddings were consistent", model_name);
}

#[test]
fn test_embedding_consistency_multiple_models() {
    // Test consistency across different models to identify which ones have deterministic issues
    let models = vec![
        EmbeddingModel::AllMiniLML6V2,
        EmbeddingModel::BGESmallENV15,
        EmbeddingModel::ClipVitB32,
        EmbeddingModel::AllMiniLML12V2,
        EmbeddingModel::BGEBaseENV15,
    ];
    
    for model in models {
        let q = "hello world";
        let mut fe = TextEmbedding::try_new(InitOptions::new(model.clone())).unwrap();
        let mut first = None;
        
        for i in 0..10 {
            let vec = fe.embed(vec![q], None).unwrap();
            if first.is_none() {
                first = Some(vec[0].clone());
            } else {
                if vec[0] != *first.as_ref().unwrap() {
                    panic!("Embedding changed for model {:?} after {} iterations", model, i);
                }
            }
        }
        
        println!("Model {:?}: All 10 embeddings were consistent", model);
    }
}

