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

#[test]
fn test_embedding_consistency_multiple_models() {
    // Test consistency across different models
    let models = vec![
        EmbeddingModel::AllMiniLML6V2,
        EmbeddingModel::BGESmallENV15,
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

