use fastembed::{RerankInitOptions, RerankerModel, TextRerank};

/// Quick smoke test for the three gte-reranker-modernbert-base variants.
#[test]
fn test_gte_reranker_modernbert_base() {
    let models = [
        RerankerModel::GteRerankerModernBertBase,
        RerankerModel::GteRerankerModernBertBaseQ,
        RerankerModel::GteRerankerModernBertBaseQ4F16,
    ];

    let documents = vec![
        "hi",
        "The giant panda, sometimes called a panda bear or simply panda, is a bear species endemic to China.",
        "panda is an animal",
        "i dont know",
        "kind of mammal",
    ];

    for model in &models {
        println!("Testing {:?}", model);
        let mut reranker = TextRerank::try_new(RerankInitOptions::new(model.clone()))
            .unwrap_or_else(|e| panic!("Failed to load {:?}: {}", model, e));

        let results = reranker
            .rerank("what is panda?", documents.clone(), true, None)
            .unwrap_or_else(|e| panic!("{:?} rerank failed: {}", model, e));

        assert_eq!(results.len(), documents.len());

        let top = results[0].document.as_ref().unwrap().as_str();
        let second = results[1].document.as_ref().unwrap().as_str();
        assert!(
            top == "panda is an animal"
                || top == "The giant panda, sometimes called a panda bear or simply panda, is a bear species endemic to China.",
            "{:?}: unexpected top result: {:?}", model, top
        );
        assert_ne!(top, second, "{:?}: top two results are identical", model);
        println!("  {:?} PASS — top: {:?} (score={:.4})", model, top, results[0].score);
    }
}
