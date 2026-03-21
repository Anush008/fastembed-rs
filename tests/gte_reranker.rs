#![cfg(feature = "hf-hub")]

use fastembed::{RerankInitOptions, RerankerModel, TextRerank};

fn model_is_available_offline(model_code: &str) -> bool {
    if std::env::var("HF_HUB_OFFLINE").as_deref() != Ok("1") {
        return true;
    }
    let dir_name = format!("models--{}", model_code.replace('/', "--"));
    let cache_dirs =
        std::env::var("FASTEMBED_CACHE_DIR").unwrap_or_else(|_| ".fastembed_cache".into());
    for dir in cache_dirs.split(':').filter(|s| !s.is_empty()) {
        let refs_main = std::path::Path::new(dir)
            .join(&dir_name)
            .join("refs/main");
        if let Ok(hash) = std::fs::read_to_string(&refs_main) {
            let snap = std::path::Path::new(dir)
                .join(&dir_name)
                .join("snapshots")
                .join(hash.trim());
            if snap.exists() {
                return true;
            }
        }
    }
    false
}

/// Smoke test for gte-reranker-modernbert-base variants.
/// Marked #[ignore] because these are large models (150–600 MB).
/// Run with: cargo test gte_reranker -- --include-ignored
#[ignore]
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
        let info = TextRerank::get_model_info(model);
        println!("Testing {:?}", model);

        let offline = std::env::var("HF_HUB_OFFLINE").as_deref() == Ok("1");
        if offline && !model_is_available_offline(&info.model_code) {
            eprintln!(
                "SKIP {:?} — not in local cache (HF_HUB_OFFLINE=1)",
                model
            );
            continue;
        }

        let mut reranker = match TextRerank::try_new(RerankInitOptions::new(model.clone())) {
            Ok(r) => r,
            Err(e) if offline => {
                eprintln!("SKIP {:?} — load failed in offline mode: {}", model, e);
                continue;
            }
            Err(e) => panic!("Failed to load {:?}: {}", model, e),
        };

        let results = reranker
            .rerank("what is panda?", documents.clone(), true, None)
            .unwrap_or_else(|e| panic!("{:?} rerank failed: {}", model, e));

        assert_eq!(results.len(), documents.len());

        let top = results[0].document.as_ref().unwrap().as_str();
        let second = results[1].document.as_ref().unwrap().as_str();
        assert!(
            top == "panda is an animal"
                || top == "The giant panda, sometimes called a panda bear or simply panda, is a bear species endemic to China.",
            "{:?}: unexpected top result: {:?}",
            model,
            top
        );
        assert_ne!(top, second, "{:?}: top two results are identical", model);
        println!(
            "  {:?} PASS — top: {:?} (score={:.4})",
            model, top, results[0].score
        );
    }
}
