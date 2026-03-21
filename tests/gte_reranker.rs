use std::path::Path;

use fastembed::{
    OnnxSource, RerankInitOptionsUserDefined, TextRerank, TokenizerFiles, UserDefinedRerankingModel,
};

const GTE_SNAP: &str = "/Volumes/backups/ai/models--Alibaba-NLP--gte-reranker-modernbert-base/snapshots/f7481e6055501a30fb19d090657df9ec1f79ab2c";

fn load_tokenizer_files(dir: &str) -> TokenizerFiles {
    let read = |name: &str| std::fs::read(format!("{}/{}", dir, name)).unwrap();
    TokenizerFiles {
        tokenizer_file: read("tokenizer.json"),
        config_file: read("config.json"),
        special_tokens_map_file: read("special_tokens_map.json"),
        tokenizer_config_file: read("tokenizer_config.json"),
    }
}

/// Quick smoke test for the three gte-reranker-modernbert-base variants.
/// Skipped automatically if the local model snapshot is not available.
#[test]
fn test_gte_reranker_modernbert_base() {
    let models: &[(&str, &str)] = &[
        ("GteRerankerModernBertBase (FP32)", "onnx/model.onnx"),
        ("GteRerankerModernBertBaseQ (INT8)", "onnx/model_int8.onnx"),
        ("GteRerankerModernBertBaseQ4F16 (Q4F16)", "onnx/model_q4f16.onnx"),
    ];

    let snap = Path::new(GTE_SNAP);
    if !snap.exists() {
        eprintln!("SKIP gte_reranker tests — local snapshot not found at {GTE_SNAP}");
        return;
    }

    let documents = vec![
        "hi",
        "The giant panda, sometimes called a panda bear or simply panda, is a bear species endemic to China.",
        "panda is an animal",
        "i dont know",
        "kind of mammal",
    ];

    for (name, model_file) in models {
        let onnx_path = snap.join(model_file);
        if !onnx_path.exists() {
            eprintln!("SKIP {name} — {onnx_path:?} not found");
            continue;
        }

        println!("Testing {name}");

        let model = UserDefinedRerankingModel::new(
            OnnxSource::File(onnx_path),
            load_tokenizer_files(GTE_SNAP),
        );

        let mut opts = RerankInitOptionsUserDefined::default();
        opts.max_length = 512;
        let mut reranker = TextRerank::try_new_from_user_defined(model, opts)
            .unwrap_or_else(|e| panic!("Failed to load {name}: {e}"));

        let results = reranker
            .rerank("what is panda?", documents.clone(), true, None)
            .unwrap_or_else(|e| panic!("{name} rerank failed: {e}"));

        assert_eq!(results.len(), documents.len());

        let top = results[0].document.as_ref().unwrap().as_str();
        let second = results[1].document.as_ref().unwrap().as_str();
        assert!(
            top == "panda is an animal"
                || top == "The giant panda, sometimes called a panda bear or simply panda, is a bear species endemic to China.",
            "{name}: unexpected top result: {top:?}"
        );
        assert_ne!(top, second, "{name}: top two results are identical");
        println!(
            "  {name} PASS — top: {top:?} (score={:.4})",
            results[0].score
        );
    }
}
