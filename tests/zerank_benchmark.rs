use fastembed::{
    OnnxSource, RerankInitOptionsUserDefined, TextRerank, TokenizerFiles, UserDefinedRerankingModel,
};
use std::time::Instant;

const TOKENIZER_DIR: &str = "/private/tmp/zerank_export/tokenizer_files";
const ZERANK_TEMPLATE: &str =
    "<|im_start|>user\nQuery: {query}\nDocument: {doc}\nRelevant:<|im_end|>\n<|im_start|>assistant\n";

fn load_tokenizer_files() -> TokenizerFiles {
    let read = |name: &str| std::fs::read(format!("{TOKENIZER_DIR}/{name}")).unwrap();
    TokenizerFiles {
        tokenizer_file: read("tokenizer.json"),
        config_file: read("config.json"),
        special_tokens_map_file: read("special_tokens_map.json"),
        tokenizer_config_file: read("tokenizer_config.json"),
    }
}

/// Benchmark for zerank-1-small ONNX variants using local model files.
///
/// Scores (query, doc) pairs via Yes-token logit [batch, 1].
/// Applies Qwen3 chat template before tokenization (required for correct rankings).
/// Batch_size=1: current export has static causal mask (re-export needed for batch > 1).
///
/// Ground-truth ranking (CrossEncoder): panda desc > panda animal > mammal > i dont know > hi
#[test]
fn test_zerank_small_benchmark() {
    let models: &[(&str, &str)] = &[
        (
            "INT8",
            "/Volumes/backups/ai/zerank_onnx_int8/model_int8.onnx",
        ),
        ("FP16", "/private/tmp/zerank_export/zerank_onnx/model.onnx"),
    ];

    let query = "what is a panda?";
    let documents = vec![
        "hi",
        "The giant panda, sometimes called a panda bear or simply panda, is a bear species endemic to China.",
        "panda is an animal",
        "i dont know",
        "kind of mammal",
    ];

    for (label, path) in models {
        if !std::path::Path::new(path).exists() {
            println!("Skipping zerank {label}: {path} not found");
            continue;
        }

        println!("\nLoading zerank-1-small ({label}) from {path}...");
        let load_start = Instant::now();
        let model =
            UserDefinedRerankingModel::new(OnnxSource::File(path.into()), load_tokenizer_files());
        let mut opts = RerankInitOptionsUserDefined::default();
        opts.max_length = 512;
        opts.prompt_template = Some(ZERANK_TEMPLATE.to_string());

        let mut reranker = TextRerank::try_new_from_user_defined(model, opts)
            .unwrap_or_else(|e| panic!("Failed to load zerank {label}: {e}"));
        println!("  Load time: {:.1}s", load_start.elapsed().as_secs_f32());

        // Warm-up (batch_size=1: static causal mask)
        let _ = reranker
            .rerank(query, documents.clone(), false, Some(1))
            .expect("warm-up failed");

        // Timed runs
        let n_runs = 3;
        let t = Instant::now();
        for _ in 0..n_runs {
            let _ = reranker
                .rerank(query, documents.clone(), false, Some(1))
                .expect("rerank failed");
        }
        let avg_ms = t.elapsed().as_millis() as f32 / n_runs as f32;

        let results = reranker
            .rerank(query, documents.clone(), true, Some(1))
            .unwrap_or_else(|e| panic!("zerank {label} rerank failed: {e}"));

        assert_eq!(results.len(), documents.len());

        // Ground truth: panda-related docs should be top-2
        let top2: Vec<&str> = results[..2]
            .iter()
            .map(|r| r.document.as_ref().unwrap().as_str())
            .collect();
        let panda_docs = [
            "The giant panda, sometimes called a panda bear or simply panda, is a bear species endemic to China.",
            "panda is an animal",
        ];
        assert!(
            top2.iter().any(|d| panda_docs.contains(d)),
            "zerank {label}: expected panda doc in top-2, got: {top2:?}"
        );

        println!("  zerank {label} PASS");
        println!(
            "  Avg latency ({} docs, batch=1): {avg_ms:.0}ms",
            documents.len()
        );
        for r in &results {
            println!("    [{:.3}] {}", r.score, r.document.as_ref().unwrap());
        }
    }
}
