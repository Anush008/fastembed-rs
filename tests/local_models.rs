//! Integration tests for recently added model variants.
//!
//! All models are loaded from the shared local CrispSorter cache, so no
//! network access is needed.  Any model not present on disk is skipped.
//!
//! Run with:
//!   cargo test --test local_models -- --nocapture

use fastembed::{
    EmbeddingModel, InitOptionsUserDefined, Pooling, TextEmbedding, TextInitOptions, TokenizerFiles,
    UserDefinedEmbeddingModel,
};
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

// ── helpers ──────────────────────────────────────────────────────────────────

const CRISP_MODELS: &str =
    "Library/Application Support/com.christianstrobele.crispsorter/models";

fn models_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap()).join(CRISP_MODELS)
}

/// Resolve the `snapshots/{hash}` directory for a HuggingFace model that is
/// already present in the local hf-hub cache.
fn hf_snap(cache: &Path, model_code: &str) -> Option<PathBuf> {
    let dir_name = format!("models--{}", model_code.replace('/', "--"));
    let refs_main = cache.join(&dir_name).join("refs/main");
    let hash = fs::read_to_string(refs_main).ok()?;
    let snap = cache
        .join(dir_name)
        .join("snapshots")
        .join(hash.trim());
    snap.exists().then_some(snap)
}

/// Read the four tokenizer files from a snapshot root (or flat model dir).
/// `special_tokens_map.json` is optional — an empty JSON object is used if absent.
fn tok(dir: &Path) -> Option<TokenizerFiles> {
    let read = |name| fs::read(dir.join(name)).ok();
    Some(TokenizerFiles {
        tokenizer_file: read("tokenizer.json")?,
        config_file: read("config.json")?,
        special_tokens_map_file: read("special_tokens_map.json")
            .unwrap_or_else(|| b"{}".to_vec()),
        tokenizer_config_file: read("tokenizer_config.json")?,
    })
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// Embed three sentences and check: correct count, unit norm, semantic ordering.
fn smoke_test(label: &str, model: &mut TextEmbedding) {
    let sentences = vec![
        "Semantic search with neural embeddings",
        "Neural retrieval using dense vector representations",
        "The quick brown fox jumped over the lazy dog",
    ];

    let t0 = Instant::now();
    let embs = model
        .embed(sentences.clone(), Some(3))
        .unwrap_or_else(|e| panic!("[{label}] embed failed: {e}"));
    let elapsed = t0.elapsed();

    assert_eq!(embs.len(), 3, "[{label}] wrong embedding count");

    for e in &embs {
        let norm: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "[{label}] embedding not unit-normalised (norm={norm:.6})"
        );
    }

    let sim_pos = cosine(&embs[0], &embs[1]);
    let sim_neg = cosine(&embs[0], &embs[2]);
    println!(
        "[{label}]  dim={}  latency={:.0?}  sim(semantic)={:.4}  sim(unrelated)={:.4}  ordering={}",
        embs[0].len(),
        elapsed,
        sim_pos,
        sim_neg,
        if sim_pos > sim_neg { "✓" } else { "✗" }
    );
}

// ── tests using TextEmbedding::try_new() + local hf-hub cache ────────────────

/// Snowflake Arctic Embed L v2 — CLS pooling, no external data.
/// Loaded via try_new() pointing at the CrispSorter hf-hub cache.
#[test]
fn test_snowflake_arctic_l_v2() {
    let dir = models_dir();
    if hf_snap(&dir, "Snowflake/snowflake-arctic-embed-l-v2.0").is_none() {
        println!("SKIPPED — Snowflake Arctic-L v2 not in local cache");
        return;
    }
    let mut model = TextEmbedding::try_new(
        TextInitOptions::new(EmbeddingModel::SnowflakeArcticEmbedLV2)
            .with_cache_dir(dir)
            .with_show_download_progress(false),
    )
    .expect("failed to load SnowflakeArcticEmbedLV2");

    smoke_test("SnowflakeArcticEmbedLV2", &mut model);
}

// ── tests using UserDefinedEmbeddingModel::from_file() ───────────────────────
// This exercises the new OnnxSource::File path — ORT resolves the companion
// .onnx.data file automatically, so only the ONNX graph path is needed.

/// PIXIE-Rune-v1.0 — mean pooling, external data companion (onnx/model.onnx.data).
#[test]
fn test_pixie_rune_v1() {
    let dir = models_dir();
    let snap = match hf_snap(&dir, "telepix/PIXIE-Rune-v1.0") {
        Some(s) => s,
        None => {
            println!("SKIPPED — PIXIE-Rune-v1.0 not in local cache");
            return;
        }
    };
    let tokenizer_files = tok(&snap).expect("PIXIE tokenizer files missing");
    let onnx_path = snap.join("onnx/model.onnx");

    let model_def = UserDefinedEmbeddingModel::from_file(onnx_path, tokenizer_files)
        .with_pooling(Pooling::Mean);
    let mut model =
        TextEmbedding::try_new_from_user_defined(model_def, InitOptionsUserDefined::new())
            .expect("failed to load PIXIE-Rune-v1.0");

    smoke_test("PixieRuneV1", &mut model);
}

/// Jina Embeddings v5 Nano — uses the pre-pooled 'sentence_embedding' output.
///
/// This model outputs both `last_hidden_state [batch, seq, 768]` and a
/// pre-pooled `sentence_embedding [batch, 768]`.  We select the latter via
/// `with_output_key` to skip redundant pooling.
///
/// Jina v5 retrieval models expect task prefixes ("query: " / "passage: ")
/// for best quality; without them similarities are near zero.
#[test]
fn test_jina_v5_nano() {
    let dir = models_dir();
    let snap = match hf_snap(&dir, "jinaai/jina-embeddings-v5-text-nano-retrieval") {
        Some(s) => s,
        None => {
            println!("SKIPPED — Jina v5 Nano not in local cache");
            return;
        }
    };
    let tokenizer_files = tok(&snap).expect("Jina v5 Nano tokenizer files missing");
    let onnx_path = snap.join("onnx/model_quantized.onnx");

    let model_def = UserDefinedEmbeddingModel::from_file(onnx_path, tokenizer_files)
        // sentence_embedding is pre-pooled [batch, 768]; Cls on 2D is a pass-through.
        .with_pooling(Pooling::Cls)
        .with_output_key(fastembed::OutputKey::ByName("sentence_embedding"));

    let mut model =
        TextEmbedding::try_new_from_user_defined(model_def, InitOptionsUserDefined::new())
            .expect("failed to load Jina v5 Nano");

    // Task prefixes are required for meaningful retrieval quality.
    let sentences = vec![
        "query: Semantic search with neural embeddings",
        "query: Neural retrieval using dense vector representations",
        "passage: The quick brown fox jumped over the lazy dog",
    ];

    let t0 = Instant::now();
    let embs = model.embed(sentences.clone(), Some(3)).expect("embed failed");
    let elapsed = t0.elapsed();

    assert_eq!(embs.len(), 3);
    for e in &embs {
        let norm: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "not unit-normalised (norm={norm:.6})");
    }

    let sim_pos = cosine(&embs[0], &embs[1]);
    let sim_neg = cosine(&embs[0], &embs[2]);
    println!(
        "[JinaV5Nano]  dim={}  latency={:.0?}  sim(semantic)={:.4}  sim(unrelated)={:.4}  ordering={}",
        embs[0].len(), elapsed, sim_pos, sim_neg,
        if sim_pos > sim_neg { "✓" } else { "✗" }
    );
    assert!(sim_pos > sim_neg, "semantic ordering failed");
}

/// Qwen3-Embedding-0.6B uint8 (electroglyph calibrated).
///
/// Uses `Pooling::PrePooledU8` which handles u8 tensor extraction and affine
/// dequantization (`f32 = (u8 - zero_point) × scale`) transparently via `embed()`.
#[test]
fn test_qwen3_uint8() {
    let dir = models_dir();
    let snap = match hf_snap(&dir, "electroglyph/Qwen3-Embedding-0.6B-onnx-uint8") {
        Some(s) => s,
        None => {
            println!("SKIPPED — Qwen3 uint8 not in local cache");
            return;
        }
    };
    let tokenizer_files = tok(&snap).expect("Qwen3 uint8 tokenizer files missing");
    let onnx_path = snap.join("dynamic_uint8.onnx");

    // Dequantization parameters from electroglyph model card:
    // range [-0.3009805, 0.3952634] → scale=0.002730, zero_point=110
    let mut model = TextEmbedding::try_new_from_user_defined(
        UserDefinedEmbeddingModel::from_file(onnx_path, tokenizer_files)
            .with_pooling(Pooling::PrePooledU8 {
                scale: 0.0027303685,
                zero_point: 110,
            })
            .with_output_key(fastembed::OutputKey::ByName("sentence_embedding_quantized")),
        InitOptionsUserDefined::new(),
    )
    .expect("failed to load Qwen3 uint8");

    let texts = vec![
        "Semantic search with neural embeddings",
        "Neural retrieval using dense vector representations",
        "The quick brown fox jumped over the lazy dog",
    ];

    let t0 = Instant::now();
    let embs = model.embed(texts.clone(), Some(texts.len())).expect("embed failed");
    let elapsed = t0.elapsed();

    assert_eq!(embs.len(), texts.len(), "wrong embedding count");
    for e in &embs {
        let norm: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "[Qwen3Uint8] embedding not unit-normalised (norm={norm:.6})");
    }

    let sim_pos = cosine(&embs[0], &embs[1]);
    let sim_neg = cosine(&embs[0], &embs[2]);
    println!(
        "[Qwen3Uint8]  dim={}  latency={:.0?}  sim(semantic)={:.4}  sim(unrelated)={:.4}  ordering={}",
        embs[0].len(), elapsed, sim_pos, sim_neg,
        if sim_pos > sim_neg { "✓" } else { "✗" }
    );
    assert!(sim_pos > sim_neg, "semantic ordering failed");
}

/// Octen-Embedding-0.6B INT8 — last-token pooling, loaded from flat local dir
/// (not hf-hub format).  Exercises UserDefinedEmbeddingModel::from_file() with
/// the companion .onnx.data resolved automatically by ORT.
#[test]
fn test_octen_int8_local() {
    let dir = models_dir().join("octen-embedding-0.6b-int8");
    if !dir.exists() {
        println!("SKIPPED — octen-embedding-0.6b-int8 not found");
        return;
    }
    let tokenizer_files = tok(&dir).expect("Octen INT8 tokenizer files missing");
    let onnx_path = dir.join("model.int8.onnx");

    let model_def = UserDefinedEmbeddingModel::from_file(onnx_path, tokenizer_files)
        .with_pooling(Pooling::LastToken);
    let mut model =
        TextEmbedding::try_new_from_user_defined(model_def, InitOptionsUserDefined::new())
            .expect("failed to load Octen INT8");

    smoke_test("OctenInt8", &mut model);
}

/// Octen-Embedding-0.6B INT4 (MatMulNBits) — last-token pooling, batch=1 only.
#[test]
fn test_octen_int4_local() {
    let dir = models_dir().join("octen-embedding-0.6b-int4");
    if !dir.exists() {
        println!("SKIPPED — octen-embedding-0.6b-int4 not found");
        return;
    }
    let tokenizer_files = tok(&dir).expect("Octen INT4 tokenizer files missing");
    let onnx_path = dir.join("model.int4.onnx");

    // INT4 was exported with the legacy ONNX exporter — static batch=1 inside
    // the causal mask, so we must embed one text at a time.
    let model_def = UserDefinedEmbeddingModel::from_file(onnx_path, tokenizer_files)
        .with_pooling(Pooling::LastToken);
    let mut model =
        TextEmbedding::try_new_from_user_defined(model_def, InitOptionsUserDefined::new())
            .expect("failed to load Octen INT4");

    // batch_size=1 to stay within the static shape constraint
    let t0 = Instant::now();
    let embs = model
        .embed(
            vec!["Semantic search with neural embeddings"],
            Some(1),
        )
        .expect("Octen INT4 embed failed");

    let norm: f32 = embs[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-3,
        "Octen INT4 embedding not unit-normalised (norm={norm:.6})"
    );
    println!(
        "[OctenInt4]  dim={}  latency={:.0?}  norm={norm:.6}",
        embs[0].len(),
        t0.elapsed()
    );
}
