//! Embedding quality benchmark for locally cached models.
//!
//! Runs a miniature retrieval task over a hand-annotated 20-document corpus
//! spanning 5 topics (4 docs/topic).  For each topic one query is used; the
//! 4 documents that share the topic are "relevant".
//!
//! Reported metrics per model:
//!   MRR   – Mean Reciprocal Rank of the first relevant document.
//!   MAP@5 – Mean Average Precision at 5.
//!   Coh   – Avg intra-cluster cosine similarity (semantic compactness).
//!   Sep   – Avg inter-cluster cosine similarity (cross-topic leakage).
//!   Ratio – Coh/Sep (higher is better; >1.5 is generally good).
//!
//! All models are loaded from the shared CrispSorter cache so no network
//! access is required.  Models not present on disk are silently skipped.
//!
//! Run with:
//!   cargo test --test quality_bench -- --nocapture

use fastembed::{InitOptionsUserDefined, Pooling, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel};
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

// ── corpus ────────────────────────────────────────────────────────────────────

/// 20 documents across 5 topics (topic_id = doc_idx / 4).
const DOCS: &[&str] = &[
    // Topic 0: Machine Learning
    "Gradient descent optimizes neural network weights by minimizing loss functions through backpropagation.",
    "Deep learning models learn hierarchical representations by stacking multiple transformation layers.",
    "Training neural networks involves forward passes to compute predictions and backward passes to update parameters.",
    "Transformer architectures use self-attention mechanisms to capture long-range dependencies in sequences.",
    // Topic 1: Climate Change
    "Greenhouse gas emissions from fossil fuels trap heat in the atmosphere, causing global temperature rise.",
    "Carbon dioxide released by burning coal and oil is the primary driver of modern climate change.",
    "Deforestation reduces the planet's capacity to absorb CO2, accelerating the greenhouse effect.",
    "Rising sea levels and extreme weather events are direct consequences of anthropogenic climate change.",
    // Topic 2: Medicine / Immunology
    "White blood cells identify and destroy pathogens through antibody production and phagocytosis.",
    "The adaptive immune system generates specific antibodies that neutralize bacteria and viruses.",
    "Vaccines train the immune system to recognise pathogens before actual infection occurs.",
    "Inflammation is an immune response that recruits defensive cells to the site of injury or infection.",
    // Topic 3: History / World War I
    "Nationalism, imperialism, and the assassination of Archduke Franz Ferdinand triggered World War I in 1914.",
    "The system of military alliances in Europe caused a local conflict to escalate into a world war.",
    "Competition for colonies and resources among European empires created conditions for global conflict.",
    "Militarism and an arms race between European powers increased tensions before the First World War.",
    // Topic 4: Food Science / Bread
    "Bread is made by mixing flour, water, yeast, and salt, then allowing gluten to develop through kneading.",
    "Fermentation by yeast produces carbon dioxide bubbles that cause bread dough to rise before baking.",
    "The Maillard reaction during baking creates the brown crust and complex flavours in baked bread.",
    "Sourdough bread uses wild yeast and lactobacillus bacteria for slow fermentation and a distinct tangy flavour.",
];

/// One query per topic.
const QUERIES: &[&str] = &[
    "How do neural networks learn from training data?",
    "What are the main causes of global warming and climate change?",
    "How does the immune system defend against infections and pathogens?",
    "What caused the outbreak of the First World War?",
    "How is bread baked from wheat flour and yeast?",
];

// ── helpers ───────────────────────────────────────────────────────────────────

const CRISP_MODELS: &str =
    "Library/Application Support/com.christianstrobele.crispsorter/models";

fn models_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap()).join(CRISP_MODELS)
}

fn hf_snap(cache: &Path, model_code: &str) -> Option<PathBuf> {
    let dir_name = format!("models--{}", model_code.replace('/', "--"));
    let refs_main = cache.join(&dir_name).join("refs/main");
    let hash = fs::read_to_string(refs_main).ok()?;
    let snap = cache.join(dir_name).join("snapshots").join(hash.trim());
    snap.exists().then_some(snap)
}

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
    if na < 1e-9 || nb < 1e-9 { 0.0 } else { dot / (na * nb) }
}

/// For a single query, rank all docs by cosine similarity.
/// Returns sorted (doc_idx, score) descending.
fn rank(query_emb: &[f32], doc_embs: &[Vec<f32>]) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = doc_embs
        .iter()
        .enumerate()
        .map(|(i, d)| (i, cosine(query_emb, d)))
        .collect();
    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored
}

/// Relevant doc indices for query `q` (docs that share the topic).
fn relevant_for(q: usize) -> Vec<usize> {
    (0..4).map(|i| q * 4 + i).collect()
}

/// Mean Reciprocal Rank of the first relevant hit.
fn mrr(rankings: &[Vec<(usize, f32)>]) -> f32 {
    rankings
        .iter()
        .enumerate()
        .map(|(q, ranked)| {
            let rel = relevant_for(q);
            ranked
                .iter()
                .position(|(d, _)| rel.contains(d))
                .map(|pos| 1.0 / (pos as f32 + 1.0))
                .unwrap_or(0.0)
        })
        .sum::<f32>()
        / rankings.len() as f32
}

/// Mean Average Precision at K.
fn map_at_k(rankings: &[Vec<(usize, f32)>], k: usize) -> f32 {
    rankings
        .iter()
        .enumerate()
        .map(|(q, ranked)| {
            let rel = relevant_for(q);
            let top_k: Vec<_> = ranked.iter().take(k).collect();
            let mut hits = 0usize;
            let mut prec_sum = 0.0f32;
            for (rank_i, &(doc_idx, _)) in top_k.iter().enumerate() {
                if rel.contains(&doc_idx) {
                    hits += 1;
                    prec_sum += hits as f32 / (rank_i as f32 + 1.0);
                }
            }
            if rel.is_empty() {
                0.0
            } else {
                prec_sum / rel.len().min(k) as f32
            }
        })
        .sum::<f32>()
        / rankings.len() as f32
}

/// Average intra-cluster cosine similarity (within-topic pairs).
fn cohesion(doc_embs: &[Vec<f32>]) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0usize;
    for topic in 0..5 {
        let base = topic * 4;
        for i in base..base + 4 {
            for j in (i + 1)..base + 4 {
                total += cosine(&doc_embs[i], &doc_embs[j]);
                count += 1;
            }
        }
    }
    if count == 0 { 0.0 } else { total / count as f32 }
}

/// Average inter-cluster cosine similarity (cross-topic pairs).
fn separation(doc_embs: &[Vec<f32>]) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0usize;
    for i in 0..doc_embs.len() {
        for j in (i + 1)..doc_embs.len() {
            if i / 4 != j / 4 {
                total += cosine(&doc_embs[i], &doc_embs[j]);
                count += 1;
            }
        }
    }
    if count == 0 { 0.0 } else { total / count as f32 }
}

// ── per-model evaluation ──────────────────────────────────────────────────────

struct Metrics {
    label: &'static str,
    dim: usize,
    latency_ms: u64,
    mrr: f32,
    map5: f32,
    coh: f32,
    sep: f32,
}

impl Metrics {
    fn ratio(&self) -> f32 {
        if self.sep < 1e-6 { 0.0 } else { self.coh / self.sep }
    }
}

fn evaluate(label: &'static str, model: &mut TextEmbedding) -> Metrics {
    let t0 = Instant::now();
    let doc_embs = model.embed(DOCS.to_vec(), Some(DOCS.len())).expect("doc embed failed");
    let query_embs = model.embed(QUERIES.to_vec(), Some(QUERIES.len())).expect("query embed failed");
    let latency_ms = t0.elapsed().as_millis() as u64;

    let rankings: Vec<Vec<(usize, f32)>> = query_embs
        .iter()
        .map(|q| rank(q, &doc_embs))
        .collect();

    Metrics {
        label,
        dim: doc_embs[0].len(),
        latency_ms,
        mrr: mrr(&rankings),
        map5: map_at_k(&rankings, 5),
        coh: cohesion(&doc_embs),
        sep: separation(&doc_embs),
    }
}

fn print_table(rows: &[Metrics]) {
    println!();
    println!(
        "{:<28} {:>5}  {:>6}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}",
        "Model", "Dim", "ms", "MRR", "MAP@5", "Coh", "Sep", "Ratio"
    );
    println!("{}", "─".repeat(78));
    for m in rows {
        println!(
            "{:<28} {:>5}  {:>6}  {:>5.3}  {:>5.3}  {:>5.3}  {:>5.3}  {:>5.2}",
            m.label, m.dim, m.latency_ms, m.mrr, m.map5, m.coh, m.sep, m.ratio()
        );
    }
    println!();
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[test]
fn bench_all_local_models() {
    let dir = models_dir();
    let mut results: Vec<Metrics> = Vec::new();

    // ── Snowflake Arctic-L v2 ────────────────────────────────────────────────
    if hf_snap(&dir, "Snowflake/snowflake-arctic-embed-l-v2.0").is_some() {
        use fastembed::{EmbeddingModel, TextInitOptions};
        let mut model = TextEmbedding::try_new(
            TextInitOptions::new(EmbeddingModel::SnowflakeArcticEmbedLV2)
                .with_cache_dir(dir.clone())
                .with_show_download_progress(false),
        )
        .expect("failed to load SnowflakeArcticEmbedLV2");
        results.push(evaluate("Snowflake-Arctic-L-v2", &mut model));
    } else {
        println!("SKIPPED Snowflake Arctic-L v2");
    }

    // ── PIXIE-Rune-v1.0 ─────────────────────────────────────────────────────
    if let Some(snap) = hf_snap(&dir, "telepix/PIXIE-Rune-v1.0") {
        if let Some(tokenizer_files) = tok(&snap) {
            let onnx_path = snap.join("onnx/model.onnx");
            let model_def = UserDefinedEmbeddingModel::from_file(onnx_path, tokenizer_files)
                .with_pooling(Pooling::Mean);
            let mut model =
                TextEmbedding::try_new_from_user_defined(model_def, InitOptionsUserDefined::new())
                    .expect("failed to load PIXIE-Rune-v1.0");
            results.push(evaluate("PIXIE-Rune-v1.0", &mut model));
        }
    } else {
        println!("SKIPPED PIXIE-Rune-v1.0");
    }

    // ── Jina Embeddings v5 Nano ──────────────────────────────────────────────
    // Run twice: without prefixes (symmetric, all models on equal footing) and
    // with the recommended "query: " / "passage: " prefixes via embed_query() /
    // embed() using the new with_query_prefix / with_doc_prefix builders.
    if let Some(snap) = hf_snap(&dir, "jinaai/jina-embeddings-v5-text-nano-retrieval") {
        if let Some(tokenizer_files) = tok(&snap) {
            let onnx_path = snap.join("onnx/model_quantized.onnx");
            let model_def = UserDefinedEmbeddingModel::from_file(onnx_path, tokenizer_files)
                .with_pooling(Pooling::Cls)
                .with_output_key(fastembed::OutputKey::ByName("sentence_embedding"));
            let mut model =
                TextEmbedding::try_new_from_user_defined(model_def, InitOptionsUserDefined::new())
                    .expect("failed to load Jina v5 Nano");
            results.push(evaluate("Jina-v5-Nano (no prefix)", &mut model));

            // Also run with proper task prefixes using the new embed_query() API.
            if let Some(snap2) = hf_snap(&dir, "jinaai/jina-embeddings-v5-text-nano-retrieval") {
                if let Some(tf2) = tok(&snap2) {
                    let op2 = snap2.join("onnx/model_quantized.onnx");
                    let md2 = UserDefinedEmbeddingModel::from_file(op2, tf2)
                        .with_pooling(Pooling::Cls)
                        .with_output_key(fastembed::OutputKey::ByName("sentence_embedding"))
                        .with_query_prefix("query: ")
                        .with_doc_prefix("passage: ");
                    let mut m2 = TextEmbedding::try_new_from_user_defined(
                        md2,
                        InitOptionsUserDefined::new(),
                    )
                    .expect("failed to load Jina v5 Nano (prefixed)");

                    let t0 = Instant::now();
                    // embed() now prepends "passage: " automatically via doc_prefix.
                    let doc_embs = m2.embed(DOCS.to_vec(), Some(DOCS.len())).expect("doc embed");
                    // embed_query() prepends "query: " automatically.
                    let query_embs = m2.embed_query(QUERIES.to_vec(), Some(QUERIES.len())).expect("query embed");
                    let latency_ms = t0.elapsed().as_millis() as u64;

                    let rankings: Vec<Vec<(usize, f32)>> =
                        query_embs.iter().map(|q| rank(q, &doc_embs)).collect();

                    results.push(Metrics {
                        label: "Jina-v5-Nano (prefixed)",
                        dim: doc_embs[0].len(),
                        latency_ms,
                        mrr: mrr(&rankings),
                        map5: map_at_k(&rankings, 5),
                        coh: cohesion(&doc_embs),
                        sep: separation(&doc_embs),
                    });
                }
            }
        }
    } else {
        println!("SKIPPED Jina v5 Nano");
    }

    // ── Qwen3-Embedding-0.6B uint8 ───────────────────────────────────────────
    if let Some(snap) = hf_snap(&dir, "electroglyph/Qwen3-Embedding-0.6B-onnx-uint8") {
        if let Some(tokenizer_files) = tok(&snap) {
            let onnx_path = snap.join("dynamic_uint8.onnx");
            let model_def = UserDefinedEmbeddingModel::from_file(onnx_path, tokenizer_files)
                .with_pooling(Pooling::PrePooledU8 {
                    scale: 0.0027303685,
                    zero_point: 110,
                })
                .with_output_key(fastembed::OutputKey::ByName("sentence_embedding_quantized"));
            let mut model =
                TextEmbedding::try_new_from_user_defined(model_def, InitOptionsUserDefined::new())
                    .expect("failed to load Qwen3 uint8");
            results.push(evaluate("Qwen3-0.6B-uint8", &mut model));
        }
    } else {
        println!("SKIPPED Qwen3 uint8");
    }

    // ── Octen-Embedding-0.6B INT8 ────────────────────────────────────────────
    let octen_dir = models_dir().join("octen-embedding-0.6b-int8");
    if octen_dir.exists() {
        if let Some(tokenizer_files) = tok(&octen_dir) {
            let onnx_path = octen_dir.join("model.int8.onnx");
            let model_def = UserDefinedEmbeddingModel::from_file(onnx_path, tokenizer_files)
                .with_pooling(Pooling::LastToken);
            let mut model =
                TextEmbedding::try_new_from_user_defined(model_def, InitOptionsUserDefined::new())
                    .expect("failed to load Octen INT8");
            results.push(evaluate("Octen-0.6B-INT8", &mut model));
        }
    } else {
        println!("SKIPPED Octen INT8");
    }

    // ── print comparison table ────────────────────────────────────────────────
    if results.is_empty() {
        println!("No models found in local cache — nothing to benchmark.");
        return;
    }

    print_table(&results);

    println!("Corpus: {} documents  |  {} queries  |  4 relevant docs per query", DOCS.len(), QUERIES.len());

    // Assert at least one model achieved meaningful retrieval (MRR > 0.3).
    let best_mrr = results.iter().map(|m| m.mrr).fold(0.0f32, f32::max);
    assert!(
        best_mrr > 0.3,
        "All models failed basic retrieval (best MRR = {best_mrr:.3})"
    );
}
