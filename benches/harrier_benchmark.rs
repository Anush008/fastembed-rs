//! Comprehensive benchmark: Microsoft Harrier OSS v1 270M vs BAAI/bge-small-en-v1.5
//!
//! Measures:
//!   - Throughput (texts/sec) and latency for both models
//!   - Semantic coherence within each model (similar vs dissimilar pairs)
//!   - Quantization impact: FP32 vs INT8 Harrier cosine similarity on same texts
//!   - Statistical accuracy via multiple timing runs

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("==========================================================");
    println!(" Harrier OSS v1 270M vs BGE-Small-EN-v1.5  |  Benchmark");
    println!("==========================================================\n");

    // ── Dataset ──────────────────────────────────────────────────────────────
    // Ten documents covering short to long, and topically diverse.
    let documents: Vec<&str> = vec![
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world in unprecedented ways.",
        "Machine learning models require careful optimisation for production deployment. \
         The process involves multiple stages: data preprocessing, model training, \
         hyperparameter tuning, and inference optimisation.",
        "Natural language processing enables computers to understand human communication \
         through algorithms that parse grammar, semantics, and contextual meaning.",
        "Transformer architectures revolutionised deep learning. Self-attention mechanisms \
         let them process sequential data far more effectively than recurrent networks.",
        "Efficient neural network design balances computational complexity with accuracy. \
         Knowledge distillation, pruning, and quantisation reduce resource requirements.",
        "Cloud platforms provide scalable infrastructure for ML workloads. AWS SageMaker, \
         Google Vertex AI, and Azure ML offer managed training and deployment services.",
        "Data privacy is critical in modern AI. Differential privacy, federated learning, \
         and secure multi-party computation protect sensitive information.",
        "CI/CD for ML requires automated testing, model versioning, and production \
         metric monitoring to maintain reliable system performance.",
        "Responsible AI development demands attention to bias, fairness, transparency, \
         and accountability throughout the full model lifecycle.",
    ];

    // Semantic pairs for coherence testing (indices into `documents` above).
    // high_sim: similar topic → should give high cosine similarity.
    // low_sim:  dissimilar topic → should give low cosine similarity.
    let high_sim_pairs: &[(usize, usize)] = &[(1, 2), (3, 4), (5, 6)]; // AI/ML pairs
    let low_sim_pairs: &[(usize, usize)] = &[(0, 7), (0, 8), (1, 9)]; // fox vs privacy/CI/ethics

    println!("Dataset : {} documents", documents.len());
    println!("Timing  : 5 warmup runs + 10 measurement runs per model\n");

    // ── Initialise models ─────────────────────────────────────────────────────
    println!("Loading BGE-Small-EN-v1.5 …");
    let mut bge_small = TextEmbedding::try_new(
        TextInitOptions::new(EmbeddingModel::BGESmallENV15).with_show_download_progress(true),
    )?;

    println!("Loading Harrier OSS v1 270M (FP32) …");
    let mut harrier_fp32 = TextEmbedding::try_new(
        TextInitOptions::new(EmbeddingModel::HarrierOSSV1_270M).with_show_download_progress(true),
    )?;

    println!("Loading Harrier OSS v1 270M (INT8) …");
    let mut harrier_int8 = TextEmbedding::try_new(
        TextInitOptions::new(EmbeddingModel::HarrierOSSV1_270MQ).with_show_download_progress(true),
    )?;

    println!();

    // ── Timing benchmark ──────────────────────────────────────────────────────
    const WARMUP: usize = 5;
    const RUNS: usize = 10;

    let (bge_embs, bge_stats) = timed_embed(&mut bge_small, &documents, WARMUP, RUNS)?;
    let (harrier_fp32_embs, harrier_fp32_stats) =
        timed_embed(&mut harrier_fp32, &documents, WARMUP, RUNS)?;
    let (harrier_int8_embs, harrier_int8_stats) =
        timed_embed(&mut harrier_int8, &documents, WARMUP, RUNS)?;

    let bge_dim = bge_embs[0].len();
    let harrier_dim = harrier_fp32_embs[0].len();

    // ── Results table ─────────────────────────────────────────────────────────
    println!("==========================================================");
    println!(" Throughput & Latency  ({} docs per run, {} runs)", documents.len(), RUNS);
    println!("==========================================================");
    println!(
        "{:<32} {:>8}  {:>9}  {:>9}  {:>9}  {:>10}",
        "Model", "Dim", "Min ms", "Mean ms", "Max ms", "texts/s"
    );
    println!("{}", "-".repeat(82));
    print_row("BGE-Small-EN-v1.5", bge_dim, &bge_stats, documents.len());
    print_row("Harrier FP32", harrier_dim, &harrier_fp32_stats, documents.len());
    print_row("Harrier INT8", harrier_dim, &harrier_int8_stats, documents.len());
    println!();

    let speed_ratio_fp32 = bge_stats.mean_ms / harrier_fp32_stats.mean_ms;
    let speed_ratio_int8 = bge_stats.mean_ms / harrier_int8_stats.mean_ms;
    println!(
        "Harrier FP32 is {:.2}x {} than BGE-Small (mean latency)",
        speed_ratio_fp32.abs(),
        if speed_ratio_fp32 >= 1.0 { "faster" } else { "slower" }
    );
    println!(
        "Harrier INT8 is {:.2}x {} than BGE-Small (mean latency)",
        speed_ratio_int8.abs(),
        if speed_ratio_int8 >= 1.0 { "faster" } else { "slower" }
    );
    println!();

    // ── Semantic coherence ────────────────────────────────────────────────────
    println!("==========================================================");
    println!(" Semantic Coherence (within-model cosine similarity)");
    println!("==========================================================");

    for (label, embs) in [
        ("BGE-Small-EN-v1.5", &bge_embs),
        ("Harrier FP32      ", &harrier_fp32_embs),
        ("Harrier INT8      ", &harrier_int8_embs),
    ] {
        let avg_hi = avg_cosine_sim(embs, high_sim_pairs);
        let avg_lo = avg_cosine_sim(embs, low_sim_pairs);
        println!(
            "{} | similar pairs avg cos: {:.4}  |  dissimilar avg cos: {:.4}  |  delta: {:.4}",
            label,
            avg_hi,
            avg_lo,
            avg_hi - avg_lo
        );
    }
    println!(
        "\n  (larger delta = better discrimination between similar and dissimilar texts)"
    );
    println!();

    // ── Quantisation fidelity (FP32 vs INT8 Harrier) ─────────────────────────
    println!("==========================================================");
    println!(" Harrier FP32 vs INT8 Quantisation Fidelity");
    println!("==========================================================");
    let mut fidelity_scores: Vec<f32> = Vec::new();
    for (i, (fp32_emb, int8_emb)) in harrier_fp32_embs.iter().zip(harrier_int8_embs.iter()).enumerate() {
        let sim = cosine_similarity(fp32_emb, int8_emb);
        fidelity_scores.push(sim);
        if i < 3 {
            println!("  doc {:2}: cosine(FP32, INT8) = {:.5}", i, sim);
        }
    }
    let avg_fidelity: f32 = fidelity_scores.iter().sum::<f32>() / fidelity_scores.len() as f32;
    let min_fidelity = fidelity_scores.iter().cloned().fold(f32::INFINITY, f32::min);
    println!("  …");
    println!("  avg fidelity across all docs : {:.5}", avg_fidelity);
    println!("  min fidelity across all docs : {:.5}", min_fidelity);
    println!("  (values near 1.0 mean INT8 is nearly identical to FP32)\n");

    // ── Save markdown report ──────────────────────────────────────────────────
    save_report(
        documents.len(),
        RUNS,
        bge_dim,
        harrier_dim,
        &bge_stats,
        &harrier_fp32_stats,
        &harrier_int8_stats,
        avg_fidelity,
        avg_cosine_sim(&bge_embs, high_sim_pairs),
        avg_cosine_sim(&bge_embs, low_sim_pairs),
        avg_cosine_sim(&harrier_fp32_embs, high_sim_pairs),
        avg_cosine_sim(&harrier_fp32_embs, low_sim_pairs),
    )?;

    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

struct TimingStats {
    min_ms: f64,
    mean_ms: f64,
    max_ms: f64,
}

/// Embed `documents` `warmup + runs` times, discard the warmup, return the last
/// set of embeddings along with timing statistics over the measured runs.
fn timed_embed(
    model: &mut TextEmbedding,
    documents: &[&str],
    warmup: usize,
    runs: usize,
) -> Result<(Vec<Vec<f32>>, TimingStats), Box<dyn std::error::Error>> {
    // Warmup — fill caches, avoid cold-start skew.
    for _ in 0..warmup {
        model.embed(documents.to_vec(), None)?;
    }

    let mut durations: Vec<Duration> = Vec::with_capacity(runs);
    let mut last_embs: Vec<Vec<f32>> = Vec::new();
    for _ in 0..runs {
        let t = Instant::now();
        last_embs = model.embed(documents.to_vec(), None)?;
        durations.push(t.elapsed());
    }

    let ms: Vec<f64> = durations.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
    let min_ms = ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_ms = ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean_ms = ms.iter().sum::<f64>() / ms.len() as f64;

    Ok((last_embs, TimingStats { min_ms, mean_ms, max_ms }))
}

fn print_row(label: &str, dim: usize, stats: &TimingStats, n_docs: usize) {
    let throughput = n_docs as f64 / (stats.mean_ms / 1000.0);
    println!(
        "{:<32} {:>8}  {:>9.1}  {:>9.1}  {:>9.1}  {:>10.1}",
        label, dim, stats.min_ms, stats.mean_ms, stats.max_ms, throughput
    );
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 { 0.0 } else { dot / (mag_a * mag_b) }
}

fn avg_cosine_sim(embs: &[Vec<f32>], pairs: &[(usize, usize)]) -> f32 {
    if pairs.is_empty() {
        return 0.0;
    }
    let sum: f32 = pairs
        .iter()
        .map(|&(i, j)| cosine_similarity(&embs[i], &embs[j]))
        .sum();
    sum / pairs.len() as f32
}

fn save_report(
    n_docs: usize,
    runs: usize,
    bge_dim: usize,
    harrier_dim: usize,
    bge: &TimingStats,
    harrier_fp32: &TimingStats,
    harrier_int8: &TimingStats,
    quant_fidelity: f32,
    bge_sim_hi: f32,
    bge_sim_lo: f32,
    harrier_sim_hi: f32,
    harrier_sim_lo: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    use chrono::Utc;
    use std::fs::File;
    use std::io::Write;

    let ts = Utc::now().to_rfc3339();
    let content = format!(
        "# Harrier OSS v1 270M vs BGE-Small-EN-v1.5 — Benchmark Report
Timestamp : {ts}
Documents : {n_docs}  |  Runs : {runs} (after 5 warmup)

## Throughput & Latency

| Model | Dim | Min ms | Mean ms | Max ms | texts/s |
|-------|-----|--------|---------|--------|---------|
| BGE-Small-EN-v1.5 | {bge_dim} | {bge_min:.1} | {bge_mean:.1} | {bge_max:.1} | {bge_tps:.1} |
| Harrier FP32 | {harrier_dim} | {fp32_min:.1} | {fp32_mean:.1} | {fp32_max:.1} | {fp32_tps:.1} |
| Harrier INT8 | {harrier_dim} | {int8_min:.1} | {int8_mean:.1} | {int8_max:.1} | {int8_tps:.1} |

## Semantic Coherence (within-model cosine similarity)

| Model | Similar-pair avg | Dissimilar-pair avg | Delta |
|-------|-----------------|---------------------|-------|
| BGE-Small | {bge_sim_hi:.4} | {bge_sim_lo:.4} | {bge_delta:.4} |
| Harrier FP32 | {harrier_sim_hi:.4} | {harrier_sim_lo:.4} | {harrier_delta:.4} |

## Quantisation Fidelity (FP32 vs INT8 Harrier)

Average cosine similarity between FP32 and INT8 embeddings for the same texts: **{quant_fidelity:.5}**

_(1.0 = identical; values above 0.99 indicate negligible quality loss from quantisation)_
",
        ts = ts,
        n_docs = n_docs,
        runs = runs,
        bge_dim = bge_dim,
        bge_min = bge.min_ms,
        bge_mean = bge.mean_ms,
        bge_max = bge.max_ms,
        bge_tps = n_docs as f64 / (bge.mean_ms / 1000.0),
        harrier_dim = harrier_dim,
        fp32_min = harrier_fp32.min_ms,
        fp32_mean = harrier_fp32.mean_ms,
        fp32_max = harrier_fp32.max_ms,
        fp32_tps = n_docs as f64 / (harrier_fp32.mean_ms / 1000.0),
        int8_min = harrier_int8.min_ms,
        int8_mean = harrier_int8.mean_ms,
        int8_max = harrier_int8.max_ms,
        int8_tps = n_docs as f64 / (harrier_int8.mean_ms / 1000.0),
        bge_sim_hi = bge_sim_hi,
        bge_sim_lo = bge_sim_lo,
        bge_delta = bge_sim_hi - bge_sim_lo,
        harrier_sim_hi = harrier_sim_hi,
        harrier_sim_lo = harrier_sim_lo,
        harrier_delta = harrier_sim_hi - harrier_sim_lo,
        quant_fidelity = quant_fidelity,
    );

    let mut f = File::create("benchmark_results.md")?;
    f.write_all(content.as_bytes())?;
    println!("Report saved to benchmark_results.md");
    Ok(())
}
