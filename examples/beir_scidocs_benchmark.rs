use anyhow::Result;
use fastembed::{
    LateInteractionInitOptions, LateInteractionModel, LateInteractionTextEmbedding, Muvera,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use ort::execution_providers::{CUDAExecutionProvider, CPUExecutionProvider};

/// Compute ColBERT MaxSim score between query and document embeddings
fn maxsim(query_emb: &[Vec<f32>], doc_emb: &[Vec<f32>]) -> f32 {
    query_emb
        .iter()
        .map(|q| {
            doc_emb
                .iter()
                .map(|d| q.iter().zip(d.iter()).map(|(a, b)| a * b).sum::<f32>())
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

/// Compute dot product between two vectors
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Compute recall@k
fn recall_at_k(relevant: &[String], retrieved: &[String], k: usize) -> f32 {
    let top_k = &retrieved[..k.min(retrieved.len())];
    let hits = top_k.iter().filter(|id| relevant.contains(id)).count();
    hits as f32 / relevant.len().max(1) as f32
}

/// Load corpus from BEIR jsonl format
fn load_corpus(path: &Path) -> Result<(Vec<String>, Vec<String>)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    
    let mut ids = Vec::new();
    let mut texts = Vec::new();
    
    for line in reader.lines() {
        let line = line?;
        let json: serde_json::Value = serde_json::from_str(&line)?;
        let id = json["_id"].as_str().unwrap_or("").to_string();
        let title = json["title"].as_str().unwrap_or("");
        let text = json["text"].as_str().unwrap_or("");
        let combined = if title.is_empty() {
            text.to_string()
        } else {
            format!("{} {}", title, text)
        };
        ids.push(id);
        texts.push(combined);
    }
    
    Ok((ids, texts))
}

/// Load queries from BEIR jsonl format
fn load_queries(path: &Path) -> Result<(Vec<String>, Vec<String>)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    
    let mut ids = Vec::new();
    let mut texts = Vec::new();
    
    for line in reader.lines() {
        let line = line?;
        let json: serde_json::Value = serde_json::from_str(&line)?;
        let id = json["_id"].as_str().unwrap_or("").to_string();
        let text = json["text"].as_str().unwrap_or("").to_string();
        ids.push(id);
        texts.push(text);
    }
    
    Ok((ids, texts))
}

/// Load qrels (relevance judgments) from TSV format
fn load_qrels(path: &Path) -> Result<HashMap<String, Vec<String>>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut qrels: HashMap<String, Vec<String>> = HashMap::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 && line.starts_with("query-id") {
            continue; // Skip header
        }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 3 {
            let query_id = parts[0].to_string();
            let doc_id = parts[1].to_string();
            let relevance: i32 = parts[2].parse().unwrap_or(0);
            if relevance > 0 {
                qrels.entry(query_id).or_default().push(doc_id);
            }
        }
    }

    Ok(qrels)
}

fn main() -> Result<()> {
    // Configuration
    let dataset_path = Path::new("datasets/scidocs"); // Adjust path as needed
    let batch_size = 8;
    let top_n_candidates = 100; // Number of candidates to retrieve with MUVERA
    
    // MUVERA parameter configurations: (r_reps, k_sim, dim_proj)
    let muvera_configs = vec![
        (20, 3, 8),   // 1280-dim
        (20, 4, 8),   // 2560-dim
        (20, 5, 8),   // 5120-dim
        (20, 5, 16),  // 10240-dim
        (30, 5, 16),  // 15360-dim
        (40, 5, 16),  // 20480-dim
    ];

    // 1. Load dataset
    println!("Loading SciDocs dataset...");
    let (corpus_ids, corpus_texts) = load_corpus(&dataset_path.join("corpus.jsonl"))?;
    let (query_ids, query_texts) = load_queries(&dataset_path.join("queries.jsonl"))?;
    let qrels = load_qrels(&dataset_path.join("qrels/test.tsv"))?;
    
    println!("Corpus size: {}", corpus_ids.len());
    println!("Query count: {}", query_ids.len());
    println!("Queries with relevance judgments: {}", qrels.len());

    // 2. Initialize ColBERT model
    println!("\nLoading ColBERT model...");

    let mut model = LateInteractionTextEmbedding::try_new(
    LateInteractionInitOptions::new(LateInteractionModel::ColBERTV2)
        .with_execution_providers(vec![
            CUDAExecutionProvider::default().build().into(),
            CPUExecutionProvider::default().build().into(),
        ])
    )?;

    // 3. Embed corpus
    println!("Embedding corpus...");
    let mut corpus_embeddings = Vec::with_capacity(corpus_texts.len());
    for (i, text) in corpus_texts.iter().enumerate() {
        if i % 1000 == 0 {
            println!("  Embedded {}/{} documents", i, corpus_texts.len());
        }
        let emb = model.embed(&[text], None)?;
        corpus_embeddings.push(emb.into_iter().next().unwrap());
    }

    // 4. Embed queries
    println!("Embedding queries...");
    let mut query_embeddings = Vec::with_capacity(query_texts.len());
    for (i, text) in query_texts.iter().enumerate() {
        if i % 100 == 0 {
            println!("  Embedded {}/{} queries", i, query_texts.len());
        }
        let emb = model.query_embed(&[text], None)?;
        query_embeddings.push(emb.into_iter().next().unwrap());
    }

    // // 5. Evaluate brute-force ColBERT (baseline)
    // println!("\n--- Evaluating brute-force ColBERT ---");
    // let mut recalls_4 = Vec::new();
    // let mut recalls_5 = Vec::new();
    // let mut recalls_10 = Vec::new();
    // 
    // for (q_idx, query_id) in query_ids.iter().enumerate() {
    //     if let Some(relevant_docs) = qrels.get(query_id) {
    //         // Score all documents with MaxSim
    //         let mut scores: Vec<(usize, f32)> = corpus_embeddings
    //             .iter()
    //             .enumerate()
    //             .map(|(d_idx, doc_emb)| (d_idx, maxsim(&query_embeddings[q_idx], doc_emb)))
    //             .collect();
    //         
    //         // Sort by score descending
    //         scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    //         
    //         // Get top-k document IDs
    //         let retrieved: Vec<String> = scores
    //             .iter()
    //             .take(10)
    //             .map(|(idx, _)| corpus_ids[*idx].clone())
    //             .collect();
    //         
    //         recalls_4.push(recall_at_k(relevant_docs, &retrieved, 4));
    //         recalls_5.push(recall_at_k(relevant_docs, &retrieved, 5));
    //         recalls_10.push(recall_at_k(relevant_docs, &retrieved, 10));
    //     }
    // }
    // 
    // let avg_r4 = recalls_4.iter().sum::<f32>() / recalls_4.len() as f32;
    // let avg_r5 = recalls_5.iter().sum::<f32>() / recalls_5.len() as f32;
    // let avg_r10 = recalls_10.iter().sum::<f32>() / recalls_10.len() as f32;
    // 
    // println!(
    //     "Vector: colbert, Recall@4: {:.4}, Recall@5: {:.4}, Recall@10: {:.4}",
    //     avg_r4, avg_r5, avg_r10
    // );

    // 6. Evaluate MUVERA configurations
    for (r_reps, k_sim, dim_proj) in &muvera_configs {
        let muvera = Muvera::from_late_interaction_model(
            &model,
            Some(*k_sim),
            Some(*dim_proj),
            Some(*r_reps),
            Some(42),
        )?;
        
        let embedding_size = muvera.embedding_size();
        println!("\n--- Evaluating MUVERA-{} (r={}, k={}, d={}) ---", 
                 embedding_size, r_reps, k_sim, dim_proj);
        
        // Convert corpus to FDEs
        let corpus_fdes: Vec<Vec<f32>> = corpus_embeddings
            .iter()
            .map(|emb| muvera.process_document(emb))
            .collect();
        
        // Convert queries to FDEs
        let query_fdes: Vec<Vec<f32>> = query_embeddings
            .iter()
            .map(|emb| muvera.process_query(emb))
            .collect();
        
        let mut recalls_4 = Vec::new();
        let mut recalls_5 = Vec::new();
        let mut recalls_10 = Vec::new();

        for (q_idx, query_id) in query_ids.iter().enumerate() {
            if let Some(relevant_docs) = qrels.get(query_id) {
                // Just MUVERA dot product - NO reranking
                let mut scores: Vec<(usize, f32)> = corpus_fdes
                    .iter()
                    .enumerate()
                    .map(|(d_idx, doc_fde)| (d_idx, dot(&query_fdes[q_idx], doc_fde)))
                    .collect();

                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                let retrieved: Vec<String> = scores
                    .iter()
                    .take(10)
                    .map(|(idx, _)| corpus_ids[*idx].clone())
                    .collect();

                recalls_4.push(recall_at_k(relevant_docs, &retrieved, 4));
                recalls_5.push(recall_at_k(relevant_docs, &retrieved, 5));
                recalls_10.push(recall_at_k(relevant_docs, &retrieved, 10));
            }
        }
        
        // for (q_idx, query_id) in query_ids.iter().enumerate() {
        //     if let Some(relevant_docs) = qrels.get(query_id) {
        //         // Stage 1: ANN candidate retrieval using MUVERA dot product
        //         let mut fde_scores: Vec<(usize, f32)> = corpus_fdes
        //             .iter()
        //             .enumerate()
        //             .map(|(d_idx, doc_fde)| (d_idx, dot(&query_fdes[q_idx], doc_fde)))
        //             .collect();
        //         
        //         fde_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        //         
        //         let candidates: Vec<usize> = fde_scores
        //             .iter()
        //             .take(top_n_candidates)
        //             .map(|(idx, _)| *idx)
        //             .collect();
        //         
        //         // Stage 2: Rerank candidates with ColBERT MaxSim
        //         let mut reranked: Vec<(usize, f32)> = candidates
        //             .iter()
        //             .map(|&d_idx| (d_idx, maxsim(&query_embeddings[q_idx], &corpus_embeddings[d_idx])))
        //             .collect();
        //         
        //         reranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        //         
        //         let retrieved: Vec<String> = reranked
        //             .iter()
        //             .take(10)
        //             .map(|(idx, _)| corpus_ids[*idx].clone())
        //             .collect();
        //         
        //         recalls_4.push(recall_at_k(relevant_docs, &retrieved, 4));
        //         recalls_5.push(recall_at_k(relevant_docs, &retrieved, 5));
        //         recalls_10.push(recall_at_k(relevant_docs, &retrieved, 10));
        //     }
        // }
        
        let avg_r4 = recalls_4.iter().sum::<f32>() / recalls_4.len() as f32;
        let avg_r5 = recalls_5.iter().sum::<f32>() / recalls_5.len() as f32;
        let avg_r10 = recalls_10.iter().sum::<f32>() / recalls_10.len() as f32;
        
        println!(
            "Vector: muvera-{}, Recall@4: {:.4}, Recall@5: {:.4}, Recall@10: {:.4}",
            embedding_size, avg_r4, avg_r5, avg_r10
        );
    }

    Ok(())
}
