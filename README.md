<div align="center">
  <h1><a href="https://crates.io/crates/fastembed">FastEmbed-rs</a></h1>
  <h3>A Rust implementation of @Qdrant/fastembed.</h3>
  <a href="https://crates.io/crates/fastembed"><img src="https://img.shields.io/crates/v/fastembed.svg" alt="Crates.io"></a>
  <a href="https://github.com/Anush008/fastembed-rs/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-mit-blue.svg" alt="MIT Licensed"></a>
  <a href="https://github.com/Anush008/fastembed-rs/actions/workflows/release.yml"><img src="https://github.com/Anush008/fastembed-rs/actions/workflows/release.yml/badge.svg?branch=main" alt="Semantic release"></a>
</div>

## 🍕 Features
* Supports synchronous usage. No dependency on Tokio.
* Uses [@huggingface/tokenizers](https://github.com/huggingface/tokenizers) for blazing-fast encodings.
* Supports batch embedddings with parallelism using Rayon.

The default embedding supports "query" and "passage" prefixes for the input text. The default model is Flag Embedding, which is top of the [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard.

## 🤖 Models

- [**BAAI/bge-base-en-v1.5**](https://huggingface.co/BAAI/bge-base-en-v1.5)
- [**BAAI/bge-small-en-v1.5**](https://huggingface.co/BAAI/bge-small-en-v1.5) - Default
- [**sentence-transformers/all-MiniLM-L12-v2**](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
- [**intfloat/multilingual-e5-large**](https://huggingface.co/intfloat/multilingual-e5-large)


## 🚀 Installation

To install the FastEmbed library, Cargo works: 

```bash
cargo install fastembed
```

## 📖 Usage

```rust
use fastembed::{FlagEmbedding, InitOptions, EmbeddingModel, EmbeddingBase};

// With default InitOptions
let model: FlagEmbedding = FlagEmbedding::try_new(Default::default())?;

// With custom InitOptions
let model: FlagEmbedding = FlagEmbedding::try_new(InitOptions {
    model_name: EmbeddingModel::BGEBaseEN,
    show_download_message: false,
    ..Default::default()
})?;

let documents = vec![
    "passage: Hello, World!",
    "query: Hello, World!",
    "passage: This is an example passage.",
    // You can leave out the prefix but it's recommended
    "fastembed-rs is licensed under MIT"
    ];

 // Generate embeddings with the default batch size, 256
 let embeddings = model.embed(documents, None)?;

 println!("Embeddings length: {}", embeddings.len()); // -> Embeddings length: 4
```

### Supports passage and query embeddings for more accurate results
```rust
 // Generate embeddings for the passages
 // The texts are prefixed with "passage" for better results
 // The batch size is set to 1 for demonstration purposes
 let passages = vec![
     "This is the first passage. It contains provides more context for retrieval.",
     "Here's the second passage, which is longer than the first one. It includes additional information.",
     "And this is the third passage, the longest of all. It contains several sentences and is meant for more extensive testing."
    ];

 let embeddings = model.passage_embed(passages, Some(1))?;

 println!("Passage embeddings length: {}", embeddings.len()); // -> Embeddings length: 3

 // Generate embeddings for the query
 // The text is prefixed with "query" for better retrieval
 let query = "What is the answer to this generic question?";

 let query_embedding = model.query_embed(query)?;

 println!("Query embedding dimension: {}", query_embedding.len()); // -> Query embedding dimension: 768
```

## 🚒 Under the hood

### Why fast?

It's important we justify the "fast" in FastEmbed. FastEmbed is fast because:

1. Quantized model weights
2. ONNX Runtime which allows for inference on CPU, GPU, and other dedicated runtimes

### Why light?
1. No hidden dependencies via Huggingface Transformers

### Why accurate?
1. Better than OpenAI Ada-002
2. Top of the Embedding leaderboards e.g. [MTEB](https://huggingface.co/spaces/mteb/leaderboard)
