<div align="center">
  <h1><a href="https://crates.io/crates/fastembed">FastEmbed-rs 🦀</a></h1>
 <h3>Rust implementation of <a href="https://github.com/qdrant/fastembed" target="_blank">@qdrant/fastembed</a></h3>
  <a href="https://crates.io/crates/fastembed"><img src="https://img.shields.io/crates/v/fastembed.svg" alt="Crates.io"></a>
  <a href="https://github.com/Anush008/fastembed-rs/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-apache-blue.svg" alt="MIT Licensed"></a>
  <a href="https://github.com/Anush008/fastembed-rs/actions/workflows/release.yml"><img src="https://github.com/Anush008/fastembed-rs/actions/workflows/release.yml/badge.svg?branch=main" alt="Semantic release"></a>
</div>

## 🍕 Features

- Supports synchronous usage. No dependency on Tokio.
- Uses [@pykeio/ort](https://github.com/pykeio/ort) for performant ONNX inference.
- Uses [@huggingface/tokenizers](https://github.com/huggingface/tokenizers) for fast encodings.
- Supports batch embeddings generation with parallelism using [@rayon-rs/rayon](https://github.com/rayon-rs/rayon).

The default model is Flag Embedding, which is top of the [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard.

## 🔍 Not looking for Rust?

- Python 🐍: [fastembed](https://github.com/qdrant/fastembed)
- Go 🐳: [fastembed-go](https://github.com/Anush008/fastembed-go)
- JavaScript 🌐: [fastembed-js](https://github.com/Anush008/fastembed-js)

## 🤖 Models

### Text Embedding

- [**BAAI/bge-base-en-v1.5**](https://huggingface.co/BAAI/bge-base-en-v1.5)
- [**BAAI/bge-small-en-v1.5**](https://huggingface.co/BAAI/bge-small-en-v1.5) - Default
- [**BAAI/bge-large-en-v1.5**](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [**BAAI/bge-small-zh-v1.5**](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- [**sentence-transformers/all-MiniLM-L6-v2**](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [**sentence-transformers/all-MiniLM-L12-v2**](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
- [**sentence-transformers/paraphrase-MiniLM-L12-v2**](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L12-v2)
- [**sentence-transformers/paraphrase-multilingual-mpnet-base-v2**](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
- [**nomic-ai/nomic-embed-text-v1**](https://huggingface.co/nomic-ai/nomic-embed-text-v1)
- [**nomic-ai/nomic-embed-text-v1.5**](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [**intfloat/multilingual-e5-small**](https://huggingface.co/intfloat/multilingual-e5-small)
- [**intfloat/multilingual-e5-base**](https://huggingface.co/intfloat/multilingual-e5-base)
- [**intfloat/multilingual-e5-large**](https://huggingface.co/intfloat/multilingual-e5-large)
- [**mixedbread-ai/mxbai-embed-large-v1**](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)
- [**Alibaba-NLP/gte-base-en-v1.5**](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5)

## Sparse vectors

- [**prithivida/Splade_PP_en_v1**](https://huggingface.co/prithivida/Splade_PP_en_v1) - Default

### Reranking

- [**BAAI/bge-reranker-base**](https://huggingface.co/BAAI/bge-reranker-base)
- [**jinaai/jina-reranker-v1-turbo-en**](https://huggingface.co/jinaai/jina-reranker-v1-turbo-en)
- [**jinaai/jina-reranker-v2-base-multiligual**](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)

## 🚀 Installation

Run the following command in your project directory:

```bash
cargo add fastembed
```

Or add the following line to your Cargo.toml:

```toml
[dependencies]
fastembed = "3"
```

## 📖 Usage

### Generating Text Embeddings

```rust
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};

// With default InitOptions
let model = TextEmbedding::try_new(Default::default())?;

// With custom InitOptions
let model = TextEmbedding::try_new(InitOptions {
    model_name: EmbeddingModel::AllMiniLML6V2,
    show_download_progress: true,
    ..Default::default()
})?;

let documents = vec![
    "passage: Hello, World!",
    "query: Hello, World!",
    "passage: This is an example passage.",
    // You can leave out the prefix but it's recommended
    "fastembed-rs is licensed under Apache  2.0"
    ];

 // Generate embeddings with the default batch size, 256
 let embeddings = model.embed(documents, None)?;

 println!("Embeddings length: {}", embeddings.len()); // -> Embeddings length: 4
 println!("Embedding dimension: {}", embeddings[0].len()); // -> Embedding dimension: 384

```

### Candidates Reranking

```rust
use fastembed::{TextRerank, RerankInitOptions, RerankerModel};

let model = TextRerank::try_new(RerankInitOptions {
    model_name: RerankerModel::BGERerankerBase,
    show_download_progress: true,
    ..Default::default()
})
.unwrap();

let documents = vec![
    "hi",
    "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear, is a bear species endemic to China.",
    "panda is animal",
    "i dont know",
    "kind of mammal",
];

// Rerank with the default batch size
let results = model.rerank("what is panda?", documents, true, None);
println!("Rerank result: {:?}", results);
```

Alternatively, raw `.onnx` files can be loaded through the `UserDefinedEmbeddingModel` struct (for "bring your own" text embedding models) using `TextEmbedding::try_new_from_user_defined(...)`. Similarly, "bring your own" reranking models can be loaded using the `UserDefinedRerankingModel` struct and `TextRerank::try_new_from_user_defined(...)`. For example:

```rust
macro_rules! local_model {
    ($folder:literal) => {
        UserDefinedEmbeddingModel {
            onnx_file: include_bytes!(concat!($folder, "/model.onnx")).to_vec(),
            tokenizer_files: TokenizerFiles {
                tokenizer_file: include_bytes!(concat!($folder, "/tokenizer.json")).to_vec(),
                config_file: include_bytes!(concat!($folder, "/config.json")).to_vec(),
                special_tokens_map_file: include_bytes!(concat!($folder, "/special_tokens_map.json")).to_vec(),
                tokenizer_config_file: include_bytes!(concat!($folder, "/tokenizer_config.json")).to_vec(),
            },
        }
    };
}

let user_def_model_data = local_model!("path/to/model");
let user_def_model = TextEmbedding::try_new_from_user_defined(user_def_model, Default::default()).unwrap();
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

## 📄 LICENSE

Apache 2.0 © [2024](https://github.com/Anush008/fastembed-rs/blob/main/LICENSE)
