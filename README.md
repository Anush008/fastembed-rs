<div align="center">
  <h1><a href="https://crates.io/crates/fastembed">FastEmbed-rs 🦀</a></h1>
 <h3>Rust library for generating vector embeddings, reranking locally!</h3>
  <a href="https://crates.io/crates/fastembed"><img src="https://img.shields.io/crates/v/fastembed.svg" alt="Crates.io"></a>
  <a href="https://github.com/Anush008/fastembed-rs/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-apache-blue.svg" alt="MIT Licensed"></a>
  <a href="https://github.com/Anush008/fastembed-rs/actions/workflows/release.yml"><img src="https://github.com/Anush008/fastembed-rs/actions/workflows/release.yml/badge.svg?branch=main" alt="Semantic release"></a>
</div>

## 🍕 Features

- Supports synchronous usage. No dependency on Tokio.
- Uses [@pykeio/ort](https://github.com/pykeio/ort) for performant ONNX inference.
- Uses [@huggingface/tokenizers](https://github.com/huggingface/tokenizers) for fast encodings.

## 🔍 Not looking for Rust?

- Python 🐍: [fastembed](https://github.com/qdrant/fastembed)
- Go 🐳: [fastembed-go](https://github.com/Anush008/fastembed-go)
- JavaScript 🌐: [fastembed-js](https://github.com/Anush008/fastembed-js)

## 🤖 Models

### Text Embedding

- [**BAAI/bge-small-en-v1.5**](https://huggingface.co/BAAI/bge-small-en-v1.5) - Default
- [**sentence-transformers/all-MiniLM-L6-v2**](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [**mixedbread-ai/mxbai-embed-large-v1**](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)
- [**Qdrant/clip-ViT-B-32-text**](https://huggingface.co/Qdrant/clip-ViT-B-32-text) - pairs with `clip-ViT-B-32-vision` for image-to-text search
- [**BAAI/bge-large-en-v1.5**](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [**BAAI/bge-small-zh-v1.5**](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- [**BAAI/bge-large-zh-v1.5**](https://huggingface.co/BAAI/bge-large-zh-v1.5)
- [**BAAI/bge-base-en-v1.5**](https://huggingface.co/BAAI/bge-base-en-v1.5)
- [**sentence-transformers/all-MiniLM-L12-v2**](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
- [**sentence-transformers/paraphrase-MiniLM-L12-v2**](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L12-v2)
- [**sentence-transformers/paraphrase-multilingual-mpnet-base-v2**](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
- [**lightonai/ModernBERT-embed-large**](https://huggingface.co/lightonai/modernbert-embed-large)
- [**nomic-ai/nomic-embed-text-v1**](https://huggingface.co/nomic-ai/nomic-embed-text-v1)
- [**nomic-ai/nomic-embed-text-v1.5**](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) - pairs with `nomic-embed-vision-v1.5` for image-to-text search
- [**intfloat/multilingual-e5-small**](https://huggingface.co/intfloat/multilingual-e5-small)
- [**intfloat/multilingual-e5-base**](https://huggingface.co/intfloat/multilingual-e5-base)
- [**intfloat/multilingual-e5-large**](https://huggingface.co/intfloat/multilingual-e5-large)
- [**Alibaba-NLP/gte-base-en-v1.5**](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5)
- [**Alibaba-NLP/gte-large-en-v1.5**](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5)

### Sparse Text Embedding

- [**prithivida/Splade_PP_en_v1**](https://huggingface.co/prithivida/Splade_PP_en_v1) - Default

### Image Embedding

- [**Qdrant/clip-ViT-B-32-vision**](https://huggingface.co/Qdrant/clip-ViT-B-32-vision) - Default
- [**Qdrant/resnet50-onnx**](https://huggingface.co/Qdrant/resnet50-onnx)
- [**Qdrant/Unicom-ViT-B-16**](https://huggingface.co/Qdrant/Unicom-ViT-B-16)
- [**Qdrant/Unicom-ViT-B-32**](https://huggingface.co/Qdrant/Unicom-ViT-B-32)
- [**nomic-ai/nomic-embed-vision-v1.5**](https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5)

### Reranking

- [**BAAI/bge-reranker-base**](https://huggingface.co/BAAI/bge-reranker-base) - Default
- [**BAAI/bge-reranker-v2-m3**](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [**jinaai/jina-reranker-v1-turbo-en**](https://huggingface.co/jinaai/jina-reranker-v1-turbo-en)
- [**jinaai/jina-reranker-v2-base-multiligual**](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)

## ✊ Support

To support the library, please donate to our primary upstream dependency, [`ort`](https://github.com/pykeio/ort?tab=readme-ov-file#-sponsor-ort) - The Rust wrapper for the ONNX runtime.

## 🚀 Installation

Run the following in your project directory:

```bash
cargo add fastembed
```

Or add the following line to your Cargo.toml:

```toml
[dependencies]
fastembed = "5"
```

## 📖 Usage

### Text Embeddings

```rust
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};

// With default options
let mut model = TextEmbedding::try_new(Default::default())?;

// With custom options
let mut model = TextEmbedding::try_new(
    InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
)?;

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

### Sparse Text Embeddings

```rust
use fastembed::{SparseEmbedding, SparseInitOptions, SparseModel, SparseTextEmbedding};

// With default options
let mut model = SparseTextEmbedding::try_new(Default::default())?;

// With custom options
let mut model = SparseTextEmbedding::try_new(
    SparseInitOptions::new(SparseModel::SPLADEPPV1).with_show_download_progress(true),
)?;

let documents = vec![
    "passage: Hello, World!",
    "query: Hello, World!",
    "passage: This is an example passage.",
    "fastembed-rs is licensed under Apache  2.0"
    ];

// Generate embeddings with the default batch size, 256
let embeddings: Vec<SparseEmbedding> = model.embed(documents, None)?;
```

### Image Embeddings

```rust
use fastembed::{ImageEmbedding, ImageInitOptions, ImageEmbeddingModel};

// With default options
let mut model = ImageEmbedding::try_new(Default::default())?;

// With custom options
let mut model = ImageEmbedding::try_new(
    ImageInitOptions::new(ImageEmbeddingModel::ClipVitB32).with_show_download_progress(true),
)?;

let images = vec!["assets/image_0.png", "assets/image_1.png"];

// Generate embeddings with the default batch size, 256
let embeddings = model.embed(images, None)?;

println!("Embeddings length: {}", embeddings.len()); // -> Embeddings length: 2
println!("Embedding dimension: {}", embeddings[0].len()); // -> Embedding dimension: 512
```

### Candidates Reranking

```rust
use fastembed::{TextRerank, RerankInitOptions, RerankerModel};

// With default options
let mut model = TextRerank::try_new(Default::default())?;

// With custom options
let mut model = TextRerank::try_new(
    RerankInitOptions::new(RerankerModel::BGERerankerBase).with_show_download_progress(true),
)?;

let documents = vec![
    "hi",
    "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear, is a bear species endemic to China.",
    "panda is animal",
    "i dont know",
    "kind of mammal",
    ];

// Rerank with the default batch size, 256 and return document contents
let results = model.rerank("what is panda?", documents, true, None)?;
println!("Rerank result: {:?}", results);
```

Alternatively, local model files can be used for inference via the `try_new_from_user_defined(...)` methods of respective structs.

## 📄 LICENSE

[Apache 2.0](https://github.com/Anush008/fastembed-rs/blob/main/LICENSE)
