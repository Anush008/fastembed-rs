<div align="center">
  <h1><a href="https://crates.io/crates/fastembed">FastEmbed-rs 🦀</a></h1>
 <h3>Rust library for generating vector embeddings, reranking locally!</h3>
  <a href="https://crates.io/crates/fastembed"><img src="https://img.shields.io/crates/v/fastembed.svg" alt="Crates.io"></a>
  <a href="https://github.com/Anush008/fastembed-rs/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-apache-blue.svg" alt="MIT Licensed"></a>
  <a href="https://github.com/Anush008/fastembed-rs/actions/workflows/release.yml"><img src="https://github.com/Anush008/fastembed-rs/actions/workflows/release.yml/badge.svg?branch=main" alt="Semantic release"></a>
</div>

## Features

- Supports synchronous usage. No dependency on Tokio.
- Uses [@pykeio/ort](https://github.com/pykeio/ort) for performant ONNX inference.
- Uses [@huggingface/tokenizers](https://github.com/huggingface/tokenizers) for fast encodings.

## Not looking for Rust?

- Python: [fastembed](https://github.com/qdrant/fastembed)
- Go: [fastembed-go](https://github.com/Anush008/fastembed-go)
- JavaScript: [fastembed-js](https://github.com/Anush008/fastembed-js)

## Models

### Text Embedding

- [**BAAI/bge-small-en-v1.5**](https://huggingface.co/BAAI/bge-small-en-v1.5) - Default
- [**BAAI/bge-base-en-v1.5**](https://huggingface.co/BAAI/bge-base-en-v1.5)
- [**BAAI/bge-large-en-v1.5**](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [**BAAI/bge-small-zh-v1.5**](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- [**BAAI/bge-large-zh-v1.5**](https://huggingface.co/BAAI/bge-large-zh-v1.5)
- [**BAAI/bge-m3**](https://huggingface.co/BAAI/bge-m3)
- [**sentence-transformers/all-MiniLM-L6-v2**](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [**sentence-transformers/all-MiniLM-L12-v2**](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
- [**sentence-transformers/all-mpnet-base-v2**](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- [**sentence-transformers/paraphrase-MiniLM-L12-v2**](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L12-v2)
- [**sentence-transformers/paraphrase-multilingual-mpnet-base-v2**](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
- [**nomic-ai/nomic-embed-text-v1**](https://huggingface.co/nomic-ai/nomic-embed-text-v1)
- [**nomic-ai/nomic-embed-text-v1.5**](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) - pairs with `nomic-embed-vision-v1.5` for image-to-text search
- [**intfloat/multilingual-e5-small**](https://huggingface.co/intfloat/multilingual-e5-small)
- [**intfloat/multilingual-e5-base**](https://huggingface.co/intfloat/multilingual-e5-base)
- [**intfloat/multilingual-e5-large**](https://huggingface.co/intfloat/multilingual-e5-large)
- [**mixedbread-ai/mxbai-embed-large-v1**](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)
- [**Alibaba-NLP/gte-base-en-v1.5**](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5)
- [**Alibaba-NLP/gte-large-en-v1.5**](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5)
- [**lightonai/ModernBERT-embed-large**](https://huggingface.co/lightonai/modernbert-embed-large)
- [**Qdrant/clip-ViT-B-32-text**](https://huggingface.co/Qdrant/clip-ViT-B-32-text) - pairs with `clip-ViT-B-32-vision` for image-to-text search
- [**jinaai/jina-embeddings-v2-base-code**](https://huggingface.co/jinaai/jina-embeddings-v2-base-code)
- [**jinaai/jina-embeddings-v2-base-en**](https://huggingface.co/jinaai/jina-embeddings-v2-base-en)
- [**google/embeddinggemma-300m**](https://huggingface.co/google/embeddinggemma-300m)
- [**nomic-ai/nomic-embed-text-v2-moe**](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe) - requires `nomic-v2-moe` feature (candle backend)
- [**Qwen/Qwen3-Embedding-0.6B**](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) - requires `qwen3` feature (candle backend)
- [**Qwen/Qwen3-Embedding-4B**](https://huggingface.co/Qwen/Qwen3-Embedding-4B) - requires `qwen3` feature (candle backend)
- [**Qwen/Qwen3-Embedding-8B**](https://huggingface.co/Qwen/Qwen3-Embedding-8B) - requires `qwen3` feature (candle backend)
- [**Qwen/Qwen3-VL-Embedding-2B**](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) - requires `qwen3` feature (candle backend, multimodal via `Qwen3VLEmbedding`)
- [**snowflake/snowflake-arctic-embed-xs**](https://huggingface.co/snowflake/snowflake-arctic-embed-xs)
- [**snowflake/snowflake-arctic-embed-s**](https://huggingface.co/snowflake/snowflake-arctic-embed-s)
- [**snowflake/snowflake-arctic-embed-m**](https://huggingface.co/snowflake/snowflake-arctic-embed-m)
- [**snowflake/snowflake-arctic-embed-m-long**](https://huggingface.co/snowflake/snowflake-arctic-embed-m-long)
- [**snowflake/snowflake-arctic-embed-l**](https://huggingface.co/snowflake/snowflake-arctic-embed-l)
- [**Snowflake/snowflake-arctic-embed-l-v2.0**](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0) — 1024d, 8k context, CLS pooling
- [**Snowflake/snowflake-arctic-embed-m-v2.0**](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0) — 768d, 8k context, CLS pooling (`SnowflakeArcticEmbedMV2`)
- [**telepix/PIXIE-Rune-v1.0**](https://huggingface.co/telepix/PIXIE-Rune-v1.0) — 1024d, 74 languages, 6k context; also available as INT8 (`PixieRuneV1Q`) and INT4 (`PixieRuneV1Int4`, `PixieRuneV1Int4Full`) via [cstr/PIXIE-Rune-v1.0-ONNX](https://huggingface.co/cstr/PIXIE-Rune-v1.0-ONNX)
- [**jinaai/jina-embeddings-v5-text-nano-retrieval**](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval) — 768d, multilingual, pre-pooled output; uses `"Document: "` query prefix (`JinaEmbeddingsV5Nano`)
- [**electroglyph/Qwen3-Embedding-0.6B-onnx-uint8**](https://huggingface.co/electroglyph/Qwen3-Embedding-0.6B-onnx-uint8) — 1024d, uint8 ONNX, decoder-style last-token pooling
- [**cstr/Octen-Embedding-0.6B-ONNX**](https://huggingface.co/cstr/Octen-Embedding-0.6B-ONNX) — 1024d, FP32 and INT4 decoder embedding model (`OctenEmbedding0_6BFp32`, `OctenEmbedding0_6BInt4`); INT8-full (~570 MB, `OctenEmbedding0_6BInt8Full`) and INT4-full (~434 MB, `OctenEmbedding0_6BInt4Full`, INT4 MatMul + INT8 embedding table) variants also available
- [**cstr/F2LLM-v2-0.6B-ONNX**](https://huggingface.co/cstr/F2LLM-v2-0.6B-ONNX) — 1024d, 200+ languages, Qwen3 decoder, last-token pooling; available as FP32 (`F2LlmV2_0_6BFp32`), INT8 (`F2LlmV2_0_6BInt8`), INT4 (`F2LlmV2_0_6BInt4`), and INT8-full (`F2LlmV2_0_6BInt8Full`) via [cstr/F2LLM-v2-0.6B-ONNX-INT8](https://huggingface.co/cstr/F2LLM-v2-0.6B-ONNX-INT8), [cstr/F2LLM-v2-0.6B-ONNX-INT4](https://huggingface.co/cstr/F2LLM-v2-0.6B-ONNX-INT4), [cstr/F2LLM-v2-0.6B-ONNX-INT8-FULL](https://huggingface.co/cstr/F2LLM-v2-0.6B-ONNX-INT8-FULL)
- [**jinaai/jina-embeddings-v5-text-small-retrieval**](https://huggingface.co/jinaai/jina-embeddings-v5-text-small-retrieval) — 677M, 1024d, 32k context, 119+ languages, Qwen3-based, last-token pooling (`JinaEmbeddingsV5Small`); #9 on MTEB reranking. Prepend `"Query: "` to queries and `"Document: "` to passages. Supports `.rerank()` via `TextEmbedding::rerank()`.

Quantized versions are also available for several models above (append `Q` to the model enum variant, e.g., `EmbeddingModel::BGESmallENV15Q`).

### Sparse Text Embedding

- [**prithivida/Splade_PP_en_v1**](https://huggingface.co/prithivida/Splade_PP_en_v1) - Default
- [**BAAI/bge-m3**](https://huggingface.co/BAAI/bge-m3)

### Image Embedding

- [**Qdrant/clip-ViT-B-32-vision**](https://huggingface.co/Qdrant/clip-ViT-B-32-vision) - Default
- [**Qdrant/resnet50-onnx**](https://huggingface.co/Qdrant/resnet50-onnx)
- [**Qdrant/Unicom-ViT-B-16**](https://huggingface.co/Qdrant/Unicom-ViT-B-16)
- [**Qdrant/Unicom-ViT-B-32**](https://huggingface.co/Qdrant/Unicom-ViT-B-32)
- [**nomic-ai/nomic-embed-vision-v1.5**](https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5)

### Reranking

- [**BAAI/bge-reranker-base**](https://huggingface.co/BAAI/bge-reranker-base) - Default — English + Chinese
- [**BAAI/bge-reranker-v2-m3**](https://huggingface.co/BAAI/bge-reranker-v2-m3) — multilingual (`BGERerankerV2M3`)
- [**jinaai/jina-reranker-v1-turbo-en**](https://huggingface.co/jinaai/jina-reranker-v1-turbo-en) — English (`JINARerankerV1TurboEn`)
- [**jinaai/jina-reranker-v2-base-multilingual**](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual) — multilingual (`JINARerankerV2BaseMultiligual`)
- [**mixedbread-ai/mxbai-rerank-xsmall-v1**](https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1) — 33M, English, fast (`MxbaiRerankXsmallV1`, INT8: `MxbaiRerankXsmallV1Q`)
- [**mixedbread-ai/mxbai-rerank-base-v1**](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v1) — 86M, English (`MxbaiRerankBaseV1`, INT8: `MxbaiRerankBaseV1Q`)
- [**mixedbread-ai/mxbai-rerank-large-v1**](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1) — 560M, English, DeBERTa-v3-large (`MxbaiRerankLargeV1`, INT8: `MxbaiRerankLargeV1Q`)
- [**cross-encoder/ms-marco-MiniLM-L-6-v2**](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) — 22M, English, very fast (`MsMarcoMiniLML6V2`)
- [**cross-encoder/ms-marco-MiniLM-L-12-v2**](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2) — 33M, English (`MsMarcoMiniLML12V2`)

## ✊ Support

To support the library, please donate to our primary upstream dependency, [`ort`](https://github.com/pykeio/ort?tab=readme-ov-file#-sponsor-ort) - The Rust wrapper for the ONNX runtime.

## Installation

Run the following in your project directory:

```bash
cargo add fastembed
```

Or add the following line to your Cargo.toml:

```toml
[dependencies]
fastembed = "5"
```

## Usage

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
    "fastembed-rs is licensed under Apache 2.0"
];

 // Generate embeddings with the default batch size, 256
 let embeddings = model.embed(documents, None)?;

 println!("Embeddings length: {}", embeddings.len()); // -> Embeddings length: 4
 println!("Embedding dimension: {}", embeddings[0].len()); // -> Embedding dimension: 384
```

### Bi-Encoder Reranking

Any `TextEmbedding` model can rank documents against a query using cosine similarity of L2-normalised embeddings — the same approach used in MTEB's reranking benchmark:

```rust
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};

let mut model = TextEmbedding::try_new(
    InitOptions::new(EmbeddingModel::JinaEmbeddingsV5Small),
)?;

let results = model.rerank(
    "Query: what is machine learning?",
    vec![
        "Document: Machine learning is a subset of artificial intelligence.",
        "Document: The weather in Paris is mild in spring.",
        "Document: Neural networks learn patterns from training data.",
    ],
    Some(2),  // top-n
    true,     // return document text
)?;

for r in &results {
    println!("{:.3}  {}", r.score, r.document.as_deref().unwrap_or(""));
}
```

Returns the same `Vec<RerankResult>` type as `TextRerank`, sorted by score descending.

### Qwen3 Embeddings

Qwen3 embedding models are available behind the `qwen3` feature flag (candle backend).

```toml
[dependencies]
fastembed = { version = "5", features = ["qwen3"] }
```

```rust
use candle_core::{DType, Device};
use fastembed::Qwen3TextEmbedding;

let device = Device::Cpu;
let model = Qwen3TextEmbedding::from_hf(
    "Qwen/Qwen3-Embedding-0.6B",
    &device,
    DType::F32,
    512,
)?;

// Text-only usage with the Qwen3-VL embedding checkpoint is also supported:
// let model = Qwen3TextEmbedding::from_hf("Qwen/Qwen3-VL-Embedding-2B", &device, DType::F32, 512)?;

let embeddings = model.embed(&["query: ...", "passage: ..."])?;
println!("Embeddings length: {}", embeddings.len());
```

For multimodal text/image usage with `Qwen/Qwen3-VL-Embedding-2B`:

```rust
use candle_core::{DType, Device};
use fastembed::Qwen3VLEmbedding;

let device = Device::Cpu;
let model = Qwen3VLEmbedding::from_hf(
    "Qwen/Qwen3-VL-Embedding-2B",
    &device,
    DType::F32,
    2048,
)?;

let image_embeddings = model.embed_images(&["tests/assets/image_0.png", "tests/assets/image_1.png"])?;
let text_embeddings = model.embed_texts(&["query: blue cat", "query: red cat"])?;

println!("Image embeddings: {}", image_embeddings.len());
println!("Text embeddings: {}", text_embeddings.len());
```

### Nomic Embed Text v2 MoE

The [nomic-embed-text-v2-moe](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe) model is available behind the `nomic-v2-moe` feature flag (candle backend). First general-purpose MoE embedding model with 100+ language support.

```toml
[dependencies]
fastembed = { version = "5", features = ["nomic-v2-moe"] }
```

```rust
use candle_core::{DType, Device};
use fastembed::NomicV2MoeTextEmbedding;

let device = Device::Cpu;
let model = NomicV2MoeTextEmbedding::from_hf(
    "nomic-ai/nomic-embed-text-v2-moe",
    &device,
    DType::F32,
    512,
)?;

let embeddings = model.embed(&["search_query: ...", "search_document: ..."])?;
println!("Embeddings length: {}", embeddings.len());
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
    "fastembed-rs is licensed under Apache 2.0"
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

## LICENSE

[Apache 2.0](https://github.com/Anush008/fastembed-rs/blob/main/LICENSE)
