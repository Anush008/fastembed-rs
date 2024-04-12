use variant_count::VariantCount;

#[derive(Debug, Clone, PartialEq, Eq, VariantCount)]
pub enum EmbeddingModel {
    /// Sentence Transformer model, MiniLM-L6-v2
    AllMiniLML6V2,
    /// v1.5 release of the base English model
    BGEBaseENV15,
    /// Quantized v1.5 release of the base English model
    BGEBaseENV15Q,
    /// v1.5 release of the large English model
    BGELargeENV15,
    /// Quantized v1.5 release of the large English model
    BGELargeENV15Q,
    /// Fast and Default English model
    BGESmallENV15,
    /// Quantized Fast and Default English model
    BGESmallENV15Q,
    /// 8192 context length english model
    NomicEmbedTextV1,
    /// v1.5 release of the 8192 context length english model
    NomicEmbedTextV15,
    /// Quantized v1.5 release of the 8192 context length english model
    NomicEmbedTextV15Q,
    /// Multi-lingual model
    ParaphraseMLMiniLML12V2,
    /// Quantized Multi-lingual model
    ParaphraseMLMiniLML12V2Q,
    /// Sentence-transformers model for tasks like clustering or semantic search
    ParaphraseMLMpnetBaseV2,
    /// v1.5 release of the small Chinese model
    BGESmallZHV15,
    /// Small model of multilingual E5 Text Embeddings
    MultilingualE5Small,
    /// Base model of multilingual E5 Text Embeddings
    MultilingualE5Base,
    /// Large model of multilingual E5 Text Embeddings
    MultilingualE5Large,
    /// Large English embedding model from MixedBreed.ai
    MxbaiEmbedLargeV1,
    /// Quantized Large English embedding model from MixedBreed.ai
    MxbaiEmbedLargeV1Q,
}

pub(crate) fn models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            model: EmbeddingModel::AllMiniLML6V2,
            dim: 384,
            description: String::from("Sentence Transformer model, MiniLM-L6-v2"),
            model_code: String::from("Qdrant/all-MiniLM-L6-v2-onnx"),
            model_file: String::from("model.onnx"),
        },
        ModelInfo {
            model: EmbeddingModel::BGEBaseENV15,
            dim: 768,
            description: String::from("v1.5 release of the base English model"),
            model_code: String::from("Xenova/bge-base-en-v1.5"),
            model_file: String::from("onnx/model.onnx"),
        },
        ModelInfo {
            model: EmbeddingModel::BGEBaseENV15Q,
            dim: 768,
            description: String::from("Quantized v1.5 release of the large English model"),
            model_code: String::from("Qdrant/bge-base-en-v1.5-onnx-Q"),
            model_file: String::from("model_optimized.onnx"),
        },
        ModelInfo {
            model: EmbeddingModel::BGELargeENV15,
            dim: 1024,
            description: String::from("v1.5 release of the large English model"),
            model_code: String::from("Xenova/bge-large-en-v1.5"),
            model_file: String::from("onnx/model.onnx"),
        },
        ModelInfo {
            model: EmbeddingModel::BGELargeENV15Q,
            dim: 1024,
            description: String::from("Quantized v1.5 release of the large English model"),
            model_code: String::from("Qdrant/bge-large-en-v1.5-onnx-Q"),
            model_file: String::from("model_optimized.onnx"),
        },
        ModelInfo {
            model: EmbeddingModel::BGESmallENV15,
            dim: 384,
            description: String::from("v1.5 release of the fast and default English model"),
            model_code: String::from("Xenova/bge-small-en-v1.5"),
            model_file: String::from("onnx/model.onnx"),
        },
        ModelInfo {
            model: EmbeddingModel::BGESmallENV15Q,
            dim: 384,
            description: String::from(
                "Quantized v1.5 release of the fast and default English model",
            ),
            model_code: String::from("Qdrant/bge-small-en-v1.5-onnx-Q"),
            model_file: String::from("model_optimized.onnx"),
        },
        ModelInfo {
            model: EmbeddingModel::NomicEmbedTextV1,
            dim: 768,
            description: String::from("8192 context length english model"),
            model_code: String::from("nomic-ai/nomic-embed-text-v1"),
            model_file: String::from("onnx/model.onnx"),
        },
        ModelInfo {
            model: EmbeddingModel::NomicEmbedTextV15,
            dim: 768,
            description: String::from("v1.5 release of the 8192 context length english model"),
            model_code: String::from("nomic-ai/nomic-embed-text-v1.5"),
            model_file: String::from("onnx/model.onnx"),
        },
        ModelInfo {
            model: EmbeddingModel::NomicEmbedTextV15Q,
            dim: 768,
            description: String::from(
                "Quantized v1.5 release of the 8192 context length english model",
            ),
            model_code: String::from("nomic-ai/nomic-embed-text-v1.5"),
            model_file: String::from("onnx/model_quantized.onnx"),
        },
        ModelInfo {
            model: EmbeddingModel::ParaphraseMLMiniLML12V2Q,
            dim: 384,
            description: String::from("Quantized Multi-lingual model"),
            model_code: String::from("Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"),
            model_file: String::from("model_optimized.onnx"),
        },
        ModelInfo {
            model: EmbeddingModel::ParaphraseMLMiniLML12V2,
            dim: 384,
            description: String::from("Multi-lingual model"),
            model_code: String::from("Xenova/paraphrase-multilingual-MiniLM-L12-v2"),
            model_file: String::from("onnx/model.onnx"),
        },
        ModelInfo {
            model: EmbeddingModel::ParaphraseMLMpnetBaseV2,
            dim: 768,
            description: String::from(
                "Sentence-transformers model for tasks like clustering or semantic search",
            ),
            model_code: String::from("Xenova/paraphrase-multilingual-mpnet-base-v2"),
            model_file: String::from("onnx/model.onnx"),
        },
        ModelInfo {
            model: EmbeddingModel::BGESmallZHV15,
            dim: 512,
            description: String::from("v1.5 release of the small Chinese model"),
            model_code: String::from("Xenova/bge-small-zh-v1.5"),
            model_file: String::from("onnx/model.onnx"),
        },
        ModelInfo {
            model: EmbeddingModel::MultilingualE5Small,
            dim: 384,
            description: String::from("Small model of multilingual E5 Text Embeddings"),
            model_code: String::from("intfloat/multilingual-e5-small"),
            model_file: String::from("onnx/model.onnx"),
        },
        ModelInfo {
            model: EmbeddingModel::MultilingualE5Base,
            dim: 768,
            description: String::from("Base model of multilingual E5 Text Embeddings"),
            model_code: String::from("intfloat/multilingual-e5-base"),
            model_file: String::from("onnx/model.onnx"),
        },
        ModelInfo {
            model: EmbeddingModel::MultilingualE5Large,
            dim: 1024,
            description: String::from("Large model of multilingual E5 Text Embeddings"),
            model_code: String::from("Qdrant/multilingual-e5-large-onnx"),
            model_file: String::from("model.onnx"),
        },
        ModelInfo {
            model: EmbeddingModel::MxbaiEmbedLargeV1,
            dim: 1024,
            description: String::from("Large English embedding model from MixedBreed.ai"),
            model_code: String::from("mixedbread-ai/mxbai-embed-large-v1"),
            model_file: String::from("onnx/model.onnx"),
        },
        ModelInfo {
            model: EmbeddingModel::MxbaiEmbedLargeV1Q,
            dim: 1024,
            description: String::from("Quantized Large English embedding model from MixedBreed.ai"),
            model_code: String::from("mixedbread-ai/mxbai-embed-large-v1"),
            model_file: String::from("onnx/model_quantized.onnx"),
        },
    ]
}
/// Data struct about the available models
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model: EmbeddingModel,
    pub dim: usize,
    pub description: String,
    pub model_code: String,
    pub model_file: String,
}
