use crate::pooling::Pooling;

use super::model_info::ModelInfo;

use super::quantization::QuantizationMode;

use std::{collections::HashMap, fmt::Display, sync::OnceLock};

/// Lazy static list of all available models.
static MODEL_MAP: OnceLock<HashMap<EmbeddingModel, ModelInfo<EmbeddingModel>>> = OnceLock::new();

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EmbeddingModel {
    /// sentence-transformers/all-MiniLM-L6-v2
    AllMiniLML6V2,
    /// Quantized sentence-transformers/all-MiniLM-L6-v2
    AllMiniLML6V2Q,
    /// sentence-transformers/all-MiniLM-L12-v2
    AllMiniLML12V2,
    /// Quantized sentence-transformers/all-MiniLM-L12-v2
    AllMiniLML12V2Q,
    /// BAAI/bge-base-en-v1.5
    BGEBaseENV15,
    /// Quantized BAAI/bge-base-en-v1.5
    BGEBaseENV15Q,
    /// BAAI/bge-large-en-v1.5
    BGELargeENV15,
    /// Quantized BAAI/bge-large-en-v1.5
    BGELargeENV15Q,
    /// BAAI/bge-small-en-v1.5 - Default
    BGESmallENV15,
    /// Quantized BAAI/bge-small-en-v1.5
    BGESmallENV15Q,
    /// nomic-ai/nomic-embed-text-v1
    NomicEmbedTextV1,
    /// nomic-ai/nomic-embed-text-v1.5
    NomicEmbedTextV15,
    /// Quantized v1.5 nomic-ai/nomic-embed-text-v1.5
    NomicEmbedTextV15Q,
    /// sentence-transformers/paraphrase-MiniLM-L6-v2
    ParaphraseMLMiniLML12V2,
    /// Quantized sentence-transformers/paraphrase-MiniLM-L6-v2
    ParaphraseMLMiniLML12V2Q,
    /// sentence-transformers/paraphrase-mpnet-base-v2
    ParaphraseMLMpnetBaseV2,
    /// BAAI/bge-small-zh-v1.5
    BGESmallZHV15,
    /// intfloat/multilingual-e5-small
    MultilingualE5Small,
    /// intfloat/multilingual-e5-base
    MultilingualE5Base,
    /// intfloat/multilingual-e5-large
    MultilingualE5Large,
    /// mixedbread-ai/mxbai-embed-large-v1
    MxbaiEmbedLargeV1,
    /// Quantized mixedbread-ai/mxbai-embed-large-v1
    MxbaiEmbedLargeV1Q,
    /// Alibaba-NLP/gte-base-en-v1.5
    GTEBaseENV15,
    /// Quantized Alibaba-NLP/gte-base-en-v1.5
    GTEBaseENV15Q,
    /// Alibaba-NLP/gte-large-en-v1.5
    GTELargeENV15,
    /// Quantized Alibaba-NLP/gte-large-en-v1.5
    GTELargeENV15Q,
    /// Qdrant/clip-ViT-B-32-text
    ClipVitB32,
}

/// Centralized function to initialize the models map.
fn init_models_map() -> HashMap<EmbeddingModel, ModelInfo<EmbeddingModel>> {
    let models_list = vec![
        ModelInfo {
            model: EmbeddingModel::AllMiniLML6V2,
            dim: 384,
            description: String::from("Sentence Transformer model, MiniLM-L6-v2"),
            model_code: String::from("Qdrant/all-MiniLM-L6-v2-onnx"),
            model_file: String::from("model.onnx"),
        },
        ModelInfo {
            model: EmbeddingModel::AllMiniLML6V2Q,
            dim: 384,
            description: String::from("Quantized Sentence Transformer model, MiniLM-L6-v2"),
            model_code: String::from("Xenova/all-MiniLM-L6-v2"),
            model_file: String::from("onnx/model_quantized.onnx"),
        },
        ModelInfo {
            model: EmbeddingModel::AllMiniLML12V2,
            dim: 384,
            description: String::from("Sentence Transformer model, MiniLM-L12-v2"),
            model_code: String::from("Xenova/all-MiniLM-L12-v2"),
            model_file: String::from("onnx/model.onnx"),
        },
        ModelInfo {
            model: EmbeddingModel::AllMiniLML12V2Q,
            dim: 384,
            description: String::from("Quantized Sentence Transformer model, MiniLM-L12-v2"),
            model_code: String::from("Xenova/all-MiniLM-L12-v2"),
            model_file: String::from("onnx/model_quantized.onnx"),
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
        ModelInfo {
            model: EmbeddingModel::GTEBaseENV15,
            dim: 768,
            description: String::from("Large multilingual embedding model from Alibaba"),
            model_code: String::from("Alibaba-NLP/gte-base-en-v1.5"),
            model_file: String::from("onnx/model.onnx"),
        },
        ModelInfo {
            model: EmbeddingModel::GTEBaseENV15Q,
            dim: 768,
            description: String::from("Quantized Large multilingual embedding model from Alibaba"),
            model_code: String::from("Alibaba-NLP/gte-base-en-v1.5"),
            model_file: String::from("onnx/model_quantized.onnx"),
        },
        ModelInfo {
            model: EmbeddingModel::GTELargeENV15,
            dim: 1024,
            description: String::from("Large multilingual embedding model from Alibaba"),
            model_code: String::from("Alibaba-NLP/gte-large-en-v1.5"),
            model_file: String::from("onnx/model.onnx"),
        },
        ModelInfo {
            model: EmbeddingModel::GTELargeENV15Q,
            dim: 1024,
            description: String::from("Quantized Large multilingual embedding model from Alibaba"),
            model_code: String::from("Alibaba-NLP/gte-large-en-v1.5"),
            model_file: String::from("onnx/model_quantized.onnx"),
        },
        ModelInfo {
            model: EmbeddingModel::ClipVitB32,
            dim: 512,
            description: String::from("CLIP text encoder based on ViT-B/32"),
            model_code: String::from("Qdrant/clip-ViT-B-32-text"),
            model_file: String::from("model.onnx"),
        },
    ];

    // TODO: Use when out in stable
    // assert_eq!(
    //     std::mem::variant_count::<EmbeddingModel>(),
    //     models_list.len(),
    //     "models::models() is not exhaustive"
    // );

    models_list
        .into_iter()
        .fold(HashMap::new(), |mut map, model| {
            // Insert the model into the map
            map.insert(model.model.clone(), model);
            map
        })
}

/// Get a map of all available models.
pub fn models_map() -> &'static HashMap<EmbeddingModel, ModelInfo<EmbeddingModel>> {
    MODEL_MAP.get_or_init(init_models_map)
}

/// Get model information by model code.
pub fn get_model_info(model: &EmbeddingModel) -> Option<&ModelInfo<EmbeddingModel>> {
    models_map().get(model)
}

/// Get a list of all available models.
///
/// This will assign new memory to the models list; where possible, use
/// [`models_map`] instead.
pub fn models_list() -> Vec<ModelInfo<EmbeddingModel>> {
    models_map().values().cloned().collect()
}

impl EmbeddingModel {
    pub fn get_default_pooling_method(&self) -> Option<Pooling> {
        match self {
            EmbeddingModel::AllMiniLML6V2 => Some(Pooling::Mean),
            EmbeddingModel::AllMiniLML6V2Q => Some(Pooling::Mean),
            EmbeddingModel::AllMiniLML12V2 => Some(Pooling::Mean),
            EmbeddingModel::AllMiniLML12V2Q => Some(Pooling::Mean),

            EmbeddingModel::BGEBaseENV15 => Some(Pooling::Cls),
            EmbeddingModel::BGEBaseENV15Q => Some(Pooling::Cls),
            EmbeddingModel::BGELargeENV15 => Some(Pooling::Cls),
            EmbeddingModel::BGELargeENV15Q => Some(Pooling::Cls),
            EmbeddingModel::BGESmallENV15 => Some(Pooling::Cls),
            EmbeddingModel::BGESmallENV15Q => Some(Pooling::Cls),
            EmbeddingModel::BGESmallZHV15 => Some(Pooling::Cls),

            EmbeddingModel::NomicEmbedTextV1 => Some(Pooling::Mean),
            EmbeddingModel::NomicEmbedTextV15 => Some(Pooling::Mean),
            EmbeddingModel::NomicEmbedTextV15Q => Some(Pooling::Mean),

            EmbeddingModel::ParaphraseMLMiniLML12V2 => Some(Pooling::Mean),
            EmbeddingModel::ParaphraseMLMiniLML12V2Q => Some(Pooling::Mean),
            EmbeddingModel::ParaphraseMLMpnetBaseV2 => Some(Pooling::Mean),

            EmbeddingModel::MultilingualE5Base => Some(Pooling::Mean),
            EmbeddingModel::MultilingualE5Small => Some(Pooling::Mean),
            EmbeddingModel::MultilingualE5Large => Some(Pooling::Mean),

            EmbeddingModel::MxbaiEmbedLargeV1 => Some(Pooling::Cls),
            EmbeddingModel::MxbaiEmbedLargeV1Q => Some(Pooling::Cls),

            EmbeddingModel::GTEBaseENV15 => Some(Pooling::Cls),
            EmbeddingModel::GTEBaseENV15Q => Some(Pooling::Cls),
            EmbeddingModel::GTELargeENV15 => Some(Pooling::Cls),
            EmbeddingModel::GTELargeENV15Q => Some(Pooling::Cls),

            EmbeddingModel::ClipVitB32 => Some(Pooling::Mean),
        }
    }

    /// Get the quantization mode of the model.
    ///
    /// Any models with a `Q` suffix in their name are quantized models.
    ///
    /// Currently only 6 supported models have dynamic quantization:
    /// - Alibaba-NLP/gte-base-en-v1.5
    /// - Alibaba-NLP/gte-large-en-v1.5
    /// - mixedbread-ai/mxbai-embed-large-v1
    /// - nomic-ai/nomic-embed-text-v1.5
    /// - Xenova/all-MiniLM-L12-v2
    /// - Xenova/all-MiniLM-L6-v2
    ///
    // TODO: Update this list when more models are added
    pub fn get_quantization_mode(&self) -> QuantizationMode {
        match self {
            EmbeddingModel::AllMiniLML6V2Q => QuantizationMode::Dynamic,
            EmbeddingModel::AllMiniLML12V2Q => QuantizationMode::Dynamic,
            EmbeddingModel::BGEBaseENV15Q => QuantizationMode::Static,
            EmbeddingModel::BGELargeENV15Q => QuantizationMode::Static,
            EmbeddingModel::BGESmallENV15Q => QuantizationMode::Static,
            EmbeddingModel::NomicEmbedTextV15Q => QuantizationMode::Dynamic,
            EmbeddingModel::ParaphraseMLMiniLML12V2Q => QuantizationMode::Static,
            EmbeddingModel::MxbaiEmbedLargeV1Q => QuantizationMode::Dynamic,
            EmbeddingModel::GTEBaseENV15Q => QuantizationMode::Dynamic,
            EmbeddingModel::GTELargeENV15Q => QuantizationMode::Dynamic,
            _ => QuantizationMode::None,
        }
    }
}

impl Display for EmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let model_info = get_model_info(self).expect("Model not found.");
        write!(f, "{}", model_info.model_code)
    }
}
