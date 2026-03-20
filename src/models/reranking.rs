use std::{fmt::Display, str::FromStr};

use crate::RerankerModelInfo;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub enum RerankerModel {
    /// BAAI/bge-reranker-base
    #[default]
    BGERerankerBase,
    /// rozgo/bge-reranker-v2-m3
    BGERerankerV2M3,
    /// jinaai/jina-reranker-v1-turbo-en
    JINARerankerV1TurboEn,
    /// jinaai/jina-reranker-v2-base-multilingual
    JINARerankerV2BaseMultiligual,
    // ── mixedbread-ai mxbai-rerank ────────────────────────────────────────────
    /// mixedbread-ai/mxbai-rerank-xsmall-v1 — 33M, English, 512 tokens
    MxbaiRerankXsmallV1,
    /// mixedbread-ai/mxbai-rerank-xsmall-v1 — INT8-quantized
    MxbaiRerankXsmallV1Q,
    /// mixedbread-ai/mxbai-rerank-base-v1 — 86M, English, 512 tokens
    MxbaiRerankBaseV1,
    /// mixedbread-ai/mxbai-rerank-base-v1 — INT8-quantized
    MxbaiRerankBaseV1Q,
    /// mixedbread-ai/mxbai-rerank-large-v1 — 560M, English, 512 tokens
    MxbaiRerankLargeV1,
    /// mixedbread-ai/mxbai-rerank-large-v1 — INT8-quantized
    MxbaiRerankLargeV1Q,
    // ── cross-encoder/ms-marco MiniLM ─────────────────────────────────────────
    /// cross-encoder/ms-marco-MiniLM-L-6-v2 — 22M, English, fast
    MsMarcoMiniLML6V2,
    /// cross-encoder/ms-marco-MiniLM-L-12-v2 — 33M, English, high quality
    MsMarcoMiniLML12V2,
}

pub fn reranker_model_list() -> Vec<RerankerModelInfo> {
    let reranker_model_list = vec![
        RerankerModelInfo {
            model: RerankerModel::BGERerankerBase,
            description: String::from("BGE reranker for English and Chinese"),
            model_code: String::from("BAAI/bge-reranker-base"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: vec![],
        },
        RerankerModelInfo {
            model: RerankerModel::BGERerankerV2M3,
            description: String::from("BGE reranker v2-m3, multilingual"),
            model_code: String::from("rozgo/bge-reranker-v2-m3"),
            model_file: String::from("model.onnx"),
            additional_files: vec![String::from("model.onnx.data")],
        },
        RerankerModelInfo {
            model: RerankerModel::JINARerankerV1TurboEn,
            description: String::from("Jina reranker v1 turbo, English"),
            model_code: String::from("jinaai/jina-reranker-v1-turbo-en"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: vec![],
        },
        RerankerModelInfo {
            model: RerankerModel::JINARerankerV2BaseMultiligual,
            description: String::from("Jina reranker v2, multilingual"),
            model_code: String::from("jinaai/jina-reranker-v2-base-multilingual"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: vec![],
        },
        // ── mxbai-rerank ─────────────────────────────────────────────────────
        RerankerModelInfo {
            model: RerankerModel::MxbaiRerankXsmallV1,
            description: String::from(
                "mxbai-rerank-xsmall-v1 — 33M, English, DeBERTa-v3 cross-encoder",
            ),
            model_code: String::from("mixedbread-ai/mxbai-rerank-xsmall-v1"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: vec![],
        },
        RerankerModelInfo {
            model: RerankerModel::MxbaiRerankXsmallV1Q,
            description: String::from(
                "mxbai-rerank-xsmall-v1 INT8 — 33M, English, DeBERTa-v3 cross-encoder",
            ),
            model_code: String::from("mixedbread-ai/mxbai-rerank-xsmall-v1"),
            model_file: String::from("onnx/model_quantized.onnx"),
            additional_files: vec![],
        },
        RerankerModelInfo {
            model: RerankerModel::MxbaiRerankBaseV1,
            description: String::from(
                "mxbai-rerank-base-v1 — 86M, English, DeBERTa-v3 cross-encoder",
            ),
            model_code: String::from("mixedbread-ai/mxbai-rerank-base-v1"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: vec![],
        },
        RerankerModelInfo {
            model: RerankerModel::MxbaiRerankBaseV1Q,
            description: String::from(
                "mxbai-rerank-base-v1 INT8 — 86M, English, DeBERTa-v3 cross-encoder",
            ),
            model_code: String::from("mixedbread-ai/mxbai-rerank-base-v1"),
            model_file: String::from("onnx/model_quantized.onnx"),
            additional_files: vec![],
        },
        RerankerModelInfo {
            model: RerankerModel::MxbaiRerankLargeV1,
            description: String::from(
                "mxbai-rerank-large-v1 — 560M, English, DeBERTa-v3-large cross-encoder",
            ),
            model_code: String::from("mixedbread-ai/mxbai-rerank-large-v1"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: vec![],
        },
        RerankerModelInfo {
            model: RerankerModel::MxbaiRerankLargeV1Q,
            description: String::from(
                "mxbai-rerank-large-v1 INT8 — 560M, English, DeBERTa-v3-large cross-encoder",
            ),
            model_code: String::from("mixedbread-ai/mxbai-rerank-large-v1"),
            model_file: String::from("onnx/model_quantized.onnx"),
            additional_files: vec![],
        },
        // ── ms-marco MiniLM ───────────────────────────────────────────────────
        RerankerModelInfo {
            model: RerankerModel::MsMarcoMiniLML6V2,
            description: String::from(
                "ms-marco-MiniLM-L-6-v2 — 22M, English, fast BERT cross-encoder",
            ),
            model_code: String::from("cross-encoder/ms-marco-MiniLM-L-6-v2"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: vec![],
        },
        RerankerModelInfo {
            model: RerankerModel::MsMarcoMiniLML12V2,
            description: String::from(
                "ms-marco-MiniLM-L-12-v2 — 33M, English, BERT cross-encoder",
            ),
            model_code: String::from("cross-encoder/ms-marco-MiniLM-L-12-v2"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: vec![],
        },
    ];
    reranker_model_list
}

impl Display for RerankerModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let model_info = reranker_model_list()
            .into_iter()
            .find(|model| model.model == *self)
            .ok_or(std::fmt::Error)?;
        write!(f, "{}", model_info.model_code)
    }
}

impl FromStr for RerankerModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        reranker_model_list()
            .into_iter()
            .find(|m| m.model_code.eq_ignore_ascii_case(s))
            .map(|m| m.model)
            .ok_or_else(|| format!("Unknown reranker model: {s}"))
    }
}

impl TryFrom<String> for RerankerModel {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}
