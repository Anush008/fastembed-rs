use anyhow::Result;
#[cfg(feature = "online")]
use hf_hub::api::sync::ApiRepo;
use std::io::Read;
use std::{fs::File, path::PathBuf};
use tokenizers::{AddedToken, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

pub const DEFAULT_CACHE_DIR: &str = ".fastembed_cache";

pub struct SparseEmbedding {
    pub indices: Vec<usize>,
    pub values: Vec<f32>,
}

/// Type alias for the embedding vector
pub type Embedding = Vec<f32>;

/// Type alias for the error type
pub type Error = anyhow::Error;

// Tokenizer files for "bring your own" models
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerFiles {
    pub tokenizer_file: Vec<u8>,
    pub config_file: Vec<u8>,
    pub special_tokens_map_file: Vec<u8>,
    pub tokenizer_config_file: Vec<u8>,
}

/// The procedure for loading tokenizer files from the hugging face hub is separated
/// from the main load_tokenizer function (which is expecting bytes, from any source).
#[cfg(feature = "online")]
pub fn load_tokenizer_hf_hub(model_repo: ApiRepo, max_length: usize) -> Result<Tokenizer> {
    let tokenizer_files: TokenizerFiles = TokenizerFiles {
        tokenizer_file: read_file_to_bytes(&model_repo.get("tokenizer.json")?)?,
        config_file: read_file_to_bytes(&model_repo.get("config.json")?)?,
        special_tokens_map_file: read_file_to_bytes(&model_repo.get("special_tokens_map.json")?)?,

        tokenizer_config_file: read_file_to_bytes(&model_repo.get("tokenizer_config.json")?)?,
    };

    load_tokenizer(tokenizer_files, max_length)
}

/// Function can be called directly from the try_new_from_user_defined function (providing file bytes)
///
/// Or indirectly from the try_new function via load_tokenizer_hf_hub (converting HF files to bytes)
pub fn load_tokenizer(tokenizer_files: TokenizerFiles, max_length: usize) -> Result<Tokenizer> {
    let base_error_message =
        "Error building TokenizerFiles for UserDefinedEmbeddingModel. Could not read {} file.";

    // Serialise each tokenizer file
    let config: serde_json::Value =
        serde_json::from_slice(&tokenizer_files.config_file).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                base_error_message.replace("{}", "config.json"),
            )
        })?;
    let special_tokens_map: serde_json::Value =
        serde_json::from_slice(&tokenizer_files.special_tokens_map_file).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                base_error_message.replace("{}", "special_tokens_map.json"),
            )
        })?;
    let tokenizer_config: serde_json::Value =
        serde_json::from_slice(&tokenizer_files.tokenizer_config_file).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                base_error_message.replace("{}", "tokenizer_config.json"),
            )
        })?;
    let mut tokenizer: tokenizers::Tokenizer =
        tokenizers::Tokenizer::from_bytes(tokenizer_files.tokenizer_file).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                base_error_message.replace("{}", "tokenizer.json"),
            )
        })?;

    //For BGEBaseSmall, the model_max_length value is set to 1000000000000000019884624838656. Which fits in a f64
    let model_max_length = tokenizer_config["model_max_length"]
        .as_f64()
        .expect("Error reading model_max_length from tokenizer_config.json")
        as f32;
    let max_length = max_length.min(model_max_length as usize);
    let pad_id = config["pad_token_id"].as_u64().unwrap_or(0) as u32;
    let pad_token = tokenizer_config["pad_token"]
        .as_str()
        .expect("Error reading pad_token from tokenier_config.json")
        .into();

    let mut tokenizer = tokenizer
        .with_padding(Some(PaddingParams {
            // TODO: the user should able to choose the padding strategy
            strategy: PaddingStrategy::BatchLongest,
            pad_token,
            pad_id,
            ..Default::default()
        }))
        .with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }))
        .map_err(anyhow::Error::msg)?
        .clone();
    if let serde_json::Value::Object(root_object) = special_tokens_map {
        for (_, value) in root_object.iter() {
            if value.is_string() {
                tokenizer.add_special_tokens(&[AddedToken {
                    content: value.as_str().unwrap().into(),
                    special: true,
                    ..Default::default()
                }]);
            } else if value.is_object() {
                tokenizer.add_special_tokens(&[AddedToken {
                    content: value["content"].as_str().unwrap().into(),
                    special: true,
                    single_word: value["single_word"].as_bool().unwrap(),
                    lstrip: value["lstrip"].as_bool().unwrap(),
                    rstrip: value["rstrip"].as_bool().unwrap(),
                    normalized: value["normalized"].as_bool().unwrap(),
                }]);
            }
        }
    }
    Ok(tokenizer.into())
}

pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = (v.iter().map(|val| val * val).sum::<f32>()).sqrt();
    let epsilon = 1e-12;

    // We add the super-small epsilon to avoid dividing by zero
    v.iter().map(|&val| val / (norm + epsilon)).collect()
}

/// Public function to read a file to bytes.
/// To be used when loading local model files.
///
/// Could be used to read the onnx file from a local cache in order to constitute a UserDefinedEmbeddingModel.
pub fn read_file_to_bytes(file: &PathBuf) -> Result<Vec<u8>> {
    let mut file = File::open(file)?;
    let file_size = file.metadata()?.len() as usize;
    let mut buffer = Vec::with_capacity(file_size);
    file.read_to_end(&mut buffer)?;
    Ok(buffer)
}
