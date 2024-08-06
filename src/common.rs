use std::io::Read;
use std::{fs::File, path::PathBuf};

use anyhow::Result;

pub const DEFAULT_CACHE_DIR: &str = ".fastembed_cache";

pub struct SparseEmbedding {
    pub indices: Vec<usize>,
    pub values: Vec<f32>,
}

/// Type alias for the embedding vector
pub type Embedding = Vec<f32>;

/// Type alias for the error type
pub type Error = anyhow::Error;

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
