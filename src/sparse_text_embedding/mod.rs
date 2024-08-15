use crate::models::sparse::SparseModel;

const DEFAULT_BATCH_SIZE: usize = 256;
const DEFAULT_MAX_LENGTH: usize = 512;
const DEFAULT_EMBEDDING_MODEL: SparseModel = SparseModel::SPLADEPPV1;

mod init;
pub use init::*;

mod r#impl;
