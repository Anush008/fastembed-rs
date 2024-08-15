use crate::RerankerModel;

const DEFAULT_RE_RANKER_MODEL: RerankerModel = RerankerModel::BGERerankerBase;
const DEFAULT_MAX_LENGTH: usize = 512;
const DEFAULT_BATCH_SIZE: usize = 256;

mod init;
pub use init::*;

mod r#impl;
