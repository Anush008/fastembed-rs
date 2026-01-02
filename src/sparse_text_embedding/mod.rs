const DEFAULT_BATCH_SIZE: usize = 256;
const DEFAULT_MAX_LENGTH: usize = 512;

mod bgem3_weights;
mod init;
pub use init::*;

mod r#impl;
