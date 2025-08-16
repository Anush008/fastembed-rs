const DEFAULT_BATCH_SIZE: usize = 256;
const DEFAULT_MAX_LENGTH: usize = 512;

mod init;
pub use init::*;

mod r#impl;
