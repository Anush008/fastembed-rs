use crate::get_cache_dir;
use ort::execution_providers::ExecutionProviderDispatch;
use std::path::PathBuf;

pub trait HasMaxLength {
    const MAX_LENGTH: usize;
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct InitOptionsWithLength<M> {
    pub model_name: M,
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub cache_dir: PathBuf,
    pub show_download_progress: bool,
    pub max_length: usize,
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct InitOptions<M> {
    pub model_name: M,
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub cache_dir: PathBuf,
    pub show_download_progress: bool,
}

impl<M: Default + HasMaxLength> Default for InitOptionsWithLength<M> {
    fn default() -> Self {
        Self {
            model_name: M::default(),
            execution_providers: Default::default(),
            cache_dir: get_cache_dir().into(),
            show_download_progress: true,
            max_length: M::MAX_LENGTH,
        }
    }
}

impl<M: Default> Default for InitOptions<M> {
    fn default() -> Self {
        Self {
            model_name: M::default(),
            execution_providers: Default::default(),
            cache_dir: get_cache_dir().into(),
            show_download_progress: true,
        }
    }
}

impl<M: Default + HasMaxLength> InitOptionsWithLength<M> {
    /// Crea a new InitOptionsWithLength with the given model name
    pub fn new(model_name: M) -> Self {
        Self {
            model_name,
            ..Default::default()
        }
    }

    /// Set the maximum maximum length
    pub fn with_max_length(mut self, max_lenght: usize) -> Self {
        self.max_length = max_lenght;
        self
    }

    /// Set the cache directory for the model file
    pub fn with_cache_dir(mut self, cache_dir: PathBuf) -> Self {
        self.cache_dir = cache_dir;
        self
    }

    /// Set the execution providers for the model
    pub fn with_execution_providers(
        mut self,
        execution_providers: Vec<ExecutionProviderDispatch>,
    ) -> Self {
        self.execution_providers = execution_providers;
        self
    }

    /// Set whether to show download progress
    pub fn with_show_download_progress(mut self, show_download_progress: bool) -> Self {
        self.show_download_progress = show_download_progress;
        self
    }
}

impl<M: Default> InitOptions<M> {
    /// Crea a new InitOptionsWithLength with the given model name
    pub fn new(model_name: M) -> Self {
        Self {
            model_name,
            ..Default::default()
        }
    }

    /// Set the cache directory for the model file
    pub fn with_cache_dir(mut self, cache_dir: PathBuf) -> Self {
        self.cache_dir = cache_dir;
        self
    }

    /// Set the execution providers for the model
    pub fn with_execution_providers(
        mut self,
        execution_providers: Vec<ExecutionProviderDispatch>,
    ) -> Self {
        self.execution_providers = execution_providers;
        self
    }

    /// Set whether to show download progress
    pub fn with_show_download_progress(mut self, show_download_progress: bool) -> Self {
        self.show_download_progress = show_download_progress;
        self
    }
}
