use crate::ModelInfo;

pub mod image_embedding;
pub mod model_info;
pub mod quantization;
pub mod reranking;
pub mod sparse;
pub mod text_embedding;

pub trait ModelTrait {
    type Model;
    fn get_model_info(model: &Self::Model) -> Option<&ModelInfo<Self::Model>>;
}
