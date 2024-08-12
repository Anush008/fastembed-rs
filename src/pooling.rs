use ndarray::{s, ArrayBase, Dim, IxDynImpl, OwnedRepr, ViewRepr};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pooling {
    Cls,
    Mean,
}

pub fn cls(
    tensor: &ArrayBase<ViewRepr<&f32>, Dim<IxDynImpl>>,
) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
    tensor.slice(s![.., 0, ..]).to_owned()
}

pub fn mean(
    tensor: &ArrayBase<ViewRepr<&f32>, Dim<IxDynImpl>>,
    attention_mask: &ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
) -> ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>> {
    let masked_tensor = tensor * attention_mask;
    let sum = masked_tensor.sum_axis(ndarray::Axis(1));
    let mask_sum = attention_mask.sum_axis(ndarray::Axis(1));
    let mask_sum = mask_sum.mapv(|x| if x == 0f32 { 1.0 } else { x });
    &sum / &mask_sum
}
