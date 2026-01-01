/// Enum for quantization mode.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMode {
    #[default]
    None,
    Static,
    Dynamic,
}
