use anyhow::{anyhow, Result};
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use ndarray::{Array, Array3};
use std::ops::{Div, Sub};
#[cfg(feature = "online")]
use std::{fs::read_to_string, path::Path};

pub enum TransformData {
    Image(DynamicImage),
    NdArray(Array3<f32>),
}

impl TransformData {
    pub fn image(self) -> anyhow::Result<DynamicImage> {
        match self {
            TransformData::Image(img) => Ok(img),
            _ => Err(anyhow!("TransformData convert error")),
        }
    }

    pub fn array(self) -> anyhow::Result<Array3<f32>> {
        match self {
            TransformData::NdArray(array) => Ok(array),
            _ => Err(anyhow!("TransformData convert error")),
        }
    }
}

pub trait Transform: Send + Sync {
    fn transform(&self, images: TransformData) -> anyhow::Result<TransformData>;
}

struct ConvertToRGB;

impl Transform for ConvertToRGB {
    fn transform(&self, data: TransformData) -> anyhow::Result<TransformData> {
        let image = data.image()?;
        let image = image.into_rgb8().into();
        Ok(TransformData::Image(image))
    }
}

pub struct Resize {
    pub size: (u32, u32),
    pub resample: FilterType,
}

impl Transform for Resize {
    fn transform(&self, data: TransformData) -> anyhow::Result<TransformData> {
        let image = data.image()?;
        let image = image.resize_exact(self.size.0, self.size.1, self.resample);
        Ok(TransformData::Image(image))
    }
}

pub struct CenterCrop {
    pub size: (u32, u32),
}

impl Transform for CenterCrop {
    fn transform(&self, data: TransformData) -> anyhow::Result<TransformData> {
        let mut image = data.image()?;
        let (mut origin_width, mut origin_height) = image.dimensions();
        let (crop_width, crop_height) = self.size;
        if origin_width >= crop_width && origin_height >= crop_height {
            // cropped area is within image boundaries
            let x = (origin_width - crop_width) / 2;
            let y = (origin_height - crop_height) / 2;
            let image = image.crop_imm(x, y, crop_width, crop_height);
            Ok(TransformData::Image(image))
        } else {
            if origin_width > crop_width || origin_height > crop_height {
                let (new_width, new_height) =
                    (origin_width.min(crop_width), origin_height.min(crop_height));
                let (x, y) = if origin_width > crop_width {
                    ((origin_width - crop_width) / 2, 0)
                } else {
                    (0, (origin_height - crop_height) / 2)
                };
                image = image.crop_imm(x, y, new_width, new_height);
                (origin_width, origin_height) = image.dimensions();
            }
            let mut pixels_array =
                Array3::zeros((3usize, crop_width as usize, crop_height as usize));
            let offset_x = (crop_width - origin_width) / 2;
            let offset_y = (crop_height - origin_height) / 2;
            // whc -> chw
            for (x, y, pixel) in image.to_rgb8().enumerate_pixels() {
                pixels_array[[0, (y + offset_y) as usize, (x + offset_x) as usize]] =
                    pixel[0] as f32;
                pixels_array[[1, (y + offset_y) as usize, (x + offset_x) as usize]] =
                    pixel[1] as f32;
                pixels_array[[2, (y + offset_y) as usize, (x + offset_x) as usize]] =
                    pixel[2] as f32;
            }
            Ok(TransformData::NdArray(pixels_array))
        }
    }
}

struct PILToNDarray;

impl Transform for PILToNDarray {
    fn transform(&self, data: TransformData) -> anyhow::Result<TransformData> {
        match data {
            TransformData::Image(image) => {
                let image = image.to_rgb8();
                let (width, height) = image.dimensions();
                // whc -> chw
                let mut pixels_array = Array3::zeros((3usize, height as usize, width as usize));
                for (x, y, pixel) in image.enumerate_pixels() {
                    pixels_array[[0, y as usize, x as usize]] = pixel[0] as f32;
                    pixels_array[[1, y as usize, x as usize]] = pixel[1] as f32;
                    pixels_array[[2, y as usize, x as usize]] = pixel[2] as f32;
                }
                Ok(TransformData::NdArray(pixels_array))
            }
            ndarray => Ok(ndarray),
        }
    }
}

pub struct Rescale {
    pub scale: f32,
}

impl Transform for Rescale {
    fn transform(&self, data: TransformData) -> anyhow::Result<TransformData> {
        let array = data.array()?;
        let array = array * self.scale;
        Ok(TransformData::NdArray(array))
    }
}

pub struct Normalize {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl Transform for Normalize {
    fn transform(&self, data: TransformData) -> anyhow::Result<TransformData> {
        let array = data.array()?;
        let mean = Array::from_vec(self.mean.clone())
            .into_shape_with_order((3, 1, 1))
            .unwrap();
        let std = Array::from_vec(self.std.clone())
            .into_shape_with_order((3, 1, 1))
            .unwrap();

        let shape = array.shape().to_vec();
        match shape.as_slice() {
            [c, h, w] => {
                let array_normalized = array
                    .sub(mean.broadcast((*c, *h, *w)).unwrap())
                    .div(std.broadcast((*c, *h, *w)).unwrap());
                Ok(TransformData::NdArray(array_normalized))
            }
            _ => Err(anyhow!(
                "Transformer convert error. Normlize operator get error shape."
            )),
        }
    }
}

pub struct Compose {
    transforms: Vec<Box<dyn Transform>>,
}

impl Compose {
    fn new(transforms: Vec<Box<dyn Transform>>) -> Self {
        Self { transforms }
    }

    #[cfg(feature = "online")]
    pub fn from_file<P: AsRef<Path>>(file: P) -> anyhow::Result<Self> {
        let content = read_to_string(file)?;
        let config = serde_json::from_str(&content)?;
        load_preprocessor(config)
    }

    pub fn from_bytes<P: AsRef<[u8]>>(bytes: P) -> anyhow::Result<Compose> {
        let config = serde_json::from_slice(bytes.as_ref())?;
        load_preprocessor(config)
    }
}

impl Transform for Compose {
    fn transform(&self, mut image: TransformData) -> anyhow::Result<TransformData> {
        for transform in &self.transforms {
            image = transform.transform(image)?;
        }
        Ok(image)
    }
}

fn load_preprocessor(config: serde_json::Value) -> anyhow::Result<Compose> {
    let mut transformers: Vec<Box<dyn Transform>> = vec![];
    transformers.push(Box::new(ConvertToRGB));

    let mode = config["image_processor_type"]
        .as_str()
        .unwrap_or("CLIPImageProcessor");
    match mode {
        "CLIPImageProcessor" => {
            if config["do_resize"].as_bool().unwrap_or(false) {
                let size = config["size"].clone();
                let shortest_edge = size["shortest_edge"].as_u64();
                let (height, width) = (size["height"].as_u64(), size["width"].as_u64());

                if let Some(shortest_edge) = shortest_edge {
                    let size = (shortest_edge as u32, shortest_edge as u32);
                    transformers.push(Box::new(Resize {
                        size,
                        resample: FilterType::CatmullRom,
                    }));
                } else if let (Some(height), Some(width)) = (height, width) {
                    let size = (height as u32, width as u32);
                    transformers.push(Box::new(Resize {
                        size,
                        resample: FilterType::CatmullRom,
                    }));
                } else {
                    return Err(anyhow!(
                        "Size must contain either 'shortest_edge' or 'height' and 'width'."
                    ));
                }
            }

            if config["do_center_crop"].as_bool().unwrap_or(false) {
                let crop_size = config["crop_size"].clone();
                let (height, width) = if crop_size.is_u64() {
                    let size = crop_size.as_u64().unwrap() as u32;
                    (size, size)
                } else if crop_size.is_object() {
                    (
                        crop_size["height"]
                            .as_u64()
                            .map(|height| height as u32)
                            .ok_or(anyhow!("crop_size height must be cotained"))?,
                        crop_size["width"]
                            .as_u64()
                            .map(|width| width as u32)
                            .ok_or(anyhow!("crop_size width must be cotained"))?,
                    )
                } else {
                    return Err(anyhow!("Invalid crop size: {:?}", crop_size));
                };
                transformers.push(Box::new(CenterCrop {
                    size: (width, height),
                }));
            }
        }
        "ConvNextFeatureExtractor" => {
            let shortest_edge = config["size"]["shortest_edge"].as_u64();
            if shortest_edge.is_none() {
                return Err(anyhow!("Size dictionary must contain 'shortest_edge' key."));
            }
            let shortest_edge = shortest_edge.unwrap() as u32;
            let crop_pct = config["crop_pct"].as_f64().unwrap_or(0.875);
            if shortest_edge < 384 {
                let resize_shortet_edge = shortest_edge as f64 / crop_pct;
                transformers.push(Box::new(Resize {
                    size: (resize_shortet_edge as u32, resize_shortet_edge as u32),
                    resample: FilterType::CatmullRom,
                }));
                transformers.push(Box::new(CenterCrop {
                    size: (shortest_edge, shortest_edge),
                }))
            } else {
                transformers.push(Box::new(Resize {
                    size: (shortest_edge, shortest_edge),
                    resample: FilterType::CatmullRom,
                }));
            }
        }
        mode => return Err(anyhow!("Preprocessror {} is not supported", mode)),
    }

    transformers.push(Box::new(PILToNDarray));

    if config["do_rescale"].as_bool().unwrap_or(true) {
        let rescale_factor = config["rescale_factor"].as_f64().unwrap_or(1.0f64 / 255.0);
        transformers.push(Box::new(Rescale {
            scale: rescale_factor as f32,
        }));
    }

    if config["do_normalize"].as_bool().unwrap_or(false) {
        let mean = config["image_mean"]
            .as_array()
            .ok_or(anyhow!("image_mean must be contained"))?
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .map(|num| num as f32)
                    .ok_or(anyhow!("image_mean must be float"))
            })
            .collect::<Result<Vec<f32>>>()?;
        let std = config["image_std"]
            .as_array()
            .ok_or(anyhow!("image_std must be contained"))?
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .map(|num| num as f32)
                    .ok_or(anyhow!("image_std must be float"))
            })
            .collect::<Result<Vec<f32>>>()?;
        transformers.push(Box::new(Normalize { mean, std }));
    }

    Ok(Compose::new(transformers))
}
