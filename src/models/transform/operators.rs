use std::{borrow::Borrow, fs::read_to_string, path::Path};

use anyhow::{anyhow, Result};

use image::{imageops::FilterType, DynamicImage};

pub trait Transform {
    fn transform(&self, images: Vec<DynamicImage>) -> Vec<DynamicImage>;
}

pub struct ConvertToRGB;

impl Transform for ConvertToRGB {
    fn transform(&self, images: Vec<DynamicImage>) -> Vec<DynamicImage> {
        images
            .into_iter()
            .map(|image| image.to_rgb8().into())
            .collect()
    }
}

pub struct CenterCrop {
    size: (u32, u32),
}

impl CenterCrop {
    pub fn new(height: u64, width: u64) -> Self {
        Self {
            size: (height as u32, width as u32),
        }
    }
}

impl Transform for CenterCrop {
    fn transform(&self, images: Vec<DynamicImage>) -> Vec<DynamicImage> {
        images
            .into_iter()
            .map(|image| {
                let (width, height) = (image.width(), image.height());
                let (crop_width, crop_height) = self.size;
                todo!()
            })
            .collect()
    }
}

pub struct Normlize {
    mean: Vec<f64>,
    std: Vec<f64>,
}

impl Normlize {
    pub fn new(mean: Vec<f64>, std: Vec<f64>) -> Self {
        Self { mean, std }
    }
}

impl Transform for Normlize {
    fn transform(&self, images: Vec<DynamicImage>) -> Vec<DynamicImage> {
        todo!()
    }
}

pub struct Resize {
    size: (u64, u64),
    resample: FilterType,
}

impl Resize {
    pub fn new(size: (u64, u64)) -> Self {
        Self {
            size,
            resample: FilterType::CatmullRom,
        }
    }
}

impl Transform for Resize {
    fn transform(&self, images: Vec<DynamicImage>) -> Vec<DynamicImage> {
        todo!()
    }
}

pub struct Rescale {
    pub scale: f64,
}

impl Rescale {
    pub fn new(scale: f64) -> Self {
        Self { scale }
    }
}

impl Transform for Rescale {
    fn transform(&self, images: Vec<DynamicImage>) -> Vec<DynamicImage> {
        todo!()
    }
}

pub struct PILToNDarray;

impl Transform for PILToNDarray {
    fn transform(&self, images: Vec<DynamicImage>) -> Vec<DynamicImage> {
        todo!()
    }
}

pub struct Compose {
    transforms: Vec<Box<dyn Transform>>,
}

impl Compose {
    pub fn new(transforms: Vec<Box<dyn Transform>>) -> Self {
        Self { transforms }
    }

    pub fn from_file<P: AsRef<Path>>(file: P) -> Result<Self> {
        let content = read_to_string(file)?;
        let config: serde_json::Value = serde_json::from_str(&content)?;
        let mut transformers: Vec<Box<dyn Transform>> = vec![];
        transformers.push(Box::new(ConvertToRGB));

        let mode = config["image_processor_type"]
            .as_str()
            .unwrap_or("CLIPImageProcessor");
        match mode {
            "CLIPImageProcessor" => {
                if config["do_resize"].as_bool().unwrap_or(false) {
                    let size = config["size"].borrow();
                    let shortest_edge = size["shortest_edge"].as_u64();
                    let (height, width) = (size["height"].as_u64(), size["width"].as_u64());
                    let size = if shortest_edge.is_some() {
                        (shortest_edge.unwrap(), shortest_edge.unwrap())
                    } else if height.is_some() && width.is_some() {
                        (height.unwrap(), width.unwrap())
                    } else {
                        return Err(anyhow!(
                            "Size must contain either 'shortest_edge' or 'height' and 'width'."
                        ));
                    };
                    transformers.push(Box::new(Resize::new(size)));
                }

                if config["do_center_crop"].as_bool().unwrap_or(false) {
                    let crop_size = config["crop_size"].borrow();
                    let (height, width) = if crop_size.is_u64() {
                        let size = crop_size.as_u64().unwrap();
                        (size, size)
                    } else if crop_size.is_object() {
                        (
                            crop_size["height"]
                                .as_u64()
                                .ok_or(anyhow!("crop_size height must be cotain"))?,
                            crop_size["width"]
                                .as_u64()
                                .ok_or(anyhow!("crop_size width must be cotain"))?,
                        )
                    } else {
                        return Err(anyhow!("Invalid crop size: {:?}", crop_size));
                    };
                    transformers.push(Box::new(CenterCrop::new(height, width)));
                }
            }
            "ConvNextFeatureExtractor" => {
                let shortest_edge = config["size"]["shortest_edge"].as_u64();
                if shortest_edge.is_none() {
                    return Err(anyhow!("Size dictionary must contain 'shortest_edge' key."));
                }
                let shortest_edge = shortest_edge.unwrap();
                let crop_pct = config["crop_pct"].as_f64().unwrap_or(0.875);
                if shortest_edge < 384 {
                    let resize_shortet_edge = shortest_edge as f64 / crop_pct;
                    transformers.push(Box::new(Resize::new((
                        resize_shortet_edge as u64,
                        resize_shortet_edge as u64,
                    ))));
                    transformers.push(Box::new(CenterCrop::new(shortest_edge, shortest_edge)))
                } else {
                    transformers.push(Box::new(Resize::new((shortest_edge, shortest_edge))));
                }
            }
            mode => return Err(anyhow!("Preprocessror {} is not supported", mode)),
        }

        transformers.push(Box::new(PILToNDarray));

        if config["do_rescale"].as_bool().unwrap_or(true) {
            let rescale_factor = config["rescale_factor"].as_f64().unwrap_or(1.0f64 / 255.0);
            transformers.push(Box::new(Rescale::new(rescale_factor)));
        }

        if config["do_normalize"].as_bool().unwrap_or(false) {
            let mean = config["image_mean"]
                .as_array()
                .ok_or(anyhow!("image_mean must be contain"))?
                .iter()
                .map(|value| value.as_f64().ok_or(anyhow!("image_mean must be float")))
                .collect::<Result<Vec<f64>>>()?;
            let std = config["image_std"]
                .as_array()
                .ok_or(anyhow!("image_std must be contain"))?
                .iter()
                .map(|value| value.as_f64().ok_or(anyhow!("image_std must be float")))
                .collect::<Result<Vec<f64>>>()?;
            transformers.push(Box::new(Normlize::new(mean, std)));
        }

        Ok(Compose::new(transformers))
    }
}

impl Transform for Compose {
    fn transform(&self, mut images: Vec<DynamicImage>) -> Vec<DynamicImage> {
        for transform in &self.transforms {
            images = transform.transform(images);
        }
        images
    }
}

#[cfg(test)]
mod tests {
    use super::Compose;

    #[test]
    fn compose_from_file() {
        let compose =
            Compose::from_file("/Users/mac/Github-Repository/fastembed-rs/preprocess.json");
        assert!(compose.is_ok());
    }
}
