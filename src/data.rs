use crate::{IMAGE_HEIGHT, IMAGE_WIDTH};
use anyhow::Result;
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{backend::Backend, Data, ElementConversion, Int, Shape, Tensor},
};
use std::{fs::read, iter::zip, path::Path};
use tap::Pipe;

#[derive(Debug)]
pub struct MnistDataset {
    inner: Vec<MnistItem>,
}

impl MnistDataset {
    fn from_idx_files<P1: AsRef<Path>, P2: AsRef<Path>>(
        images_path: P1,
        labels_path: P2,
    ) -> Result<Self> {
        let images_buf = read(images_path)?;
        let num_images = <[u8; 4]>::try_from(&images_buf[4..7])?.pipe(u32::from_be_bytes);

        let mut labels_buf = read(labels_path)?;
        let num_labels = <[u8; 4]>::try_from(&labels_buf[4..7])?.pipe(u32::from_be_bytes);

        assert_eq!(
            num_images, num_labels,
            "Every image must have exactly one corresponding label"
        );

        let mut dataset = Vec::with_capacity(num_images as usize);

        for (image, label) in zip(
            images_buf[16..].chunks_exact(IMAGE_WIDTH * IMAGE_HEIGHT),
            labels_buf.drain(8..),
        ) {
            // We normalize pixel values so that every pixel is in the range [0, 1]
            // and, across the entire dataset, the mean is 0 and the std dev is 1.
            // According to the PyTorch MNIST example[1], after converting to the range, the mean is 0.1307 and the std dev is 0.3081.
            // [1]: https://github.com/pytorch/examples/blob/7df10c2a8606d26a251f322b62c6c4de501b6519/mnist/main.py#L122
            let image_data = image
                .iter()
                .map(|n| f32::from(*n))
                .map(|n| ((n / 255.) - 0.1307) / 0.3081)
                .collect();

            dataset.push(MnistItem {
                image: image_data,
                label: i16::from(label),
            });
        }

        Ok(Self { inner: dataset })
    }
}

impl Dataset<MnistItem> for MnistDataset {
    fn get(&self, index: usize) -> Option<MnistItem> {
        self.inner.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

#[derive(Clone, Debug)]
pub struct MnistItem {
    image: Vec<f32>,
    label: i16,
}

#[derive(Debug)]
pub struct MnistBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> MnistBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<MnistItem, MnistBatch<B>> for MnistBatcher<B> {
    fn batch(&self, items: Vec<MnistItem>) -> MnistBatch<B> {
        let labels = items
            .iter()
            .map(|item| item.label)
            .map(|label| Tensor::from_ints([label.elem()], &self.device))
            .collect();

        let images = items
            .into_iter()
            .map(|item| item.image)
            .map(|image| {
                Tensor::from_data(
                    Data::new(image, Shape::new([1, IMAGE_WIDTH, IMAGE_HEIGHT])).convert(),
                    &self.device,
                )
            })
            .collect();

        MnistBatch {
            images: Tensor::cat(images, 0),
            labels: Tensor::cat(labels, 0),
        }
    }
}

#[derive(Debug)]
pub struct MnistBatch<B: Backend> {
    images: Tensor<B, 3>,
    labels: Tensor<B, 1, Int>,
}
