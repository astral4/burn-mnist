use anyhow::Result;
use burn::data::dataset::Dataset;
use burn::tensor::backend::Backend;
use burn::tensor::{Data, ElementConversion, Int, Shape, Tensor};
use std::{fs::read, iter::zip, path::Path};
use tap::Pipe;

const IMAGE_WIDTH: usize = 28;
const IMAGE_HEIGHT: usize = 28;

#[derive(Debug)]
pub struct MnistDataset<B: Backend> {
    inner: Vec<MnistItem<B>>,
}

impl<B: Backend> MnistDataset<B> {
    fn from_idx_files<P1: AsRef<Path>, P2: AsRef<Path>>(
        images_path: P1,
        labels_path: P2,
        device: &B::Device,
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
            // convert image to tensor
            let image_data = Data {
                // convert pixel values to f32
                value: image
                    .iter()
                    .map(|n| f32::from(*n))
                    .map(ElementConversion::elem)
                    .collect(),
                shape: Shape {
                    dims: [1, IMAGE_WIDTH, IMAGE_HEIGHT],
                },
            };
            let mut image_tensor = Tensor::from_data(image_data, device);

            // We normalize pixel values so that every pixel is in the range [0, 1]
            // and, across the entire dataset, the mean is 0 and the std dev is 1.
            // According to the PyTorch MNIST example[1], after converting to the range, the mean is 0.1307 and the std dev is 0.3081.
            // [1]: https://github.com/pytorch/examples/blob/7df10c2a8606d26a251f322b62c6c4de501b6519/mnist/main.py#L122
            image_tensor = ((image_tensor / 255) - 0.1307) / 0.3081;

            // convert label to tensor
            let label_data = Data {
                value: vec![i16::from(label).elem()],
                shape: Shape { dims: [1] },
            };
            let label_tensor = Tensor::from_data(label_data, device);

            dataset.push(MnistItem {
                image: image_tensor,
                label: label_tensor,
            });
        }

        Ok(Self { inner: dataset })
    }
}

impl<B: Backend> Dataset<MnistItem<B>> for MnistDataset<B> {
    fn get(&self, index: usize) -> Option<MnistItem<B>> {
        self.inner.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

#[derive(Clone, Debug)]
pub struct MnistItem<B: Backend> {
    image: Tensor<B, 3>,
    label: Tensor<B, 1, Int>,
}
