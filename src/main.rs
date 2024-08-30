use anyhow::{Context, Result};
use burn::{
    backend::{
        wgpu::{Wgpu, WgpuDevice},
        Autodiff,
    },
    optim::AdamWConfig,
};
use burn_mnist::{
    data::MnistDataset,
    model::ModelConfig,
    train::{train, TrainingConfig},
};

type Backend = Autodiff<Wgpu>;

fn main() -> Result<()> {
    let train_dataset = MnistDataset::from_idx_files(
        "./data/train-images.idx3-ubyte",
        "./data/train-labels.idx1-ubyte",
    )
    .context("Failed to load training dataset")?;

    let test_dataset = MnistDataset::from_idx_files(
        "./data/t10k-images.idx3-ubyte",
        "./data/t10k-labels.idx1-ubyte",
    )
    .context("Failed to load validation dataset")?;

    let device = WgpuDevice::BestAvailable;

    train::<_, Backend>(
        "./model",
        &TrainingConfig::new(ModelConfig::new(10, 512), AdamWConfig::new()),
        &device,
        train_dataset,
        test_dataset,
    )
    .context("Failed to train model")?;

    /*
    // Here's an example of using a trained model for inference.

    use burn::data::dataloader::Dataset;
    use burn_mnist::infer::infer;

    let item = test_dataset.get(42).unwrap();
    let prediction = infer::<_, Backend>("./model", device, item.clone())?;
    println!("Prediction: {prediction}; Actual: {}", item.label());
    */

    Ok(())
}
