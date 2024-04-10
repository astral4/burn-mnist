use anyhow::{Context, Result};
use burn::{
    backend::{
        wgpu::{AutoGraphicsApi, WgpuDevice},
        Autodiff, Wgpu,
    },
    optim::AdamWConfig,
};
use burn_mnist::{
    data::MnistDataset,
    model::ModelConfig,
    train::{train, TrainingConfig},
};

type Backend = Autodiff<Wgpu<AutoGraphicsApi, f32, i32>>;

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

    let device = WgpuDevice::default();

    train::<_, Backend>(
        "./model",
        &TrainingConfig::new(ModelConfig::new(10, 512), AdamWConfig::new()),
        &device,
        train_dataset,
        test_dataset,
    )
    .context("Failed to train model")?;

    Ok(())
}
