use crate::{
    data::{MnistBatcher, MnistItem},
    train::TrainingConfig,
    Recorder as ModelRecorder,
};
use anyhow::{Context, Result};
use burn::{
    config::Config, data::dataloader::batcher::Batcher, module::Module, record::Recorder,
    tensor::backend::Backend,
};
use std::path::Path;

/// # Errors
///
/// This function returns an error if:
/// - The training configuration file cannot be loaded
/// - The trained model cannot be loaded
pub fn infer<P: AsRef<Path>, B: Backend>(
    artifact_dir: P,
    device: B::Device,
    item: MnistItem,
) -> Result<B::IntElem> {
    let artifact_dir = artifact_dir.as_ref();

    let config = TrainingConfig::load(artifact_dir.join("config.json"))
        .context("Failed to load training config file")?;

    let record = ModelRecorder::new()
        .load(artifact_dir.join("model"), &device)
        .context("Failed to load trained model")?;

    let model = config.model.init::<B>(&device).load_record(record);

    let batcher = MnistBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.images);
    let prediction = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    Ok(prediction)
}
