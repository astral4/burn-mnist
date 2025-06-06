use crate::{
    data::{MnistBatcher, MnistDataset},
    model::ModelConfig,
    Recorder,
};
use anyhow::{Context, Result};
use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::Module,
    optim::AdamWConfig,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder,
    },
};
use std::{
    fmt::{Debug, Formatter, Result as FmtResult},
    fs::create_dir_all,
    path::Path,
};

#[derive(Config)]
pub struct TrainingConfig {
    pub(crate) model: ModelConfig,
    pub(crate) optimizer: AdamWConfig,
    #[config(default = 10)]
    pub(crate) num_epochs: usize,
    #[config(default = 64)]
    pub(crate) batch_size: usize,
    #[config(default = 4)]
    pub(crate) num_workers: usize,
    #[config(default = 42)]
    pub(crate) seed: u64,
    #[config(default = 1.0e-4)]
    pub(crate) learning_rate: f64,
}

impl Debug for TrainingConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("TrainingConfig")
            .field("model", &self.model)
            .field("num_epochs", &self.num_epochs)
            .field("batch_size", &self.batch_size)
            .field("num_workers", &self.num_workers)
            .field("seed", &self.seed)
            .field("learning_rate", &self.learning_rate)
            .finish_non_exhaustive()
    }
}

/// # Errors
///
/// This function returns an error if:
/// - The directory for storing training artifacts cannot be created
/// - The training configuration file cannot be saved
/// - The trained model cannot be saved
pub fn train<P: AsRef<Path>, B: AutodiffBackend>(
    artifact_dir: P,
    config: &TrainingConfig,
    device: &B::Device,
    train_dataset: MnistDataset,
    test_dataset: MnistDataset,
) -> Result<()> {
    let artifact_dir = artifact_dir.as_ref();

    create_dir_all(artifact_dir).context("Failed to create directory for training artifacts")?;

    config
        .save(artifact_dir.join("config.json"))
        .context("Failed to save training config file")?;

    B::seed(config.seed);

    let dataloader_train = DataLoaderBuilder::<B, _, _>::new(MnistBatcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_test = DataLoaderBuilder::new(MnistBatcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(test_dataset);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(Recorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .build(
            config.model.init(device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model = learner.fit(dataloader_train, dataloader_test);

    model
        .save_file(artifact_dir.join("model"), &Recorder::new())
        .context("Failed to save trained model")?;

    Ok(())
}
