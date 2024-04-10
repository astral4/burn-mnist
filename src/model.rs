use crate::{IMAGE_HEIGHT, IMAGE_WIDTH};
use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        loss::CrossEntropyLoss,
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, GELU,
    },
    tensor::{backend::Backend, Int, Tensor},
    train::ClassificationOutput,
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: GELU,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init(device),
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init(device),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: GELU::new(),
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> Model<B> {
    // input: (batch size) * (image width) * (image height)
    // output: (batch size) * (# of classes)
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let batch_size = images.dims()[0];

        let x = images.reshape([batch_size, 1, IMAGE_WIDTH, IMAGE_HEIGHT]); // create channel at dimension 2
        let x = self.conv1.forward(x); // (batch size) * 8 * (image width) * (image height)
        let x = self.dropout.forward(x);
        let x = self.conv2.forward(x); // (batch size) * 16 * (image width) * (image height)
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);
        let x = self.pool.forward(x); // (batch size) * 16 * 8 * 8
        let x = x.reshape([batch_size, 16 * 8 * 8]);
        let x = self.linear1.forward(x); // (batch size) * (hidden layer size)
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);
        self.linear2.forward(x) // (batch size) * (# of classes)
    }

    pub fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        labels: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss =
            CrossEntropyLoss::new(None, &output.device()).forward(output.clone(), labels.clone());

        ClassificationOutput::new(loss, output, labels)
    }
}
