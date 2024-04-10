use burn::record::{HalfPrecisionSettings, NamedMpkGzFileRecorder};

pub mod data;
pub mod infer;
pub mod model;
pub mod train;

const IMAGE_WIDTH: usize = 28;
const IMAGE_HEIGHT: usize = 28;

type Recorder = NamedMpkGzFileRecorder<HalfPrecisionSettings>;
