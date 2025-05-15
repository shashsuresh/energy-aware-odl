// Just some configs for the runner
pub struct Config {
    pub max_memory: usize,
    pub model: Models,
}

pub enum Models {
    MCUnet,
    MobileNetV2,
    ProxylessNAS,
}

impl Models {
    pub fn get_last_layer_idx(&self) -> usize {
        match self {
            Models::MCUnet => 42,
            Models::MobileNetV2 => todo!(),
            Models::ProxylessNAS => todo!(),
        }
    }
}
