//! Transformer model with KV-cache support.

mod attention;
mod transformer;

pub use attention::KVCacheAttention;
pub use transformer::KVCacheTrader;

/// Prediction result from the model.
#[derive(Debug, Clone)]
pub struct Prediction {
    /// Predicted value (e.g., expected return)
    pub value: f32,
    /// Inference latency in milliseconds
    pub latency_ms: f64,
    /// Current context length
    pub context_length: usize,
}

/// Output type for the trading model.
#[derive(Debug, Clone, Copy)]
pub enum OutputType {
    /// Regression output (continuous value)
    Regression,
    /// Direction classification (long/short/neutral)
    Direction,
    /// Portfolio allocation weights
    Allocation,
}

impl Default for OutputType {
    fn default() -> Self {
        Self::Regression
    }
}
