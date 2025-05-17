//! KV-Cache implementations for efficient Transformer inference.

mod standard;
mod paged;
mod quantized;

pub use standard::KVCache;
pub use paged::PagedKVCache;
pub use quantized::QuantizedKVCache;

/// Configuration for KV-cache optimization.
#[derive(Debug, Clone)]
pub struct KVCacheConfig {
    /// Cache type: "standard", "paged", "quantized", "selective"
    pub cache_type: String,
    /// Maximum cache size in tokens
    pub max_cache_size: usize,
    /// Block size for paged attention
    pub block_size: usize,
    /// Quantization type: "fp16", "fp8", "int8"
    pub quantization: String,
    /// Retention strategy for selective caching
    pub retention_strategy: String,
}

impl Default for KVCacheConfig {
    fn default() -> Self {
        Self {
            cache_type: "standard".to_string(),
            max_cache_size: 4096,
            block_size: 16,
            quantization: "fp16".to_string(),
            retention_strategy: "attention_score".to_string(),
        }
    }
}

impl KVCacheConfig {
    /// Create a new configuration with paged attention.
    pub fn paged(block_size: usize) -> Self {
        Self {
            cache_type: "paged".to_string(),
            block_size,
            ..Default::default()
        }
    }

    /// Create a new configuration with quantization.
    pub fn quantized(quantization: &str) -> Self {
        Self {
            cache_type: "quantized".to_string(),
            quantization: quantization.to_string(),
            ..Default::default()
        }
    }

    /// Create a new configuration with selective retention.
    pub fn selective(max_cache_size: usize, strategy: &str) -> Self {
        Self {
            cache_type: "selective".to_string(),
            max_cache_size,
            retention_strategy: strategy.to_string(),
            ..Default::default()
        }
    }
}
