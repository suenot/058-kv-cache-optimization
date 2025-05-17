//! KV-Cache Optimization for Algorithmic Trading
//!
//! This crate provides efficient KV-cache implementations for Transformer-based
//! trading systems, enabling low-latency inference for real-time market prediction.
//!
//! # Features
//!
//! - **Standard KV-Cache**: Basic implementation with append and retrieval
//! - **Paged KV-Cache**: Memory-efficient paged attention implementation
//! - **Quantized KV-Cache**: INT8/FP8 quantization for reduced memory footprint
//! - **Trading Model**: Transformer with KV-cache support for market prediction
//! - **Bybit Integration**: Real-time market data fetching
//!
//! # Example
//!
//! ```no_run
//! use kv_cache_trading::{KVCacheTrader, KVCacheConfig, StreamingEngine};
//!
//! # fn main() -> anyhow::Result<()> {
//! // Create model with KV-cache
//! let config = KVCacheConfig::default();
//! let model = KVCacheTrader::new(5, 256, 8, 6, config)?;
//!
//! // Create streaming engine
//! let mut engine = StreamingEngine::new(model, 4096);
//!
//! // Process market data
//! let features = vec![0.001, 0.02, 1.1, 0.05, 0.5];
//! let prediction = engine.process_tick(&features)?;
//!
//! println!("Prediction: {}, Latency: {} ms", prediction.value, prediction.latency_ms);
//! # Ok(())
//! # }
//! ```

pub mod cache;
pub mod model;
pub mod data;
pub mod strategy;

// Re-exports
pub use cache::{KVCache, KVCacheConfig, PagedKVCache, QuantizedKVCache};
pub use model::{KVCacheTrader, KVCacheAttention, Prediction};
pub use data::{BybitClient, MarketData, KlineData};
pub use strategy::{Backtester, BacktestResult, TradingSignal};

/// Streaming inference engine for real-time trading.
pub struct StreamingEngine {
    model: KVCacheTrader,
    cache: Option<Vec<(ndarray::Array4<f32>, ndarray::Array4<f32>)>>,
    current_length: usize,
    max_context_length: usize,
}

impl StreamingEngine {
    /// Create a new streaming engine.
    pub fn new(model: KVCacheTrader, max_context_length: usize) -> Self {
        Self {
            model,
            cache: None,
            current_length: 0,
            max_context_length,
        }
    }

    /// Reset the streaming state.
    pub fn reset(&mut self) {
        self.cache = None;
        self.current_length = 0;
    }

    /// Process a single tick of market data.
    pub fn process_tick(&mut self, features: &[f32]) -> anyhow::Result<Prediction> {
        use std::time::Instant;

        let start = Instant::now();

        // Handle context window overflow
        if self.current_length >= self.max_context_length {
            if let Some(ref mut cache) = self.cache {
                // Truncate cache (sliding window)
                for (k, v) in cache.iter_mut() {
                    // Remove first token from each layer's cache
                    let k_shape = k.shape();
                    let v_shape = v.shape();
                    if k_shape[2] > 1 {
                        *k = k.slice(ndarray::s![.., .., 1.., ..]).to_owned();
                        *v = v.slice(ndarray::s![.., .., 1.., ..]).to_owned();
                    }
                }
                self.current_length -= 1;
            }
        }

        // Forward pass
        let (output, new_cache) = self.model.forward_with_cache(
            features,
            self.cache.take(),
        )?;

        self.cache = Some(new_cache);
        self.current_length += 1;

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(Prediction {
            value: output,
            latency_ms,
            context_length: self.current_length,
        })
    }

    /// Get current cache memory usage in bytes.
    pub fn cache_memory_bytes(&self) -> usize {
        self.cache.as_ref().map_or(0, |cache| {
            cache.iter().map(|(k, v)| {
                k.len() * std::mem::size_of::<f32>() +
                v.len() * std::mem::size_of::<f32>()
            }).sum()
        })
    }
}

/// Optimized inference engine for batch processing.
pub struct OptimizedInferenceEngine {
    model: KVCacheTrader,
    request_caches: std::collections::HashMap<String, Vec<(ndarray::Array4<f32>, ndarray::Array4<f32>)>>,
    max_cache_memory_bytes: usize,
    cache_hits: usize,
    cache_misses: usize,
}

impl OptimizedInferenceEngine {
    /// Create a new optimized inference engine.
    pub fn new(model: KVCacheTrader, max_cache_memory_mb: f64) -> Self {
        Self {
            model,
            request_caches: std::collections::HashMap::new(),
            max_cache_memory_bytes: (max_cache_memory_mb * 1024.0 * 1024.0) as usize,
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Make prediction for a single request.
    pub fn predict(&mut self, request_id: &str, features: &[f32]) -> anyhow::Result<Prediction> {
        use std::time::Instant;

        let start = Instant::now();

        let past_cache = self.request_caches.remove(request_id);
        let cache_hit = past_cache.is_some();

        if cache_hit {
            self.cache_hits += 1;
        } else {
            self.cache_misses += 1;
        }

        let (output, new_cache) = self.model.forward_with_cache(features, past_cache)?;

        // Store updated cache
        self.request_caches.insert(request_id.to_string(), new_cache);

        // Evict old caches if memory limit exceeded
        self.evict_if_needed();

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(Prediction {
            value: output,
            latency_ms,
            context_length: 1,
        })
    }

    /// Clear cache for a specific request.
    pub fn clear_cache(&mut self, request_id: &str) {
        self.request_caches.remove(request_id);
    }

    /// Clear all caches.
    pub fn clear_all_caches(&mut self) {
        self.request_caches.clear();
    }

    /// Get cache hit rate.
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }

    fn evict_if_needed(&mut self) {
        let mut total_memory: usize = self.request_caches.values().map(|cache| {
            cache.iter().map(|(k, v)| {
                k.len() * std::mem::size_of::<f32>() +
                v.len() * std::mem::size_of::<f32>()
            }).sum::<usize>()
        }).sum();

        while total_memory > self.max_cache_memory_bytes && !self.request_caches.is_empty() {
            // Remove first entry (simple eviction strategy)
            if let Some(key) = self.request_caches.keys().next().cloned() {
                if let Some(cache) = self.request_caches.remove(&key) {
                    let cache_size: usize = cache.iter().map(|(k, v)| {
                        k.len() * std::mem::size_of::<f32>() +
                        v.len() * std::mem::size_of::<f32>()
                    }).sum();
                    total_memory -= cache_size;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_engine() {
        let config = KVCacheConfig::default();
        let model = KVCacheTrader::new(5, 64, 4, 2, config).unwrap();
        let mut engine = StreamingEngine::new(model, 100);

        // Process multiple ticks
        for _ in 0..10 {
            let features = vec![0.001, 0.02, 1.1, 0.05, 0.5];
            let result = engine.process_tick(&features).unwrap();
            assert!(result.latency_ms >= 0.0);
        }

        assert_eq!(engine.current_length, 10);
    }

    #[test]
    fn test_optimized_engine() {
        let config = KVCacheConfig::default();
        let model = KVCacheTrader::new(5, 64, 4, 2, config).unwrap();
        let mut engine = OptimizedInferenceEngine::new(model, 100.0);

        // First request (cache miss)
        let features = vec![0.001, 0.02, 1.1, 0.05, 0.5];
        let _ = engine.predict("req1", &features).unwrap();

        // Second request (cache hit)
        let _ = engine.predict("req1", &features).unwrap();

        assert_eq!(engine.cache_hits, 1);
        assert_eq!(engine.cache_misses, 1);
    }
}
