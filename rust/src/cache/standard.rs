//! Standard KV-Cache implementation.

use ndarray::{Array4, Axis};
use std::collections::VecDeque;

/// Standard Key-Value Cache for Transformer inference.
///
/// Structure: [batch_size, num_heads, seq_len, head_dim]
///
/// For trading models:
/// - batch_size: Number of different assets or scenarios
/// - num_heads: Attention heads (captures different patterns)
/// - seq_len: Historical context length (grows during inference)
/// - head_dim: Dimension per attention head
pub struct KVCache {
    num_layers: usize,
    keys: Vec<Option<Array4<f32>>>,
    values: Vec<Option<Array4<f32>>>,
    max_seq_len: Option<usize>,
    current_seq_len: usize,
}

impl KVCache {
    /// Create a new KV-cache.
    pub fn new(num_layers: usize, max_seq_len: Option<usize>) -> Self {
        Self {
            num_layers,
            keys: vec![None; num_layers],
            values: vec![None; num_layers],
            max_seq_len,
            current_seq_len: 0,
        }
    }

    /// Update cache for a specific layer.
    ///
    /// # Arguments
    /// * `layer_idx` - Which transformer layer
    /// * `new_keys` - New key tensor [batch, num_heads, new_tokens, head_dim]
    /// * `new_values` - New value tensor [batch, num_heads, new_tokens, head_dim]
    pub fn update(
        &mut self,
        layer_idx: usize,
        new_keys: Array4<f32>,
        new_values: Array4<f32>,
    ) {
        if let Some(ref mut existing_keys) = self.keys[layer_idx] {
            // Concatenate along sequence dimension (axis 2)
            let combined_keys = ndarray::concatenate(
                Axis(2),
                &[existing_keys.view(), new_keys.view()],
            ).expect("Key concatenation failed");
            *existing_keys = combined_keys;
        } else {
            self.keys[layer_idx] = Some(new_keys);
        }

        if let Some(ref mut existing_values) = self.values[layer_idx] {
            let combined_values = ndarray::concatenate(
                Axis(2),
                &[existing_values.view(), new_values.view()],
            ).expect("Value concatenation failed");
            *existing_values = combined_values;
        } else {
            self.values[layer_idx] = Some(new_values);
        }

        // Update current sequence length
        if let Some(ref keys) = self.keys[layer_idx] {
            self.current_seq_len = keys.shape()[2];
        }
    }

    /// Get cached keys and values for a layer.
    pub fn get(&self, layer_idx: usize) -> (Option<&Array4<f32>>, Option<&Array4<f32>>) {
        (self.keys[layer_idx].as_ref(), self.values[layer_idx].as_ref())
    }

    /// Get current sequence length.
    pub fn seq_len(&self) -> usize {
        self.current_seq_len
    }

    /// Calculate total memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let key_size: usize = self.keys.iter()
            .filter_map(|k| k.as_ref())
            .map(|k| k.len() * std::mem::size_of::<f32>())
            .sum();

        let value_size: usize = self.values.iter()
            .filter_map(|v| v.as_ref())
            .map(|v| v.len() * std::mem::size_of::<f32>())
            .sum();

        key_size + value_size
    }

    /// Truncate cache to keep only the last N tokens.
    pub fn truncate(&mut self, keep_last: usize) {
        for i in 0..self.num_layers {
            if let Some(ref mut keys) = self.keys[i] {
                let seq_len = keys.shape()[2];
                if seq_len > keep_last {
                    let start = seq_len - keep_last;
                    *keys = keys.slice(ndarray::s![.., .., start.., ..]).to_owned();
                }
            }
            if let Some(ref mut values) = self.values[i] {
                let seq_len = values.shape()[2];
                if seq_len > keep_last {
                    let start = seq_len - keep_last;
                    *values = values.slice(ndarray::s![.., .., start.., ..]).to_owned();
                }
            }
        }
        self.current_seq_len = keep_last.min(self.current_seq_len);
    }

    /// Clear all cached data.
    pub fn clear(&mut self) {
        for i in 0..self.num_layers {
            self.keys[i] = None;
            self.values[i] = None;
        }
        self.current_seq_len = 0;
    }

    /// Export cache as vector of (keys, values) tuples.
    pub fn export(&self) -> Vec<(Option<Array4<f32>>, Option<Array4<f32>>)> {
        self.keys.iter()
            .zip(self.values.iter())
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

/// Sliding window KV-cache with fixed maximum length.
pub struct SlidingWindowKVCache {
    inner: KVCache,
    window_size: usize,
}

impl SlidingWindowKVCache {
    /// Create a new sliding window cache.
    pub fn new(num_layers: usize, window_size: usize) -> Self {
        Self {
            inner: KVCache::new(num_layers, Some(window_size)),
            window_size,
        }
    }

    /// Update cache, automatically truncating if window is exceeded.
    pub fn update(
        &mut self,
        layer_idx: usize,
        new_keys: Array4<f32>,
        new_values: Array4<f32>,
    ) {
        self.inner.update(layer_idx, new_keys, new_values);

        // Truncate if necessary
        if self.inner.seq_len() > self.window_size {
            self.inner.truncate(self.window_size);
        }
    }

    /// Get cached keys and values.
    pub fn get(&self, layer_idx: usize) -> (Option<&Array4<f32>>, Option<&Array4<f32>>) {
        self.inner.get(layer_idx)
    }

    /// Get current sequence length.
    pub fn seq_len(&self) -> usize {
        self.inner.seq_len()
    }

    /// Get memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.inner.memory_usage()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;

    #[test]
    fn test_kv_cache_update() {
        let mut cache = KVCache::new(2, None);

        // Initial update
        let k1 = Array4::zeros((1, 4, 10, 32));
        let v1 = Array4::zeros((1, 4, 10, 32));
        cache.update(0, k1, v1);

        assert_eq!(cache.seq_len(), 10);

        // Incremental update
        let k2 = Array4::zeros((1, 4, 1, 32));
        let v2 = Array4::zeros((1, 4, 1, 32));
        cache.update(0, k2, v2);

        assert_eq!(cache.seq_len(), 11);
    }

    #[test]
    fn test_kv_cache_truncate() {
        let mut cache = KVCache::new(2, None);

        let k = Array4::zeros((1, 4, 100, 32));
        let v = Array4::zeros((1, 4, 100, 32));
        cache.update(0, k, v);

        cache.truncate(50);
        assert_eq!(cache.seq_len(), 50);
    }

    #[test]
    fn test_sliding_window() {
        let mut cache = SlidingWindowKVCache::new(2, 10);

        for _ in 0..15 {
            let k = Array4::zeros((1, 4, 1, 32));
            let v = Array4::zeros((1, 4, 1, 32));
            cache.update(0, k, v);
        }

        // Should be capped at window size
        assert_eq!(cache.seq_len(), 10);
    }
}
