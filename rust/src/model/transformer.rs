//! Transformer model for trading with KV-cache support.

use ndarray::{Array1, Array2, Array3, Array4};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

use crate::cache::KVCacheConfig;
use super::attention::KVCacheAttention;

/// Transformer model for trading with optimized KV-cache inference.
pub struct KVCacheTrader {
    input_dim: usize,
    d_model: usize,
    n_heads: usize,
    n_layers: usize,
    max_seq_len: usize,

    /// Input projection
    input_proj: Array2<f32>,

    /// Positional encoding
    pos_encoding: Array2<f32>,

    /// Attention layers
    attention_layers: Vec<KVCacheAttention>,

    /// Feed-forward layers (simplified)
    ff_weights_1: Vec<Array2<f32>>,
    ff_weights_2: Vec<Array2<f32>>,

    /// Layer norms (simplified as scale factors)
    layer_norm_scales: Vec<f32>,

    /// Output head
    output_weight: Array2<f32>,

    /// Configuration
    config: KVCacheConfig,
}

impl KVCacheTrader {
    /// Create a new trading model.
    pub fn new(
        input_dim: usize,
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        config: KVCacheConfig,
    ) -> anyhow::Result<Self> {
        let max_seq_len = config.max_cache_size;
        let d_ff = d_model * 4;

        // Initialize weights
        let bound = (6.0 / (input_dim + d_model) as f32).sqrt();
        let dist = Uniform::new(-bound, bound);

        let input_proj = Array2::random((input_dim, d_model), dist);

        // Sinusoidal positional encoding
        let pos_encoding = Self::create_positional_encoding(max_seq_len, d_model);

        // Create layers
        let attention_layers: Vec<_> = (0..n_layers)
            .map(|_| KVCacheAttention::new(d_model, n_heads))
            .collect();

        let ff_dist = Uniform::new(-0.1, 0.1);
        let ff_weights_1: Vec<_> = (0..n_layers)
            .map(|_| Array2::random((d_model, d_ff), ff_dist))
            .collect();

        let ff_weights_2: Vec<_> = (0..n_layers)
            .map(|_| Array2::random((d_ff, d_model), ff_dist))
            .collect();

        let layer_norm_scales = vec![1.0; n_layers * 2];

        let output_weight = Array2::random((d_model, 1), ff_dist);

        Ok(Self {
            input_dim,
            d_model,
            n_heads,
            n_layers,
            max_seq_len,
            input_proj,
            pos_encoding,
            attention_layers,
            ff_weights_1,
            ff_weights_2,
            layer_norm_scales,
            output_weight,
            config,
        })
    }

    /// Create sinusoidal positional encoding.
    fn create_positional_encoding(max_len: usize, d_model: usize) -> Array2<f32> {
        let mut pe = Array2::zeros((max_len, d_model));

        for pos in 0..max_len {
            for i in 0..d_model / 2 {
                let angle = pos as f32 / (10000_f32).powf(2.0 * i as f32 / d_model as f32);
                pe[[pos, 2 * i]] = angle.sin();
                pe[[pos, 2 * i + 1]] = angle.cos();
            }
        }

        pe
    }

    /// Forward pass with KV-cache support.
    ///
    /// # Arguments
    /// * `features` - Input features [input_dim]
    /// * `past_kv_cache` - Optional cached KV pairs from previous steps
    ///
    /// # Returns
    /// * Prediction value
    /// * Updated KV cache for all layers
    pub fn forward_with_cache(
        &self,
        features: &[f32],
        past_kv_cache: Option<Vec<(Array4<f32>, Array4<f32>)>>,
    ) -> anyhow::Result<(f32, Vec<(Array4<f32>, Array4<f32>)>)> {
        // Convert features to array
        let x = Array1::from(features.to_vec());

        // Project input: [input_dim] -> [d_model]
        let mut hidden = Array1::zeros(self.d_model);
        for o in 0..self.d_model {
            let mut sum = 0.0;
            for i in 0..self.input_dim {
                sum += x[i] * self.input_proj[[i, o]];
            }
            hidden[o] = sum;
        }

        // Determine position
        let past_length = past_kv_cache.as_ref()
            .and_then(|c| c.first())
            .map(|(k, _)| k.shape()[2])
            .unwrap_or(0);

        // Add positional encoding
        for i in 0..self.d_model {
            hidden[i] += self.pos_encoding[[past_length, i]];
        }

        // Reshape to [batch=1, seq_len=1, d_model]
        let mut x = hidden.into_shape_with_order((1, 1, self.d_model))?;

        // Initialize cache list
        let mut past_cache = past_kv_cache.unwrap_or_else(|| vec![
            (Array4::zeros((0, 0, 0, 0)), Array4::zeros((0, 0, 0, 0)));
            self.n_layers
        ]);

        let mut present_cache = Vec::with_capacity(self.n_layers);

        // Forward through transformer layers
        for (layer_idx, attn) in self.attention_layers.iter().enumerate() {
            // Get past KV for this layer
            let past_kv = if past_cache[layer_idx].0.len() > 0 {
                Some((past_cache[layer_idx].0.clone(), past_cache[layer_idx].1.clone()))
            } else {
                None
            };

            // Self-attention with cache
            let (attn_out, (k, v)) = attn.forward(&x, past_kv);
            present_cache.push((k, v));

            // Residual connection + simplified layer norm
            let residual = x.clone();
            x = self.add_and_norm(&residual, &attn_out);

            // Feed-forward network
            let ff_out = self.feed_forward(&x, layer_idx);

            // Residual connection + layer norm
            let residual = x.clone();
            x = self.add_and_norm(&residual, &ff_out);
        }

        // Get last token's hidden state: [batch=1, seq_len=1, d_model] -> [d_model]
        let last_hidden = x.into_shape_with_order(self.d_model)?;

        // Output projection
        let mut output = 0.0;
        for i in 0..self.d_model {
            output += last_hidden[i] * self.output_weight[[i, 0]];
        }

        Ok((output, present_cache))
    }

    /// Add two tensors and apply simplified layer normalization.
    fn add_and_norm(&self, a: &Array3<f32>, b: &Array3<f32>) -> Array3<f32> {
        let mut result = a + b;

        // Simplified layer norm per position
        let batch_size = result.shape()[0];
        let seq_len = result.shape()[1];

        for batch in 0..batch_size {
            for seq in 0..seq_len {
                // Compute mean and variance
                let mut mean = 0.0;
                for i in 0..self.d_model {
                    mean += result[[batch, seq, i]];
                }
                mean /= self.d_model as f32;

                let mut var = 0.0;
                for i in 0..self.d_model {
                    let diff = result[[batch, seq, i]] - mean;
                    var += diff * diff;
                }
                var /= self.d_model as f32;

                // Normalize
                let std = (var + 1e-5).sqrt();
                for i in 0..self.d_model {
                    result[[batch, seq, i]] = (result[[batch, seq, i]] - mean) / std;
                }
            }
        }

        result
    }

    /// Feed-forward network.
    fn feed_forward(&self, x: &Array3<f32>, layer_idx: usize) -> Array3<f32> {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];
        let d_ff = self.ff_weights_1[layer_idx].shape()[1];

        // First linear + GELU
        let mut hidden = Array3::zeros((batch_size, seq_len, d_ff));
        for b in 0..batch_size {
            for s in 0..seq_len {
                for o in 0..d_ff {
                    let mut sum = 0.0;
                    for i in 0..self.d_model {
                        sum += x[[b, s, i]] * self.ff_weights_1[layer_idx][[i, o]];
                    }
                    // GELU activation (approximation)
                    hidden[[b, s, o]] = sum * 0.5 * (1.0 + (sum * 0.7978845608 * (1.0 + 0.044715 * sum * sum)).tanh());
                }
            }
        }

        // Second linear
        let mut output = Array3::zeros((batch_size, seq_len, self.d_model));
        for b in 0..batch_size {
            for s in 0..seq_len {
                for o in 0..self.d_model {
                    let mut sum = 0.0;
                    for i in 0..d_ff {
                        sum += hidden[[b, s, i]] * self.ff_weights_2[layer_idx][[i, o]];
                    }
                    output[[b, s, o]] = sum;
                }
            }
        }

        output
    }

    /// Get model configuration.
    pub fn config(&self) -> &KVCacheConfig {
        &self.config
    }

    /// Get number of layers.
    pub fn num_layers(&self) -> usize {
        self.n_layers
    }

    /// Get number of attention heads.
    pub fn num_heads(&self) -> usize {
        self.n_heads
    }

    /// Get head dimension.
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_forward() {
        let config = KVCacheConfig::default();
        let model = KVCacheTrader::new(5, 64, 4, 2, config).unwrap();

        let features = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let (output, cache) = model.forward_with_cache(&features, None).unwrap();

        // Output should be a single value
        assert!(output.is_finite());

        // Cache should have entries for all layers
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_model_incremental() {
        let config = KVCacheConfig::default();
        let model = KVCacheTrader::new(5, 64, 4, 2, config).unwrap();

        // First forward pass
        let features1 = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let (_, cache1) = model.forward_with_cache(&features1, None).unwrap();

        // Verify cache shape
        assert_eq!(cache1[0].0.shape()[2], 1); // 1 token cached

        // Second forward pass with cache
        let features2 = vec![0.2, 0.3, 0.4, 0.5, 0.6];
        let (_, cache2) = model.forward_with_cache(&features2, Some(cache1)).unwrap();

        // Cache should have grown
        assert_eq!(cache2[0].0.shape()[2], 2); // 2 tokens cached
    }
}
