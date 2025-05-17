//! Multi-head attention with KV-cache support.

use ndarray::{Array2, Array3, Array4, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

/// Multi-head attention layer with KV-cache support.
pub struct KVCacheAttention {
    d_model: usize,
    n_heads: usize,
    head_dim: usize,

    /// Query projection weights
    w_q: Array2<f32>,
    /// Key projection weights
    w_k: Array2<f32>,
    /// Value projection weights
    w_v: Array2<f32>,
    /// Output projection weights
    w_o: Array2<f32>,

    scale: f32,
}

impl KVCacheAttention {
    /// Create a new attention layer.
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        assert!(d_model % n_heads == 0, "d_model must be divisible by n_heads");

        let head_dim = d_model / n_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Initialize weights with Xavier uniform
        let bound = (6.0 / (d_model + d_model) as f32).sqrt();
        let dist = Uniform::new(-bound, bound);

        let w_q = Array2::random((d_model, d_model), dist);
        let w_k = Array2::random((d_model, d_model), dist);
        let w_v = Array2::random((d_model, d_model), dist);
        let w_o = Array2::random((d_model, d_model), dist);

        Self {
            d_model,
            n_heads,
            head_dim,
            w_q,
            w_k,
            w_v,
            w_o,
            scale,
        }
    }

    /// Forward pass with KV-cache support.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, d_model]
    /// * `past_kv` - Optional cached (keys, values) from previous steps
    ///
    /// # Returns
    /// * Output tensor [batch, seq_len, d_model]
    /// * Updated cache (keys, values) [batch, n_heads, total_seq_len, head_dim]
    pub fn forward(
        &self,
        x: &Array3<f32>,
        past_kv: Option<(Array4<f32>, Array4<f32>)>,
    ) -> (Array3<f32>, (Array4<f32>, Array4<f32>)) {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];

        // Project to Q, K, V
        let q = self.linear_3d(x, &self.w_q);
        let k = self.linear_3d(x, &self.w_k);
        let v = self.linear_3d(x, &self.w_v);

        // Reshape to multi-head format: [batch, seq_len, n_heads, head_dim]
        // Then transpose to: [batch, n_heads, seq_len, head_dim]
        let q = self.reshape_for_attention(&q, batch_size, seq_len);
        let k = self.reshape_for_attention(&k, batch_size, seq_len);
        let v = self.reshape_for_attention(&v, batch_size, seq_len);

        // Concatenate with cached K, V if present
        let (k, v) = if let Some((past_k, past_v)) = past_kv {
            let k = ndarray::concatenate(Axis(2), &[past_k.view(), k.view()])
                .expect("K concatenation failed");
            let v = ndarray::concatenate(Axis(2), &[past_v.view(), v.view()])
                .expect("V concatenation failed");
            (k, v)
        } else {
            (k, v)
        };

        // Compute attention scores
        let scores = self.batch_matmul_4d(&q, &self.transpose_last_two(&k));
        let scores = scores.mapv(|v| v * self.scale);

        // Softmax
        let attn_weights = self.softmax_last_dim(&scores);

        // Apply attention to values
        let context = self.batch_matmul_4d(&attn_weights, &v);

        // Reshape back: [batch, n_heads, seq_len, head_dim] -> [batch, seq_len, d_model]
        let output = self.reshape_from_attention(&context, batch_size, seq_len);

        // Output projection
        let output = self.linear_3d(&output, &self.w_o);

        (output, (k, v))
    }

    /// Apply linear transformation to 3D tensor.
    fn linear_3d(&self, x: &Array3<f32>, w: &Array2<f32>) -> Array3<f32> {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];
        let mut output = Array3::zeros((batch_size, seq_len, self.d_model));

        for b in 0..batch_size {
            for s in 0..seq_len {
                for o in 0..self.d_model {
                    let mut sum = 0.0;
                    for i in 0..self.d_model {
                        sum += x[[b, s, i]] * w[[i, o]];
                    }
                    output[[b, s, o]] = sum;
                }
            }
        }

        output
    }

    /// Reshape tensor for multi-head attention.
    fn reshape_for_attention(
        &self,
        x: &Array3<f32>,
        batch_size: usize,
        seq_len: usize,
    ) -> Array4<f32> {
        let mut output = Array4::zeros((batch_size, self.n_heads, seq_len, self.head_dim));

        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..self.n_heads {
                    for d in 0..self.head_dim {
                        output[[b, h, s, d]] = x[[b, s, h * self.head_dim + d]];
                    }
                }
            }
        }

        output
    }

    /// Reshape tensor from multi-head format back to standard format.
    fn reshape_from_attention(
        &self,
        x: &Array4<f32>,
        batch_size: usize,
        seq_len: usize,
    ) -> Array3<f32> {
        let mut output = Array3::zeros((batch_size, seq_len, self.d_model));

        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..self.n_heads {
                    for d in 0..self.head_dim {
                        output[[b, s, h * self.head_dim + d]] = x[[b, h, s, d]];
                    }
                }
            }
        }

        output
    }

    /// Transpose last two dimensions of 4D tensor.
    fn transpose_last_two(&self, x: &Array4<f32>) -> Array4<f32> {
        x.clone().permuted_axes([0, 1, 3, 2])
    }

    /// Batch matrix multiplication for 4D tensors.
    fn batch_matmul_4d(&self, a: &Array4<f32>, b: &Array4<f32>) -> Array4<f32> {
        let batch_size = a.shape()[0];
        let n_heads = a.shape()[1];
        let m = a.shape()[2];
        let k = a.shape()[3];
        let n = b.shape()[3];

        let mut output = Array4::zeros((batch_size, n_heads, m, n));

        for batch in 0..batch_size {
            for head in 0..n_heads {
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0;
                        for kk in 0..k {
                            sum += a[[batch, head, i, kk]] * b[[batch, head, kk, j]];
                        }
                        output[[batch, head, i, j]] = sum;
                    }
                }
            }
        }

        output
    }

    /// Softmax along the last dimension.
    fn softmax_last_dim(&self, x: &Array4<f32>) -> Array4<f32> {
        let mut output = x.clone();
        let shape = x.shape();

        for b in 0..shape[0] {
            for h in 0..shape[1] {
                for i in 0..shape[2] {
                    // Find max for numerical stability
                    let mut max_val = f32::NEG_INFINITY;
                    for j in 0..shape[3] {
                        max_val = max_val.max(x[[b, h, i, j]]);
                    }

                    // Compute exp and sum
                    let mut sum = 0.0;
                    for j in 0..shape[3] {
                        let exp_val = (x[[b, h, i, j]] - max_val).exp();
                        output[[b, h, i, j]] = exp_val;
                        sum += exp_val;
                    }

                    // Normalize
                    for j in 0..shape[3] {
                        output[[b, h, i, j]] /= sum;
                    }
                }
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_forward() {
        let attn = KVCacheAttention::new(64, 4);

        // Create input
        let x = Array3::zeros((1, 10, 64));

        // Forward pass
        let (output, (k, v)) = attn.forward(&x, None);

        assert_eq!(output.shape(), &[1, 10, 64]);
        assert_eq!(k.shape(), &[1, 4, 10, 16]);
        assert_eq!(v.shape(), &[1, 4, 10, 16]);
    }

    #[test]
    fn test_attention_with_cache() {
        let attn = KVCacheAttention::new(64, 4);

        // First forward pass
        let x1 = Array3::zeros((1, 10, 64));
        let (_, (k1, v1)) = attn.forward(&x1, None);

        // Second forward pass with cache
        let x2 = Array3::zeros((1, 1, 64));
        let (output, (k2, v2)) = attn.forward(&x2, Some((k1, v1)));

        // Cache should have grown
        assert_eq!(k2.shape(), &[1, 4, 11, 16]); // 10 + 1 = 11
        assert_eq!(v2.shape(), &[1, 4, 11, 16]);
    }
}
