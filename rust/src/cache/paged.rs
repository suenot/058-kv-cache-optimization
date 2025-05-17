//! PagedAttention-style KV-Cache implementation.
//!
//! Based on the vLLM paper: "Efficient Memory Management for Large Language
//! Model Serving with PagedAttention"

use ndarray::{Array4, Array5};
use std::collections::HashMap;

/// Paged KV-Cache for memory-efficient inference.
///
/// Uses a block-based memory management approach similar to OS virtual memory,
/// allowing non-contiguous storage of KV pairs and near-zero memory waste.
pub struct PagedKVCache {
    /// Size of each block in tokens
    block_size: usize,
    /// Total number of blocks available
    num_blocks: usize,
    /// Number of layers
    num_layers: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Dimension per head
    head_dim: usize,

    /// Pre-allocated key block pool: [num_blocks, num_layers, num_heads, block_size, head_dim]
    key_blocks: Array5<f32>,
    /// Pre-allocated value block pool
    value_blocks: Array5<f32>,

    /// List of free block indices
    free_blocks: Vec<usize>,

    /// Block tables: maps sequence_id -> list of block indices
    block_tables: HashMap<u64, Vec<usize>>,

    /// Current token count per sequence
    sequence_lengths: HashMap<u64, usize>,
}

impl PagedKVCache {
    /// Create a new paged KV-cache.
    ///
    /// # Arguments
    /// * `num_layers` - Number of transformer layers
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per attention head
    /// * `block_size` - Tokens per block (default 16)
    /// * `num_blocks` - Total blocks to allocate
    pub fn new(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
        num_blocks: usize,
    ) -> Self {
        // Pre-allocate block pools
        let key_blocks = Array5::zeros((num_blocks, num_layers, num_heads, block_size, head_dim));
        let value_blocks = Array5::zeros((num_blocks, num_layers, num_heads, block_size, head_dim));

        // Initialize free block list
        let free_blocks: Vec<usize> = (0..num_blocks).collect();

        Self {
            block_size,
            num_blocks,
            num_layers,
            num_heads,
            head_dim,
            key_blocks,
            value_blocks,
            free_blocks,
            block_tables: HashMap::new(),
            sequence_lengths: HashMap::new(),
        }
    }

    /// Allocate blocks for a new sequence.
    ///
    /// # Arguments
    /// * `sequence_id` - Unique identifier for the sequence
    /// * `num_tokens` - Initial number of tokens to allocate space for
    ///
    /// # Returns
    /// Vector of allocated block indices
    pub fn allocate_blocks(&mut self, sequence_id: u64, num_tokens: usize) -> Result<Vec<usize>, &'static str> {
        let num_blocks_needed = (num_tokens + self.block_size - 1) / self.block_size;

        if self.free_blocks.len() < num_blocks_needed {
            return Err("Not enough free blocks");
        }

        let mut allocated = Vec::with_capacity(num_blocks_needed);
        for _ in 0..num_blocks_needed {
            if let Some(block_idx) = self.free_blocks.pop() {
                allocated.push(block_idx);
            }
        }

        self.block_tables.insert(sequence_id, allocated.clone());
        self.sequence_lengths.insert(sequence_id, 0);

        Ok(allocated)
    }

    /// Free blocks when a sequence completes.
    pub fn free_sequence(&mut self, sequence_id: u64) {
        if let Some(blocks) = self.block_tables.remove(&sequence_id) {
            self.free_blocks.extend(blocks);
        }
        self.sequence_lengths.remove(&sequence_id);
    }

    /// Append tokens to a sequence's cache.
    ///
    /// # Arguments
    /// * `sequence_id` - Sequence to append to
    /// * `layer_idx` - Which transformer layer
    /// * `keys` - Key tensor for new tokens [num_tokens, num_heads, head_dim]
    /// * `values` - Value tensor for new tokens
    pub fn append_tokens(
        &mut self,
        sequence_id: u64,
        layer_idx: usize,
        keys: &ndarray::Array3<f32>,
        values: &ndarray::Array3<f32>,
    ) -> Result<(), &'static str> {
        let block_indices = self.block_tables.get_mut(&sequence_id)
            .ok_or("Sequence not found")?;
        let current_len = *self.sequence_lengths.get(&sequence_id).unwrap_or(&0);

        let num_new_tokens = keys.shape()[0];

        for i in 0..num_new_tokens {
            let global_pos = current_len + i;
            let block_idx_in_seq = global_pos / self.block_size;
            let pos_in_block = global_pos % self.block_size;

            // Allocate new block if needed
            while block_idx_in_seq >= block_indices.len() {
                if self.free_blocks.is_empty() {
                    return Err("No free blocks available");
                }
                let new_block = self.free_blocks.pop().unwrap();
                block_indices.push(new_block);
            }

            let physical_block = block_indices[block_idx_in_seq];

            // Copy key and value data
            for h in 0..self.num_heads {
                for d in 0..self.head_dim {
                    self.key_blocks[[physical_block, layer_idx, h, pos_in_block, d]] = keys[[i, h, d]];
                    self.value_blocks[[physical_block, layer_idx, h, pos_in_block, d]] = values[[i, h, d]];
                }
            }
        }

        *self.sequence_lengths.get_mut(&sequence_id).unwrap() += num_new_tokens;

        Ok(())
    }

    /// Get KV cache for a sequence.
    ///
    /// # Returns
    /// Tuple of (keys, values) tensors [seq_len, num_heads, head_dim]
    pub fn get_cache(
        &self,
        sequence_id: u64,
        layer_idx: usize,
    ) -> Result<(ndarray::Array3<f32>, ndarray::Array3<f32>), &'static str> {
        let block_indices = self.block_tables.get(&sequence_id)
            .ok_or("Sequence not found")?;
        let seq_len = *self.sequence_lengths.get(&sequence_id).unwrap_or(&0);

        let mut keys = ndarray::Array3::zeros((seq_len, self.num_heads, self.head_dim));
        let mut values = ndarray::Array3::zeros((seq_len, self.num_heads, self.head_dim));

        for pos in 0..seq_len {
            let block_idx_in_seq = pos / self.block_size;
            let pos_in_block = pos % self.block_size;
            let physical_block = block_indices[block_idx_in_seq];

            for h in 0..self.num_heads {
                for d in 0..self.head_dim {
                    keys[[pos, h, d]] = self.key_blocks[[physical_block, layer_idx, h, pos_in_block, d]];
                    values[[pos, h, d]] = self.value_blocks[[physical_block, layer_idx, h, pos_in_block, d]];
                }
            }
        }

        Ok((keys, values))
    }

    /// Get number of free blocks.
    pub fn free_block_count(&self) -> usize {
        self.free_blocks.len()
    }

    /// Get total memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let used_blocks = self.num_blocks - self.free_blocks.len();
        used_blocks * self.num_layers * self.num_heads * self.block_size * self.head_dim * 2 * std::mem::size_of::<f32>()
    }

    /// Get memory efficiency (used vs allocated).
    pub fn memory_efficiency(&self) -> f64 {
        let total_tokens: usize = self.sequence_lengths.values().sum();
        let allocated_tokens = (self.num_blocks - self.free_blocks.len()) * self.block_size;

        if allocated_tokens == 0 {
            1.0
        } else {
            total_tokens as f64 / allocated_tokens as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paged_cache_allocation() {
        let mut cache = PagedKVCache::new(4, 8, 64, 16, 100);

        // Allocate blocks for a sequence
        let blocks = cache.allocate_blocks(1, 50).unwrap();
        assert_eq!(blocks.len(), 4); // 50 tokens / 16 block_size = 4 blocks

        // Check free blocks reduced
        assert_eq!(cache.free_block_count(), 96);

        // Free sequence
        cache.free_sequence(1);
        assert_eq!(cache.free_block_count(), 100);
    }

    #[test]
    fn test_paged_cache_append() {
        let mut cache = PagedKVCache::new(4, 8, 64, 16, 100);

        // Allocate
        cache.allocate_blocks(1, 10).unwrap();

        // Append tokens
        let keys = ndarray::Array3::ones((5, 8, 64));
        let values = ndarray::Array3::ones((5, 8, 64));
        cache.append_tokens(1, 0, &keys, &values).unwrap();

        // Retrieve
        let (k, v) = cache.get_cache(1, 0).unwrap();
        assert_eq!(k.shape(), &[5, 8, 64]);
        assert_eq!(v.shape(), &[5, 8, 64]);
    }
}
