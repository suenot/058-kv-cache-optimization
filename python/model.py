"""
KV-Cache Transformer Model for Trading

This module implements a Transformer model with optimized KV-cache support
for efficient inference in algorithmic trading applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass


class KVCache:
    """
    Key-Value Cache for efficient Transformer inference.

    Structure: [batch_size, num_heads, seq_len, head_dim]

    For trading models:
    - batch_size: Number of different assets or scenarios
    - num_heads: Attention heads (captures different patterns)
    - seq_len: Historical context length (grows during inference)
    - head_dim: Dimension per attention head
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        max_seq_len: Optional[int] = None
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.keys = [None] * num_layers
        self.values = [None] * num_layers

        # Pre-allocate for known maximum length (optional optimization)
        self.max_seq_len = max_seq_len
        self.current_seq_len = 0

    def update(self, layer_idx: int, new_keys: torch.Tensor, new_values: torch.Tensor):
        """
        Append new keys and values to the cache.

        Args:
            layer_idx: Which transformer layer
            new_keys: [batch, num_heads, new_tokens, head_dim]
            new_values: [batch, num_heads, new_tokens, head_dim]
        """
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = new_keys
            self.values[layer_idx] = new_values
        else:
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], new_keys], dim=2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], new_values], dim=2)

        self.current_seq_len = self.keys[layer_idx].shape[2]

    def get(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Retrieve cached keys and values for a layer."""
        return self.keys[layer_idx], self.values[layer_idx]

    def memory_usage(self) -> int:
        """Calculate total memory usage in bytes."""
        total = 0
        for k, v in zip(self.keys, self.values):
            if k is not None:
                total += k.numel() * k.element_size()
                total += v.numel() * v.element_size()
        return total

    def truncate(self, keep_last: int):
        """Truncate cache to keep only the last N tokens."""
        for i in range(self.num_layers):
            if self.keys[i] is not None:
                self.keys[i] = self.keys[i][:, :, -keep_last:, :]
                self.values[i] = self.values[i][:, :, -keep_last:, :]
        self.current_seq_len = keep_last


@dataclass
class KVCacheConfig:
    """Configuration for KV-cache optimization."""
    cache_type: str = 'standard'  # 'standard', 'paged', 'quantized', 'selective'
    max_cache_size: int = 4096
    block_size: int = 16  # For paged attention
    quantization: str = 'fp16'  # 'fp16', 'fp8', 'int8'
    retention_strategy: str = 'attention_score'


class KVCacheAttention(nn.Module):
    """
    Multi-head attention with KV-cache support.

    Optimized for incremental inference in trading applications.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        cache_config: Optional[KVCacheConfig] = None
    ):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.cache_config = cache_config or KVCacheConfig()

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with KV-cache support.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            past_kv: Cached (keys, values) from previous steps
            use_cache: Whether to return updated cache
            attention_mask: Optional attention mask

        Returns:
            output: Attention output
            present_kv: Updated cache (if use_cache=True)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Transpose for attention: [batch, n_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Concatenate with cached K, V
        if past_kv is not None:
            past_K, past_V = past_kv
            K = torch.cat([past_K, K], dim=2)
            V = torch.cat([past_V, V], dim=2)

        # Compute attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)

        # Prepare cache
        present_kv = (K, V) if use_cache else None

        return output, present_kv


class KVCacheTransformerBlock(nn.Module):
    """Transformer block with KV-cache support."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        cache_config: Optional[KVCacheConfig] = None
    ):
        super().__init__()

        self.attention = KVCacheAttention(d_model, n_heads, dropout, cache_config)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        # Pre-norm attention
        residual = x
        x = self.norm1(x)
        attn_out, present_kv = self.attention(x, past_kv, use_cache, attention_mask)
        x = residual + attn_out

        # Pre-norm FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.ff(x)

        return x, present_kv


class KVCacheTrader(nn.Module):
    """
    Transformer model for trading with optimized KV-cache inference.

    Key features:
    - Efficient incremental inference for streaming data
    - Multiple cache optimization strategies
    - Low-latency predictions for real-time trading
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 4096,
        n_outputs: int = 1,
        output_type: str = 'regression',
        dropout: float = 0.1,
        cache_config: Optional[KVCacheConfig] = None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.head_dim = d_model // n_heads
        self.n_outputs = n_outputs
        self.output_type = output_type
        self.cache_config = cache_config or KVCacheConfig()

        # For external access
        self.num_layers = n_layers
        self.num_heads = n_heads

        # Input embedding
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # Transformer layers
        self.layers = nn.ModuleList([
            KVCacheTransformerBlock(d_model, n_heads, d_ff, dropout, cache_config)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Output head
        if output_type == 'regression':
            self.head = nn.Linear(d_model, n_outputs)
        elif output_type == 'direction':
            self.head = nn.Linear(d_model, n_outputs)
        elif output_type == 'allocation':
            self.head = nn.Sequential(
                nn.Linear(d_model, n_outputs),
                nn.Tanh()
            )
        else:
            self.head = nn.Linear(d_model, n_outputs)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        past_kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass with KV-cache support.

        Args:
            x: Input features [batch, seq_len, input_dim]
            past_kv_cache: List of (K, V) tuples from previous steps
            use_cache: Whether to return updated cache
            attention_mask: Optional attention mask

        Returns:
            predictions: Model output
            present_kv_cache: Updated cache for all layers
        """
        batch_size, seq_len, _ = x.shape

        # Determine position offset from cache
        if past_kv_cache is not None and past_kv_cache[0] is not None:
            past_length = past_kv_cache[0][0].shape[2]
        else:
            past_length = 0

        # Project input
        x = self.input_proj(x)

        # Add positional encoding (accounting for past tokens)
        x = x + self.pos_encoding[:, past_length:past_length + seq_len, :]

        # Initialize cache list
        if past_kv_cache is None:
            past_kv_cache = [None] * self.n_layers

        present_kv_cache = []

        # Forward through transformer layers
        for i, layer in enumerate(self.layers):
            x, present_kv = layer(
                x,
                past_kv=past_kv_cache[i],
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            if use_cache:
                present_kv_cache.append(present_kv)

        x = self.norm(x)

        # Use last token for prediction
        x = x[:, -1, :]

        # Output head
        output = self.head(x)

        if use_cache:
            return output, present_kv_cache
        return output, None

    def generate(
        self,
        initial_context: torch.Tensor,
        num_steps: int,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate predictions autoregressively with KV-cache.

        Efficient generation for multi-step forecasting.
        """
        self.eval()

        with torch.no_grad():
            # First step: process full context
            predictions = []
            output, kv_cache = self(initial_context, use_cache=True)
            predictions.append(output)

            # Subsequent steps: incremental with cache
            current_input = output.unsqueeze(1)  # Use prediction as next input

            for _ in range(num_steps - 1):
                output, kv_cache = self(
                    current_input,
                    past_kv_cache=kv_cache,
                    use_cache=True
                )
                predictions.append(output)
                current_input = output.unsqueeze(1)

        return torch.stack(predictions, dim=1)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss based on output type."""
        if self.output_type == 'regression':
            return F.mse_loss(predictions, targets)
        elif self.output_type == 'direction':
            return F.binary_cross_entropy_with_logits(predictions, (targets > 0).float())
        elif self.output_type == 'allocation':
            return -torch.mean(predictions * targets)
        else:
            raise ValueError(f"Unknown output type: {self.output_type}")


def benchmark_kv_cache(
    model: KVCacheTrader,
    context_length: int,
    num_iterations: int = 100,
    device: str = 'cuda'
) -> dict:
    """
    Benchmark KV-cache performance.

    Compares inference with and without caching.
    """
    import time

    # Use CPU if CUDA not available
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    model = model.to(device)
    model.eval()

    # Generate dummy input
    x = torch.randn(1, context_length, model.input_dim, device=device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(x, use_cache=False)

    # Synchronize if using CUDA
    if device == 'cuda':
        torch.cuda.synchronize()

    # Without cache (full recomputation each step)
    start = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(x, use_cache=False)
    if device == 'cuda':
        torch.cuda.synchronize()
    no_cache_time = (time.time() - start) / num_iterations

    # With cache (incremental)
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            # Initial context
            _, kv_cache = model(x[:, :-1], use_cache=True)
            # Incremental step
            _ = model(x[:, -1:], past_kv_cache=kv_cache, use_cache=True)
    if device == 'cuda':
        torch.cuda.synchronize()
    cache_time = (time.time() - start) / num_iterations

    return {
        'context_length': context_length,
        'no_cache_ms': no_cache_time * 1000,
        'with_cache_ms': cache_time * 1000,
        'speedup': no_cache_time / cache_time if cache_time > 0 else float('inf')
    }


if __name__ == '__main__':
    # Quick test
    model = KVCacheTrader(input_dim=5, d_model=128, n_heads=4, n_layers=3)
    x = torch.randn(2, 10, 5)

    # Test forward pass
    output, cache = model(x, use_cache=True)
    print(f"Output shape: {output.shape}")
    print(f"Cache layers: {len(cache)}")

    # Test incremental inference
    new_x = torch.randn(2, 1, 5)
    output2, cache2 = model(new_x, past_kv_cache=cache, use_cache=True)
    print(f"Incremental output shape: {output2.shape}")

    # Benchmark
    results = benchmark_kv_cache(model, context_length=256, num_iterations=50, device='cpu')
    print(f"Benchmark results: {results}")
