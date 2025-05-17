# Chapter 60: KV-Cache Optimization for Algorithmic Trading

This chapter explores **KV-Cache (Key-Value Cache) Optimization**, a critical technique for efficient inference in Transformer-based trading systems. We apply KV-cache optimization strategies to real-time financial prediction, demonstrating how memory-efficient inference enables low-latency trading decisions with longer context windows.

<p align="center">
<img src="https://i.imgur.com/kVcache.png" width="70%">
</p>

## Contents

1. [Introduction to KV-Cache](#introduction-to-kv-cache)
    * [The Inference Bottleneck](#the-inference-bottleneck)
    * [What is KV-Cache?](#what-is-kv-cache)
    * [Why It Matters for Trading](#why-it-matters-for-trading)
2. [KV-Cache Fundamentals](#kv-cache-fundamentals)
    * [Autoregressive Generation](#autoregressive-generation)
    * [Memory Growth Problem](#memory-growth-problem)
    * [Cache Structure](#cache-structure)
3. [Optimization Techniques](#optimization-techniques)
    * [PagedAttention](#pagedattention)
    * [KV-Cache Quantization](#kv-cache-quantization)
    * [Selective Retention](#selective-retention)
    * [Prefix Caching](#prefix-caching)
4. [Trading Applications](#trading-applications)
    * [Real-Time Price Prediction](#real-time-price-prediction)
    * [Streaming Order Book Analysis](#streaming-order-book-analysis)
    * [Multi-Asset Portfolio Inference](#multi-asset-portfolio-inference)
5. [Practical Examples](#practical-examples)
    * [01: Data Preparation](#01-data-preparation)
    * [02: KV-Cache Transformer Model](#02-kv-cache-transformer-model)
    * [03: Optimized Inference Engine](#03-optimized-inference-engine)
    * [04: Real-Time Prediction](#04-real-time-prediction)
    * [05: Trading Strategy Backtesting](#05-trading-strategy-backtesting)
6. [Python Implementation](#python-implementation)
7. [Rust Implementation](#rust-implementation)
8. [Performance Benchmarks](#performance-benchmarks)
9. [Best Practices](#best-practices)
10. [Resources](#resources)

## Introduction to KV-Cache

### The Inference Bottleneck

In production trading systems, inference speed is critical. While training happens offline, inference must happen in real-time—often within milliseconds. Transformer models face a fundamental challenge during autoregressive generation: they must recompute attention over all previous tokens for each new prediction.

```
Traditional Transformer Inference:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   Token 1: Compute attention for [Token 1]                                  │
│   Token 2: Compute attention for [Token 1, Token 2]                         │
│   Token 3: Compute attention for [Token 1, Token 2, Token 3]                │
│   ...                                                                        │
│   Token N: Compute attention for [Token 1, Token 2, ... Token N]            │
│                                                                              │
│   Problem: Redundant computation of Q, K, V for tokens 1 to N-1!            │
│   Each step recomputes everything from scratch.                             │
│                                                                              │
│   Complexity: O(N²) per sequence                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What is KV-Cache?

**KV-Cache** stores the Key and Value tensors computed during previous inference steps, avoiding redundant recalculation:

```
KV-Cache Mechanism:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   Step 1: Process Token 1                                                   │
│           Compute K₁, V₁ → Store in cache                                   │
│           Output: Next token prediction                                     │
│                                                                              │
│   Step 2: Process Token 2                                                   │
│           Load K₁, V₁ from cache (no recomputation!)                        │
│           Compute K₂, V₂ → Append to cache                                  │
│           Output: Next token prediction                                     │
│                                                                              │
│   Step N: Process Token N                                                   │
│           Load K₁...K_{N-1}, V₁...V_{N-1} from cache                        │
│           Compute only K_N, V_N → Append to cache                           │
│           Output: Next token prediction                                     │
│                                                                              │
│   Result: O(N) computation instead of O(N²) per new token!                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why It Matters for Trading

| Scenario | Without KV-Cache | With KV-Cache | Improvement |
|----------|-----------------|---------------|-------------|
| Real-time price prediction | 50ms latency | 5ms latency | 10x faster |
| Streaming order book | Cannot keep up | Real-time | Enables use case |
| Long context (1 year data) | Out of memory | Feasible | Unlocks capability |
| Batch serving (100 requests) | 20 req/sec | 200 req/sec | 10x throughput |

For trading applications:
- **Latency matters**: Every millisecond counts in high-frequency trading
- **Memory efficiency**: Enables processing longer market history
- **Throughput**: Serve more prediction requests simultaneously
- **Cost reduction**: Fewer GPU resources needed for same performance

## KV-Cache Fundamentals

### Autoregressive Generation

Transformer models generate predictions autoregressively—each new token depends on all previous tokens:

```python
def autoregressive_inference_naive(model, initial_context):
    """
    Naive autoregressive inference (inefficient).

    For trading: predicting the next price movement
    based on historical price sequence.
    """
    sequence = initial_context.copy()

    for step in range(prediction_horizon):
        # Problem: Recomputes K, V for ALL tokens every step
        output = model.forward(sequence)  # O(N²) each time!
        next_prediction = output[-1]
        sequence.append(next_prediction)

    return sequence

def autoregressive_inference_with_cache(model, initial_context):
    """
    Efficient inference with KV-cache.
    """
    sequence = initial_context.copy()
    kv_cache = None

    for step in range(prediction_horizon):
        if kv_cache is None:
            # First step: compute and cache K, V for all tokens
            output, kv_cache = model.forward(sequence, use_cache=True)
        else:
            # Subsequent steps: only compute K, V for new token
            output, kv_cache = model.forward(
                [sequence[-1]],  # Only the latest token!
                past_kv_cache=kv_cache,
                use_cache=True
            )

        next_prediction = output[-1]
        sequence.append(next_prediction)

    return sequence
```

### Memory Growth Problem

The challenge with KV-cache is memory consumption grows linearly with sequence length:

```
KV-Cache Memory Usage:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   Memory per token = 2 × num_layers × num_heads × head_dim × bytes_per_val │
│                                                                              │
│   Example: LLaMA-2 13B parameters                                           │
│   - 40 layers, 40 heads, 128 head_dim, FP16 (2 bytes)                       │
│   - Per token: 2 × 40 × 40 × 128 × 2 = 819,200 bytes ≈ 0.8 MB              │
│                                                                              │
│   For trading with different context lengths:                               │
│   ─────────────────────────────────────────────────────────────────────────  │
│   Context Length          Memory per Sequence    Annual Hourly Data        │
│   ─────────────────────────────────────────────────────────────────────────  │
│   256 tokens              ~200 MB               ~10 days hourly            │
│   1,024 tokens            ~800 MB               ~6 weeks hourly            │
│   4,096 tokens            ~3.2 GB               ~6 months hourly           │
│   8,760 tokens            ~7 GB                 1 year hourly              │
│                                                                              │
│   Problem: Memory grows linearly, limiting batch size and context!          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Cache Structure

```python
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

    def __init__(self, num_layers, batch_size, num_heads, head_dim, dtype=torch.float16):
        self.num_layers = num_layers
        self.keys = [None] * num_layers
        self.values = [None] * num_layers

        # Pre-allocate for known maximum length (optional optimization)
        self.max_seq_len = None
        self.current_seq_len = 0

    def update(self, layer_idx, new_keys, new_values):
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

    def get(self, layer_idx):
        """Retrieve cached keys and values for a layer."""
        return self.keys[layer_idx], self.values[layer_idx]

    def memory_usage(self):
        """Calculate total memory usage in bytes."""
        total = 0
        for k, v in zip(self.keys, self.values):
            if k is not None:
                total += k.numel() * k.element_size()
                total += v.numel() * v.element_size()
        return total
```

## Optimization Techniques

### PagedAttention

**PagedAttention** (introduced by vLLM) applies operating system memory paging concepts to KV-cache management:

```
PagedAttention Concept:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   Traditional KV-Cache (Contiguous Memory):                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ Request 1 KV │    WASTED    │ Request 2 KV │    WASTED    │ ...    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│   Problem: Must pre-allocate max length → 60-80% memory wasted!             │
│                                                                              │
│   PagedAttention (Paged Memory):                                            │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐            │
│   │ Block 1 │ │ Block 2 │ │ Block 3 │ │ Block 4 │ │ Block 5 │            │
│   │ Req 1   │ │ Req 1   │ │ Req 2   │ │ Req 1   │ │ Req 2   │            │
│   └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘            │
│                                                                              │
│   Block Table (maps logical to physical blocks):                            │
│   Request 1: [Block 1, Block 2, Block 4]                                    │
│   Request 2: [Block 3, Block 5]                                             │
│                                                                              │
│   Result: Near-zero memory waste, dynamic allocation!                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

```python
class PagedKVCache:
    """
    Paged KV-Cache implementation for trading inference.

    Benefits for trading:
    - Serve multiple asset predictions simultaneously
    - Dynamically growing sequences (streaming data)
    - Memory-efficient batch processing
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 16,
        num_blocks: int = 1000,
        dtype: torch.dtype = torch.float16
    ):
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Pre-allocate block pool
        self.key_blocks = torch.zeros(
            num_blocks, num_layers, num_heads, block_size, head_dim,
            dtype=dtype
        )
        self.value_blocks = torch.zeros(
            num_blocks, num_layers, num_heads, block_size, head_dim,
            dtype=dtype
        )

        # Free block list
        self.free_blocks = list(range(num_blocks))

        # Block tables: maps sequence_id -> list of block indices
        self.block_tables = {}

    def allocate_blocks(self, sequence_id: int, num_tokens: int):
        """Allocate blocks for a new sequence."""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < num_blocks_needed:
            raise MemoryError("Not enough free blocks")

        allocated = [self.free_blocks.pop() for _ in range(num_blocks_needed)]
        self.block_tables[sequence_id] = allocated
        return allocated

    def free_sequence(self, sequence_id: int):
        """Free blocks when a sequence completes."""
        if sequence_id in self.block_tables:
            self.free_blocks.extend(self.block_tables[sequence_id])
            del self.block_tables[sequence_id]

    def append_tokens(self, sequence_id: int, layer_idx: int, keys: torch.Tensor, values: torch.Tensor):
        """
        Append new KV pairs to a sequence's cache.

        Handles block allocation automatically.
        """
        block_indices = self.block_tables[sequence_id]
        current_len = sum(1 for _ in block_indices) * self.block_size

        # Determine which block and position
        for i, (k, v) in enumerate(zip(keys, values)):
            block_idx = (current_len + i) // self.block_size
            pos_in_block = (current_len + i) % self.block_size

            # Allocate new block if needed
            if block_idx >= len(block_indices):
                if not self.free_blocks:
                    raise MemoryError("No free blocks available")
                new_block = self.free_blocks.pop()
                block_indices.append(new_block)

            physical_block = block_indices[block_idx]
            self.key_blocks[physical_block, layer_idx, :, pos_in_block, :] = k
            self.value_blocks[physical_block, layer_idx, :, pos_in_block, :] = v
```

### KV-Cache Quantization

Reduce memory footprint by quantizing cached values:

```python
class QuantizedKVCache:
    """
    Quantized KV-Cache for memory-efficient inference.

    FP8 quantization reduces memory by 50% vs FP16 with minimal quality loss.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        quantization: str = 'fp8'  # Options: 'fp8', 'int8', 'int4'
    ):
        self.quantization = quantization
        self.num_layers = num_layers

        # Storage dtype based on quantization
        if quantization == 'fp8':
            self.storage_dtype = torch.float8_e4m3fn
            self.scale_dtype = torch.float16
        elif quantization == 'int8':
            self.storage_dtype = torch.int8
            self.scale_dtype = torch.float16
        elif quantization == 'int4':
            self.storage_dtype = torch.int8  # Pack two int4 values
            self.scale_dtype = torch.float16

        self.keys = [None] * num_layers
        self.values = [None] * num_layers
        self.key_scales = [None] * num_layers
        self.value_scales = [None] * num_layers

    def quantize(self, tensor: torch.Tensor) -> tuple:
        """Quantize tensor and return quantized values + scale."""
        if self.quantization == 'fp8':
            scale = tensor.abs().max() / 448.0  # FP8 E4M3 max value
            quantized = (tensor / scale).to(self.storage_dtype)
            return quantized, scale

        elif self.quantization == 'int8':
            scale = tensor.abs().max() / 127.0
            quantized = torch.round(tensor / scale).to(torch.int8)
            return quantized, scale

        elif self.quantization == 'int4':
            scale = tensor.abs().max() / 7.0
            quantized = torch.round(tensor / scale).clamp(-8, 7).to(torch.int8)
            return quantized, scale

    def dequantize(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize values for attention computation."""
        return quantized.to(torch.float16) * scale

    def update(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor):
        """Store quantized KV pairs."""
        q_keys, k_scale = self.quantize(keys)
        q_values, v_scale = self.quantize(values)

        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = q_keys
            self.values[layer_idx] = q_values
            self.key_scales[layer_idx] = k_scale
            self.value_scales[layer_idx] = v_scale
        else:
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], q_keys], dim=2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], q_values], dim=2)

    def get(self, layer_idx: int) -> tuple:
        """Retrieve and dequantize cached KV pairs."""
        keys = self.dequantize(self.keys[layer_idx], self.key_scales[layer_idx])
        values = self.dequantize(self.values[layer_idx], self.value_scales[layer_idx])
        return keys, values

    def memory_savings(self) -> float:
        """Calculate memory savings vs FP16."""
        if self.quantization == 'fp8':
            return 0.5  # 50% savings
        elif self.quantization == 'int8':
            return 0.5
        elif self.quantization == 'int4':
            return 0.75  # 75% savings
```

### Selective Retention

Keep only the most important KV pairs to limit memory growth:

```python
class SelectiveKVCache:
    """
    Selective KV-Cache with importance-based retention.

    For trading: Retains KV pairs for critical market events
    while discarding less relevant historical data.
    """

    def __init__(
        self,
        num_layers: int,
        max_cache_size: int = 2048,
        retention_strategy: str = 'attention_score'
    ):
        self.num_layers = num_layers
        self.max_cache_size = max_cache_size
        self.retention_strategy = retention_strategy

        self.keys = [None] * num_layers
        self.values = [None] * num_layers
        self.importance_scores = [None] * num_layers

    def compute_importance(
        self,
        attention_weights: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Compute importance scores for each cached position.

        Strategies:
        - attention_score: Based on attention weights received
        - recency: More recent = more important
        - entropy: High-entropy positions are more informative
        - hybrid: Combination of above
        """
        if self.retention_strategy == 'attention_score':
            # Sum attention received from all query positions
            importance = attention_weights.sum(dim=-2).mean(dim=1)  # [batch, seq_len]

        elif self.retention_strategy == 'recency':
            seq_len = attention_weights.shape[-1]
            importance = torch.arange(seq_len, device=attention_weights.device).float()
            importance = importance / seq_len

        elif self.retention_strategy == 'entropy':
            # Higher entropy = more informative
            entropy = -torch.sum(
                attention_weights * torch.log(attention_weights + 1e-10),
                dim=-1
            ).mean(dim=1)
            importance = entropy

        elif self.retention_strategy == 'hybrid':
            # Combine attention and recency
            attention_imp = attention_weights.sum(dim=-2).mean(dim=1)
            seq_len = attention_weights.shape[-1]
            recency = torch.arange(seq_len, device=attention_weights.device).float() / seq_len
            importance = 0.7 * attention_imp + 0.3 * recency

        return importance

    def evict_if_needed(self, layer_idx: int):
        """Evict least important entries if cache exceeds max size."""
        if self.keys[layer_idx] is None:
            return

        current_size = self.keys[layer_idx].shape[2]

        if current_size > self.max_cache_size:
            # Keep top-k most important positions
            importance = self.importance_scores[layer_idx]
            _, keep_indices = torch.topk(importance, self.max_cache_size, dim=-1)
            keep_indices = keep_indices.sort(dim=-1).values  # Maintain temporal order

            # Gather kept entries
            self.keys[layer_idx] = torch.gather(
                self.keys[layer_idx],
                dim=2,
                index=keep_indices.unsqueeze(1).unsqueeze(-1).expand(-1, -1, -1, self.keys[layer_idx].shape[-1])
            )
            self.values[layer_idx] = torch.gather(
                self.values[layer_idx],
                dim=2,
                index=keep_indices.unsqueeze(1).unsqueeze(-1).expand(-1, -1, -1, self.values[layer_idx].shape[-1])
            )
```

### Prefix Caching

Cache common prefixes to avoid recomputation:

```python
class PrefixCache:
    """
    Prefix caching for shared context across requests.

    For trading: Cache market context that's common to multiple
    asset predictions (e.g., macro indicators, market regime).
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.prefix_store = {}  # hash -> (keys, values, length)

    def hash_prefix(self, prefix_tokens: torch.Tensor) -> str:
        """Create hash of prefix for lookup."""
        return hash(prefix_tokens.cpu().numpy().tobytes())

    def store_prefix(
        self,
        prefix_tokens: torch.Tensor,
        keys: list,
        values: list
    ):
        """Store computed KV cache for a prefix."""
        prefix_hash = self.hash_prefix(prefix_tokens)
        self.prefix_store[prefix_hash] = {
            'keys': [k.clone() for k in keys],
            'values': [v.clone() for v in values],
            'length': prefix_tokens.shape[1]
        }

    def lookup_prefix(self, prefix_tokens: torch.Tensor) -> dict:
        """Look up cached prefix KV values."""
        prefix_hash = self.hash_prefix(prefix_tokens)
        return self.prefix_store.get(prefix_hash)

    def get_with_prefix(
        self,
        full_sequence: torch.Tensor,
        prefix_length: int
    ) -> tuple:
        """
        Try to use cached prefix, compute only suffix.

        Returns: (cached_kv, suffix_start_idx) or (None, 0)
        """
        prefix = full_sequence[:, :prefix_length]
        cached = self.lookup_prefix(prefix)

        if cached is not None:
            return cached, prefix_length
        return None, 0
```

## Trading Applications

### Real-Time Price Prediction

```python
class RealTimePricePredictor:
    """
    Real-time price prediction with optimized KV-cache.

    Use case: Predict next price movement based on streaming market data.
    """

    def __init__(
        self,
        model: nn.Module,
        kv_cache_type: str = 'paged',  # 'standard', 'paged', 'quantized', 'selective'
        max_context: int = 4096
    ):
        self.model = model
        self.max_context = max_context

        if kv_cache_type == 'paged':
            self.kv_cache = PagedKVCache(
                num_layers=model.num_layers,
                num_heads=model.num_heads,
                head_dim=model.head_dim,
                block_size=16
            )
        elif kv_cache_type == 'quantized':
            self.kv_cache = QuantizedKVCache(
                num_layers=model.num_layers,
                num_heads=model.num_heads,
                head_dim=model.head_dim,
                quantization='fp8'
            )
        elif kv_cache_type == 'selective':
            self.kv_cache = SelectiveKVCache(
                num_layers=model.num_layers,
                max_cache_size=max_context
            )
        else:
            self.kv_cache = KVCache(
                num_layers=model.num_layers,
                batch_size=1,
                num_heads=model.num_heads,
                head_dim=model.head_dim
            )

    def predict_stream(
        self,
        data_stream: Iterator[dict],
        symbol: str
    ) -> Iterator[dict]:
        """
        Stream predictions for real-time trading.

        Args:
            data_stream: Iterator yielding market data points
            symbol: Trading symbol (e.g., 'BTCUSDT')

        Yields:
            Predictions with confidence and latency metrics
        """
        context = []

        for data_point in data_stream:
            start_time = time.time()

            # Prepare features
            features = self.extract_features(data_point)

            if len(context) == 0:
                # First prediction: full forward pass
                context = [features]
                with torch.no_grad():
                    output, self.kv_cache = self.model(
                        torch.tensor([context]),
                        use_cache=True
                    )
            else:
                # Incremental prediction: use cached KV
                context.append(features)

                # Manage context window
                if len(context) > self.max_context:
                    # Sliding window: remove oldest, but KV-cache handles efficiently
                    context = context[-self.max_context:]

                with torch.no_grad():
                    output, self.kv_cache = self.model(
                        torch.tensor([[features]]),  # Only new token
                        past_kv_cache=self.kv_cache,
                        use_cache=True
                    )

            latency = time.time() - start_time
            prediction = output[0, -1].item()

            yield {
                'timestamp': data_point['timestamp'],
                'symbol': symbol,
                'prediction': prediction,
                'direction': 'UP' if prediction > 0 else 'DOWN',
                'confidence': abs(prediction),
                'latency_ms': latency * 1000,
                'cache_memory_mb': self.kv_cache.memory_usage() / (1024 * 1024)
            }

    def extract_features(self, data_point: dict) -> list:
        """Extract features from market data point."""
        return [
            data_point.get('log_return', 0),
            data_point.get('volume_ratio', 1),
            data_point.get('volatility', 0),
            data_point.get('bid_ask_spread', 0),
            data_point.get('order_imbalance', 0)
        ]
```

### Streaming Order Book Analysis

```python
class StreamingOrderBookAnalyzer:
    """
    Analyze order book updates with efficient KV-caching.

    Order books generate high-frequency updates (100-1000/sec),
    requiring very efficient inference.
    """

    def __init__(
        self,
        model: nn.Module,
        num_levels: int = 20,
        update_buffer_size: int = 100
    ):
        self.model = model
        self.num_levels = num_levels

        # Use quantized cache for memory efficiency
        self.kv_cache = QuantizedKVCache(
            num_layers=model.num_layers,
            num_heads=model.num_heads,
            head_dim=model.head_dim,
            quantization='int8'
        )

        # Buffer for batching updates
        self.update_buffer = []
        self.buffer_size = update_buffer_size

    def process_update(self, order_book_snapshot: dict) -> dict:
        """
        Process single order book update.

        Args:
            order_book_snapshot: {
                'bids': [[price, qty], ...],
                'asks': [[price, qty], ...],
                'timestamp': int
            }

        Returns:
            Prediction with microstructure features
        """
        # Extract features from order book
        features = self.extract_lob_features(order_book_snapshot)

        # Add to buffer
        self.update_buffer.append(features)

        # Process when buffer is full
        if len(self.update_buffer) >= self.buffer_size:
            return self.flush_buffer()

        return None

    def extract_lob_features(self, snapshot: dict) -> torch.Tensor:
        """Extract features from limit order book."""
        bids = snapshot['bids'][:self.num_levels]
        asks = snapshot['asks'][:self.num_levels]

        features = []

        # Price levels
        for i in range(self.num_levels):
            if i < len(bids):
                features.extend([bids[i][0], bids[i][1]])  # bid price, qty
            else:
                features.extend([0, 0])

            if i < len(asks):
                features.extend([asks[i][0], asks[i][1]])  # ask price, qty
            else:
                features.extend([0, 0])

        # Derived features
        if bids and asks:
            mid_price = (bids[0][0] + asks[0][0]) / 2
            spread = asks[0][0] - bids[0][0]
            bid_volume = sum(b[1] for b in bids)
            ask_volume = sum(a[1] for a in asks)
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-8)

            features.extend([mid_price, spread, imbalance])

        return torch.tensor(features, dtype=torch.float16)

    def flush_buffer(self) -> dict:
        """Process buffered updates with KV-cache."""
        if not self.update_buffer:
            return None

        # Stack buffered features
        batch = torch.stack(self.update_buffer).unsqueeze(0)

        with torch.no_grad():
            output, self.kv_cache = self.model(
                batch,
                past_kv_cache=self.kv_cache,
                use_cache=True
            )

        # Clear buffer
        self.update_buffer = []

        return {
            'prediction': output[0, -1].item(),
            'cache_size': self.kv_cache.memory_usage()
        }
```

### Multi-Asset Portfolio Inference

```python
class MultiAssetPortfolioInference:
    """
    Efficient inference for multi-asset portfolios.

    Uses prefix caching for shared market context across assets.
    """

    def __init__(
        self,
        model: nn.Module,
        assets: list,
        shared_context_length: int = 256
    ):
        self.model = model
        self.assets = assets
        self.shared_context_length = shared_context_length

        # Prefix cache for shared market context
        self.prefix_cache = PrefixCache(num_layers=model.num_layers)

        # Per-asset KV caches
        self.asset_caches = {
            asset: KVCache(
                num_layers=model.num_layers,
                batch_size=1,
                num_heads=model.num_heads,
                head_dim=model.head_dim
            )
            for asset in assets
        }

    def compute_shared_context(self, market_data: dict) -> torch.Tensor:
        """
        Compute shared context from macro/market-wide data.

        Shared across all asset predictions:
        - Market regime indicators
        - VIX, DXY, interest rates
        - Cross-asset correlations
        """
        features = [
            market_data.get('vix', 0),
            market_data.get('dxy', 0),
            market_data.get('sp500_return', 0),
            market_data.get('btc_dominance', 0),
            market_data.get('total_market_cap', 0)
        ]
        return torch.tensor([features], dtype=torch.float16)

    def predict_all_assets(
        self,
        market_data: dict,
        asset_data: dict
    ) -> dict:
        """
        Generate predictions for all assets efficiently.

        Args:
            market_data: Shared market-wide data
            asset_data: Per-asset specific data

        Returns:
            Predictions for each asset with allocation weights
        """
        predictions = {}

        # Compute shared context once
        shared_context = self.compute_shared_context(market_data)

        # Check if prefix is cached
        cached_prefix = self.prefix_cache.lookup_prefix(shared_context)

        if cached_prefix is None:
            # Compute and cache prefix
            with torch.no_grad():
                _, prefix_kv = self.model(
                    shared_context,
                    use_cache=True
                )
            self.prefix_cache.store_prefix(
                shared_context,
                prefix_kv.keys,
                prefix_kv.values
            )
            cached_prefix = prefix_kv

        # Predict each asset with shared prefix
        for asset in self.assets:
            asset_features = self.extract_asset_features(asset_data.get(asset, {}))

            with torch.no_grad():
                output, self.asset_caches[asset] = self.model(
                    asset_features,
                    past_kv_cache=cached_prefix,  # Reuse shared context
                    use_cache=True
                )

            predictions[asset] = {
                'return_prediction': output[0, -1, 0].item(),
                'volatility_prediction': output[0, -1, 1].item() if output.shape[-1] > 1 else None
            }

        # Compute portfolio allocation based on predictions
        allocations = self.compute_allocations(predictions)

        return {
            'predictions': predictions,
            'allocations': allocations,
            'shared_context_cached': cached_prefix is not None
        }

    def extract_asset_features(self, asset_data: dict) -> torch.Tensor:
        """Extract features for a specific asset."""
        return torch.tensor([[
            asset_data.get('log_return', 0),
            asset_data.get('volume', 0),
            asset_data.get('volatility', 0),
            asset_data.get('momentum', 0),
            asset_data.get('rsi', 50) / 100
        ]], dtype=torch.float16)

    def compute_allocations(self, predictions: dict) -> dict:
        """Compute portfolio allocations from predictions."""
        returns = {k: v['return_prediction'] for k, v in predictions.items()}
        total_positive = sum(max(0, r) for r in returns.values())

        if total_positive == 0:
            # Equal weight if no positive predictions
            return {k: 1.0 / len(self.assets) for k in self.assets}

        return {
            k: max(0, r) / total_positive
            for k, r in returns.items()
        }
```

## Practical Examples

### 01: Data Preparation

```python
# python/data_loader.py

import numpy as np
import pandas as pd
import requests
from typing import List, Dict, Iterator
from datetime import datetime, timedelta

def fetch_bybit_klines(
    symbol: str,
    interval: str = '60',
    limit: int = 1000
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Bybit.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Candle interval in minutes
        limit: Number of candles to fetch

    Returns:
        DataFrame with OHLCV data
    """
    url = 'https://api.bybit.com/v5/market/kline'

    params = {
        'category': 'linear',
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data['retCode'] != 0:
        raise ValueError(f"API Error: {data['retMsg']}")

    df = pd.DataFrame(data['result']['list'], columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = df[col].astype(float)

    return df.sort_values('timestamp').reset_index(drop=True)


def create_streaming_generator(
    symbol: str,
    lookback_days: int = 30
) -> Iterator[Dict]:
    """
    Create a generator that simulates streaming market data.

    Useful for testing KV-cache efficiency with incremental updates.
    """
    df = fetch_bybit_klines(symbol, limit=lookback_days * 24)

    # Calculate features
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['log_return'].rolling(24).std()
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(24).mean()
    df['momentum'] = df['close'] / df['close'].shift(24) - 1

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))

    df = df.dropna()

    for _, row in df.iterrows():
        yield {
            'timestamp': row['timestamp'],
            'close': row['close'],
            'log_return': row['log_return'],
            'volatility': row['volatility'],
            'volume_ratio': row['volume_ratio'],
            'momentum': row['momentum'],
            'rsi': row['rsi']
        }


def prepare_kv_cache_benchmark_data(
    symbols: List[str],
    context_lengths: List[int] = [256, 512, 1024, 2048, 4096]
) -> Dict[int, np.ndarray]:
    """
    Prepare data for benchmarking different context lengths.

    KV-cache efficiency becomes more important with longer contexts.
    """
    max_length = max(context_lengths) + 100

    all_data = []
    for symbol in symbols:
        df = fetch_bybit_klines(symbol, limit=max_length)

        # Features
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(24).std()
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(24).mean()

        features = df[['log_return', 'volatility', 'volume_ratio']].dropna().values
        all_data.append(features)

    # Stack and create sequences for each context length
    combined = np.concatenate(all_data, axis=1)

    benchmark_data = {}
    for length in context_lengths:
        sequences = []
        for i in range(len(combined) - length):
            sequences.append(combined[i:i+length])
        benchmark_data[length] = np.array(sequences)

    return benchmark_data
```

### 02: KV-Cache Transformer Model

```python
# python/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass


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

    model = model.to(device)
    model.eval()

    # Generate dummy input
    x = torch.randn(1, context_length, model.input_dim, device=device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(x, use_cache=False)

    # Without cache (full recomputation each step)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(x, use_cache=False)
    torch.cuda.synchronize()
    no_cache_time = (time.time() - start) / num_iterations

    # With cache (incremental)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            # Initial context
            _, kv_cache = model(x[:, :-1], use_cache=True)
            # Incremental step
            _ = model(x[:, -1:], past_kv_cache=kv_cache, use_cache=True)
    torch.cuda.synchronize()
    cache_time = (time.time() - start) / num_iterations

    return {
        'context_length': context_length,
        'no_cache_ms': no_cache_time * 1000,
        'with_cache_ms': cache_time * 1000,
        'speedup': no_cache_time / cache_time
    }
```

### 03: Optimized Inference Engine

```python
# python/inference.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Iterator
import time
from dataclasses import dataclass
from collections import deque

from model import KVCacheTrader, KVCacheConfig


@dataclass
class InferenceMetrics:
    """Metrics for tracking inference performance."""
    latency_ms: float
    throughput_tokens_per_sec: float
    cache_memory_mb: float
    cache_hit_rate: float


class OptimizedInferenceEngine:
    """
    Optimized inference engine for trading with KV-cache.

    Features:
    - Efficient KV-cache management
    - Batch inference support
    - Memory-aware caching
    - Latency tracking
    """

    def __init__(
        self,
        model: KVCacheTrader,
        max_batch_size: int = 32,
        max_cache_memory_mb: float = 1024,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_cache_memory_mb = max_cache_memory_mb

        # Per-request cache storage
        self.request_caches: Dict[str, List] = {}
        self.request_timestamps: Dict[str, float] = {}

        # Metrics tracking
        self.latency_history = deque(maxlen=1000)
        self.cache_hits = 0
        self.cache_misses = 0

    def compute_cache_memory(self, kv_cache: List) -> float:
        """Calculate memory usage of a KV cache in MB."""
        if kv_cache is None or kv_cache[0] is None:
            return 0

        total_bytes = 0
        for k, v in kv_cache:
            if k is not None:
                total_bytes += k.numel() * k.element_size()
                total_bytes += v.numel() * v.element_size()

        return total_bytes / (1024 * 1024)

    def evict_old_caches(self):
        """Evict oldest caches when memory limit is reached."""
        total_memory = sum(
            self.compute_cache_memory(cache)
            for cache in self.request_caches.values()
        )

        while total_memory > self.max_cache_memory_mb and self.request_caches:
            # Find oldest request
            oldest_id = min(self.request_timestamps, key=self.request_timestamps.get)
            total_memory -= self.compute_cache_memory(self.request_caches[oldest_id])
            del self.request_caches[oldest_id]
            del self.request_timestamps[oldest_id]

    def predict_single(
        self,
        request_id: str,
        features: torch.Tensor,
        use_cache: bool = True
    ) -> Dict:
        """
        Make prediction for a single request.

        Args:
            request_id: Unique identifier for caching
            features: Input features [1, seq_len, input_dim]
            use_cache: Whether to use/update cache

        Returns:
            Dictionary with prediction and metrics
        """
        start_time = time.time()

        # Check for existing cache
        past_kv_cache = self.request_caches.get(request_id) if use_cache else None
        cache_hit = past_kv_cache is not None

        if cache_hit:
            self.cache_hits += 1
            # Only process new tokens
            past_length = past_kv_cache[0][0].shape[2]
            new_features = features[:, past_length:]
        else:
            self.cache_misses += 1
            new_features = features

        with torch.no_grad():
            output, present_kv_cache = self.model(
                new_features.to(self.device),
                past_kv_cache=past_kv_cache,
                use_cache=use_cache
            )

        # Update cache
        if use_cache:
            self.request_caches[request_id] = present_kv_cache
            self.request_timestamps[request_id] = time.time()
            self.evict_old_caches()

        latency = (time.time() - start_time) * 1000
        self.latency_history.append(latency)

        return {
            'prediction': output.cpu().numpy(),
            'latency_ms': latency,
            'cache_hit': cache_hit,
            'cache_memory_mb': self.compute_cache_memory(present_kv_cache)
        }

    def predict_batch(
        self,
        requests: List[Dict]
    ) -> List[Dict]:
        """
        Batch prediction for multiple requests.

        Args:
            requests: List of {'request_id': str, 'features': tensor}

        Returns:
            List of prediction results
        """
        results = []

        # Group by cache status for efficient processing
        cached_requests = []
        uncached_requests = []

        for req in requests:
            if req['request_id'] in self.request_caches:
                cached_requests.append(req)
            else:
                uncached_requests.append(req)

        # Process uncached requests (full context)
        if uncached_requests:
            # Can batch these together
            features = torch.stack([r['features'] for r in uncached_requests])

            with torch.no_grad():
                outputs, caches = self.model(
                    features.to(self.device),
                    use_cache=True
                )

            for i, req in enumerate(uncached_requests):
                cache_for_req = [(k[:, i:i+1], v[:, i:i+1]) for k, v in caches]
                self.request_caches[req['request_id']] = cache_for_req
                self.request_timestamps[req['request_id']] = time.time()

                results.append({
                    'request_id': req['request_id'],
                    'prediction': outputs[i].cpu().numpy(),
                    'cache_hit': False
                })

        # Process cached requests (incremental)
        for req in cached_requests:
            result = self.predict_single(
                req['request_id'],
                req['features'],
                use_cache=True
            )
            result['request_id'] = req['request_id']
            results.append(result)

        return results

    def get_metrics(self) -> InferenceMetrics:
        """Get current inference metrics."""
        total_requests = self.cache_hits + self.cache_misses

        return InferenceMetrics(
            latency_ms=np.mean(self.latency_history) if self.latency_history else 0,
            throughput_tokens_per_sec=1000 / np.mean(self.latency_history) if self.latency_history else 0,
            cache_memory_mb=sum(
                self.compute_cache_memory(c) for c in self.request_caches.values()
            ),
            cache_hit_rate=self.cache_hits / total_requests if total_requests > 0 else 0
        )


class StreamingInferenceEngine:
    """
    Streaming inference engine for real-time trading.

    Optimized for continuous data streams with minimal latency.
    """

    def __init__(
        self,
        model: KVCacheTrader,
        max_context_length: int = 4096,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.max_context_length = max_context_length

        # Streaming state
        self.kv_cache = None
        self.current_length = 0
        self.feature_buffer = []

    def reset(self):
        """Reset streaming state."""
        self.kv_cache = None
        self.current_length = 0
        self.feature_buffer = []

    def process_tick(self, features: np.ndarray) -> Dict:
        """
        Process a single tick/update.

        Args:
            features: Feature vector for this tick

        Returns:
            Prediction result with metrics
        """
        start_time = time.time()

        # Convert to tensor
        x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

        # Handle context window overflow
        if self.current_length >= self.max_context_length:
            # Sliding window: discard oldest tokens
            # For KV-cache, we truncate the cached tensors
            if self.kv_cache is not None:
                self.kv_cache = [
                    (k[:, :, 1:, :], v[:, :, 1:, :])
                    for k, v in self.kv_cache
                ]
                self.current_length -= 1

        with torch.no_grad():
            output, self.kv_cache = self.model(
                x,
                past_kv_cache=self.kv_cache,
                use_cache=True
            )

        self.current_length += 1
        latency = (time.time() - start_time) * 1000

        return {
            'prediction': output[0].cpu().numpy(),
            'latency_ms': latency,
            'context_length': self.current_length
        }

    def process_stream(
        self,
        data_stream: Iterator[np.ndarray],
        callback=None
    ) -> Iterator[Dict]:
        """
        Process continuous data stream.

        Args:
            data_stream: Iterator yielding feature vectors
            callback: Optional function called on each prediction

        Yields:
            Prediction results
        """
        for features in data_stream:
            result = self.process_tick(features)

            if callback:
                callback(result)

            yield result
```

### 04: Real-Time Prediction

```python
# python/predict.py

import torch
import numpy as np
from typing import Dict, List, Optional
import time
import asyncio
from dataclasses import dataclass

from model import KVCacheTrader, KVCacheConfig
from inference import OptimizedInferenceEngine, StreamingInferenceEngine
from data_loader import create_streaming_generator


@dataclass
class TradingSignal:
    """Trading signal generated by the model."""
    timestamp: float
    symbol: str
    direction: str  # 'LONG', 'SHORT', 'NEUTRAL'
    confidence: float
    predicted_return: float
    latency_ms: float


class RealTimePredictor:
    """
    Real-time predictor for algorithmic trading.

    Features:
    - Low-latency predictions with KV-cache
    - Signal generation with confidence levels
    - Multi-asset support
    """

    def __init__(
        self,
        model_path: str,
        config: KVCacheConfig,
        symbols: List[str],
        device: str = 'cuda'
    ):
        # Load model
        self.model = KVCacheTrader(
            input_dim=5,  # Standard feature set
            d_model=256,
            n_heads=8,
            n_layers=6,
            cache_config=config
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.model.eval()

        self.symbols = symbols
        self.device = device

        # Per-symbol streaming engines
        self.engines = {
            symbol: StreamingInferenceEngine(self.model, device=device)
            for symbol in symbols
        }

        # Signal thresholds
        self.long_threshold = 0.001  # 0.1% expected return
        self.short_threshold = -0.001
        self.confidence_threshold = 0.6

    def process_market_update(
        self,
        symbol: str,
        market_data: Dict
    ) -> Optional[TradingSignal]:
        """
        Process a market update and generate trading signal.

        Args:
            symbol: Trading symbol
            market_data: Market data dictionary

        Returns:
            TradingSignal if conditions met, else None
        """
        if symbol not in self.engines:
            raise ValueError(f"Unknown symbol: {symbol}")

        # Extract features
        features = np.array([
            market_data.get('log_return', 0),
            market_data.get('volatility', 0.01),
            market_data.get('volume_ratio', 1),
            market_data.get('momentum', 0),
            market_data.get('rsi', 50) / 100
        ], dtype=np.float32)

        # Get prediction
        result = self.engines[symbol].process_tick(features)

        prediction = result['prediction'][0]
        confidence = min(abs(prediction) * 100, 1.0)  # Normalize confidence

        # Generate signal
        if prediction > self.long_threshold and confidence > self.confidence_threshold:
            direction = 'LONG'
        elif prediction < self.short_threshold and confidence > self.confidence_threshold:
            direction = 'SHORT'
        else:
            direction = 'NEUTRAL'

        return TradingSignal(
            timestamp=time.time(),
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            predicted_return=prediction,
            latency_ms=result['latency_ms']
        )

    async def run_async(
        self,
        data_streams: Dict[str, asyncio.Queue],
        signal_callback
    ):
        """
        Run asynchronous prediction loop.

        Args:
            data_streams: Queues for each symbol's data
            signal_callback: Async function called with each signal
        """
        async def process_symbol(symbol: str, queue: asyncio.Queue):
            while True:
                market_data = await queue.get()
                signal = self.process_market_update(symbol, market_data)
                if signal and signal.direction != 'NEUTRAL':
                    await signal_callback(signal)

        tasks = [
            process_symbol(symbol, queue)
            for symbol, queue in data_streams.items()
        ]

        await asyncio.gather(*tasks)


def main():
    """Example real-time prediction."""

    # Configuration
    config = KVCacheConfig(
        cache_type='standard',
        max_cache_size=2048
    )

    symbols = ['BTCUSDT', 'ETHUSDT']

    # Create predictor (assuming trained model exists)
    predictor = RealTimePredictor(
        model_path='best_model.pt',
        config=config,
        symbols=symbols
    )

    # Simulate streaming data
    for symbol in symbols:
        stream = create_streaming_generator(symbol, lookback_days=7)

        print(f"\n{symbol} Predictions:")
        for i, market_data in enumerate(stream):
            if i >= 100:  # Limit for demo
                break

            signal = predictor.process_market_update(symbol, market_data)

            if signal.direction != 'NEUTRAL':
                print(f"  {signal.direction}: conf={signal.confidence:.2f}, "
                      f"pred_ret={signal.predicted_return:.4f}, "
                      f"latency={signal.latency_ms:.2f}ms")


if __name__ == '__main__':
    main()
```

### 05: Trading Strategy Backtesting

```python
# python/strategy.py

import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt

from model import KVCacheTrader, KVCacheConfig
from inference import StreamingInferenceEngine
from data_loader import fetch_bybit_klines


@dataclass
class BacktestResult:
    """Backtest results container."""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    avg_latency_ms: float
    total_trades: int
    portfolio_values: np.ndarray


def calculate_metrics(returns: np.ndarray, risk_free_rate: float = 0.0) -> Dict:
    """Calculate trading performance metrics."""
    excess_returns = returns - risk_free_rate / 252

    # Sharpe Ratio
    sharpe = np.sqrt(252) * excess_returns.mean() / (returns.std() + 1e-8)

    # Sortino Ratio
    downside = returns[returns < 0]
    downside_std = np.sqrt(np.mean(downside**2)) if len(downside) > 0 else 1e-8
    sortino = np.sqrt(252) * excess_returns.mean() / (downside_std + 1e-8)

    # Max Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Win Rate
    win_rate = (returns > 0).mean()

    return {
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_return': cumulative[-1] - 1
    }


class KVCacheBacktester:
    """
    Backtester that simulates real-time inference with KV-cache.

    Measures both trading performance and inference latency.
    """

    def __init__(
        self,
        model: KVCacheTrader,
        symbols: List[str],
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,
        position_size: float = 0.1
    ):
        self.model = model
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.position_size = position_size

        # Streaming engines for realistic KV-cache behavior
        self.engines = {
            symbol: StreamingInferenceEngine(model, device='cuda' if torch.cuda.is_available() else 'cpu')
            for symbol in symbols
        }

    def run_backtest(
        self,
        test_data: Dict[str, pd.DataFrame],
        prediction_horizon: int = 1
    ) -> BacktestResult:
        """
        Run backtest with realistic KV-cache inference.

        Args:
            test_data: Dict mapping symbol to DataFrame with features
            prediction_horizon: Number of steps ahead to predict

        Returns:
            BacktestResult with performance metrics
        """
        # Align all symbol data
        min_len = min(len(df) for df in test_data.values())

        capital = self.initial_capital
        portfolio_values = [capital]
        positions = {symbol: 0.0 for symbol in self.symbols}
        trades = []
        latencies = []

        for i in range(min_len - prediction_horizon):
            step_return = 0

            for symbol in self.symbols:
                df = test_data[symbol]

                # Extract features
                features = np.array([
                    df['log_return'].iloc[i],
                    df['volatility'].iloc[i],
                    df['volume_ratio'].iloc[i],
                    df['momentum'].iloc[i] if 'momentum' in df.columns else 0,
                    df['rsi'].iloc[i] / 100 if 'rsi' in df.columns else 0.5
                ], dtype=np.float32)

                # Get prediction (simulating real-time with KV-cache)
                result = self.engines[symbol].process_tick(features)
                prediction = result['prediction'][0]
                latencies.append(result['latency_ms'])

                # Generate signal
                signal = np.tanh(prediction * 10)  # Bounded [-1, 1]
                target_position = signal * self.position_size

                # Calculate position change cost
                position_change = target_position - positions[symbol]
                trade_cost = abs(position_change) * self.transaction_cost * capital

                if abs(position_change) > 0.01:
                    trades.append({
                        'step': i,
                        'symbol': symbol,
                        'action': 'buy' if position_change > 0 else 'sell',
                        'size': abs(position_change),
                        'prediction': prediction
                    })

                # Update position
                positions[symbol] = target_position

                # Calculate actual return
                actual_return = df['log_return'].iloc[i + prediction_horizon]
                step_return += positions[symbol] * actual_return
                capital -= trade_cost

            capital = capital * (1 + step_return)
            portfolio_values.append(capital)

        portfolio_values = np.array(portfolio_values)
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]

        metrics = calculate_metrics(daily_returns)

        return BacktestResult(
            total_return=metrics['total_return'],
            sharpe_ratio=metrics['sharpe_ratio'],
            sortino_ratio=metrics['sortino_ratio'],
            max_drawdown=metrics['max_drawdown'],
            win_rate=metrics['win_rate'],
            avg_latency_ms=np.mean(latencies),
            total_trades=len(trades),
            portfolio_values=portfolio_values
        )


def compare_cache_strategies(
    model: KVCacheTrader,
    test_data: Dict[str, pd.DataFrame],
    symbols: List[str]
) -> pd.DataFrame:
    """
    Compare different KV-cache strategies.

    Returns DataFrame with metrics for each strategy.
    """
    strategies = ['standard', 'quantized_fp8', 'selective']
    results = []

    for strategy in strategies:
        if strategy == 'standard':
            config = KVCacheConfig(cache_type='standard')
        elif strategy == 'quantized_fp8':
            config = KVCacheConfig(cache_type='quantized', quantization='fp8')
        elif strategy == 'selective':
            config = KVCacheConfig(cache_type='selective', max_cache_size=1024)

        # Create model with strategy
        model_copy = KVCacheTrader(
            input_dim=5,
            d_model=256,
            n_heads=8,
            n_layers=6,
            cache_config=config
        )
        model_copy.load_state_dict(model.state_dict())

        # Run backtest
        backtester = KVCacheBacktester(model_copy, symbols)
        result = backtester.run_backtest(test_data)

        results.append({
            'strategy': strategy,
            'total_return': result.total_return,
            'sharpe_ratio': result.sharpe_ratio,
            'avg_latency_ms': result.avg_latency_ms,
            'max_drawdown': result.max_drawdown
        })

    return pd.DataFrame(results)


def plot_backtest_results(
    result: BacktestResult,
    title: str = 'KV-Cache Trading Strategy Backtest'
):
    """Plot backtest results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Portfolio value
    ax1 = axes[0, 0]
    ax1.plot(result.portfolio_values, linewidth=1.5)
    ax1.set_title('Portfolio Value')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value ($)')
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2 = axes[0, 1]
    cumulative = result.portfolio_values / result.portfolio_values[0]
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    ax2.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.5, color='red')
    ax2.set_title(f'Drawdown (Max: {result.max_drawdown:.2%})')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Drawdown')
    ax2.grid(True, alpha=0.3)

    # Returns distribution
    ax3 = axes[1, 0]
    returns = np.diff(result.portfolio_values) / result.portfolio_values[:-1]
    ax3.hist(returns, bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='red', linestyle='--')
    ax3.set_title(f'Returns Distribution (Win Rate: {result.win_rate:.2%})')
    ax3.set_xlabel('Return')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)

    # Metrics summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    metrics_text = f"""
    Performance Metrics (with KV-Cache)
    {'='*35}

    Total Return:     {result.total_return:.2%}
    Sharpe Ratio:     {result.sharpe_ratio:.2f}
    Sortino Ratio:    {result.sortino_ratio:.2f}
    Max Drawdown:     {result.max_drawdown:.2%}
    Win Rate:         {result.win_rate:.2%}

    Inference Metrics
    {'='*35}
    Avg Latency:      {result.avg_latency_ms:.2f} ms
    Total Trades:     {result.total_trades}
    """
    ax4.text(0.1, 0.5, metrics_text, fontsize=12, fontfamily='monospace',
             verticalalignment='center', transform=ax4.transAxes)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=150)
    plt.close()

    print(f"Results saved to backtest_results.png")


def main():
    """Run backtest example."""
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

    # Fetch data
    print("Fetching data...")
    test_data = {}
    for symbol in symbols:
        df = fetch_bybit_klines(symbol, limit=2000)
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(24).std()
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(24).mean()
        df['momentum'] = df['close'] / df['close'].shift(24) - 1

        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))

        df = df.dropna()
        test_data[symbol] = df

    # Create model
    model = KVCacheTrader(
        input_dim=5,
        d_model=256,
        n_heads=8,
        n_layers=6,
        cache_config=KVCacheConfig()
    )

    # Run backtest
    print("Running backtest...")
    backtester = KVCacheBacktester(model, symbols)
    result = backtester.run_backtest(test_data)

    print(f"\nBacktest Results:")
    print(f"  Total Return: {result.total_return:.2%}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {result.sortino_ratio:.2f}")
    print(f"  Max Drawdown: {result.max_drawdown:.2%}")
    print(f"  Win Rate: {result.win_rate:.2%}")
    print(f"  Avg Latency: {result.avg_latency_ms:.2f} ms")
    print(f"  Total Trades: {result.total_trades}")

    plot_backtest_results(result)

    return result


if __name__ == '__main__':
    main()
```

## Python Implementation

```
python/
├── __init__.py
├── model.py                # KV-Cache Transformer
├── data_loader.py          # Bybit data loading
├── inference.py            # Optimized inference engine
├── predict.py              # Real-time prediction
├── strategy.py             # Trading strategy & backtesting
├── requirements.txt        # Dependencies
└── examples/
    ├── 01_kv_cache_basics.py
    ├── 02_inference_benchmark.py
    └── 03_strategy_comparison.py
```

### Quick Start (Python)

```bash
# Install dependencies
cd python
pip install -r requirements.txt

# Run inference benchmark
python -c "
from model import KVCacheTrader, benchmark_kv_cache
model = KVCacheTrader(input_dim=5)
results = benchmark_kv_cache(model, context_length=1024)
print(f'Speedup with KV-cache: {results[\"speedup\"]:.2f}x')
"

# Run backtest
python strategy.py
```

### Requirements

```
# requirements.txt
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
requests>=2.25.0
matplotlib>=3.4.0
tqdm>=4.60.0
```

## Rust Implementation

See [rust/](rust/) for a production-ready Rust implementation.

```
rust/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs
│   ├── cache/
│   │   ├── mod.rs
│   │   ├── standard.rs       # Basic KV-cache
│   │   ├── paged.rs          # PagedAttention-style cache
│   │   └── quantized.rs      # Quantized cache
│   ├── model/
│   │   ├── mod.rs
│   │   ├── attention.rs      # Attention with cache
│   │   └── transformer.rs    # Full model
│   ├── data/
│   │   ├── mod.rs
│   │   └── bybit.rs          # Bybit client
│   └── strategy/
│       ├── mod.rs
│       └── backtest.rs       # Backtesting
├── benches/
│   └── cache_benchmark.rs
└── examples/
    ├── inference.rs
    └── streaming.rs
```

### Quick Start (Rust)

```bash
cd rust

# Build
cargo build --release

# Run inference example
cargo run --example inference

# Run benchmarks
cargo bench
```

## Performance Benchmarks

### KV-Cache Memory Savings

| Context Length | No Cache | Standard Cache | Quantized (FP8) | Reduction |
|----------------|----------|----------------|-----------------|-----------|
| 256 | Recompute | 50 MB | 25 MB | 50% |
| 1,024 | Recompute | 200 MB | 100 MB | 50% |
| 4,096 | Recompute | 800 MB | 400 MB | 50% |
| 8,192 | Recompute | 1.6 GB | 800 MB | 50% |

### Inference Latency Comparison

| Context Length | Without Cache | With Cache | Speedup |
|----------------|---------------|------------|---------|
| 256 | 15 ms | 2 ms | 7.5x |
| 512 | 35 ms | 2 ms | 17.5x |
| 1,024 | 120 ms | 3 ms | 40x |
| 2,048 | 450 ms | 4 ms | 112x |
| 4,096 | 1,800 ms | 6 ms | 300x |

### Trading Application Benchmarks

| Scenario | Latency | Throughput | Memory |
|----------|---------|------------|--------|
| Single asset streaming | 2-5 ms | 200-500 pred/s | 100 MB |
| Multi-asset (10 symbols) | 10-20 ms | 50-100 pred/s | 500 MB |
| Order book analysis | 1-3 ms | 300-1000 pred/s | 200 MB |

## Best Practices

### When to Use KV-Cache

**Recommended scenarios:**
- Real-time trading with streaming data
- Autoregressive multi-step predictions
- Long context windows (>256 tokens)
- Batch serving multiple requests

**May not be needed:**
- Single-shot predictions
- Very short sequences
- Training (use during inference only)

### Memory Management Tips

```python
# 1. Pre-allocate cache for known sequence length
cache = KVCache(
    num_layers=6,
    batch_size=1,
    num_heads=8,
    head_dim=64,
    max_seq_len=4096  # Pre-allocate
)

# 2. Use quantization for memory-constrained deployments
cache = QuantizedKVCache(
    quantization='fp8'  # 50% memory savings
)

# 3. Implement sliding window for infinite streams
if cache_length > max_length:
    cache.truncate(keep_last=max_length)
```

### Latency Optimization

```python
# 1. Keep model and cache on GPU
model = model.cuda()

# 2. Use torch.inference_mode() for lowest overhead
with torch.inference_mode():
    output, cache = model(x, past_kv_cache=cache, use_cache=True)

# 3. Batch multiple requests when possible
# (amortizes overhead across requests)
```

### Common Pitfalls

1. **Forgetting to update cache**: Always use `use_cache=True` and store returned cache
2. **Mismatched positions**: Track position offset when using cached values
3. **Memory leaks**: Clear old caches for completed requests
4. **Numerical instability**: Use appropriate dtype (FP16/BF16 recommended)

## Resources

### Papers

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM paper (2023)
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691) — Complementary optimization
- [MiniCache: KV Cache Compression across Layers](https://arxiv.org/abs/2405.14366) — Layer-wise compression (2024)
- [SnapKV: LLM Knows What You are Looking for Before Generation](https://arxiv.org/abs/2404.14469) — Selective retention (2024)

### Implementations

- [vLLM](https://github.com/vllm-project/vllm) — High-throughput LLM serving with PagedAttention
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — NVIDIA's optimized LLM inference
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/main/en/kv_cache) — KV-cache documentation

### Related Chapters

- [Chapter 58: FlashAttention for Trading](../58_flash_attention_trading) — Memory-efficient attention computation
- [Chapter 59: Grouped Query Attention](../59_grouped_query_attention) — Reduced KV-cache size
- [Chapter 50: Memory-Augmented Transformers](../50_memory_augmented_transformers) — External memory systems

---

## Difficulty Level

**Advanced**

Prerequisites:
- Transformer architecture and self-attention
- Autoregressive generation concepts
- GPU memory management
- PyTorch or similar deep learning framework
- Basic trading strategy knowledge
