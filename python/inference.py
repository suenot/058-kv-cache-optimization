"""
Optimized Inference Engine for Trading

This module provides efficient inference engines with KV-cache support
for low-latency trading predictions.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Iterator
import time
from dataclasses import dataclass
from collections import deque

from .model import KVCacheTrader, KVCacheConfig


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
        # Use CPU if CUDA not available
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'

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

    def clear_cache(self, request_id: Optional[str] = None):
        """Clear cache for a specific request or all caches."""
        if request_id is not None:
            if request_id in self.request_caches:
                del self.request_caches[request_id]
                del self.request_timestamps[request_id]
        else:
            self.request_caches.clear()
            self.request_timestamps.clear()

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
        # Use CPU if CUDA not available
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'

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

    def get_cache_memory_mb(self) -> float:
        """Get current cache memory usage in MB."""
        if self.kv_cache is None or self.kv_cache[0] is None:
            return 0

        total_bytes = 0
        for k, v in self.kv_cache:
            if k is not None:
                total_bytes += k.numel() * k.element_size()
                total_bytes += v.numel() * v.element_size()

        return total_bytes / (1024 * 1024)


if __name__ == '__main__':
    from .model import KVCacheTrader

    # Create model
    model = KVCacheTrader(input_dim=5, d_model=128, n_heads=4, n_layers=3)

    # Test OptimizedInferenceEngine
    print("Testing OptimizedInferenceEngine...")
    engine = OptimizedInferenceEngine(model, device='cpu')

    features = torch.randn(1, 100, 5)
    result = engine.predict_single('test_request', features)
    print(f"First prediction: latency={result['latency_ms']:.2f}ms, cache_hit={result['cache_hit']}")

    # Second request with more data (should use cache)
    features2 = torch.randn(1, 110, 5)
    result2 = engine.predict_single('test_request', features2)
    print(f"Second prediction: latency={result2['latency_ms']:.2f}ms, cache_hit={result2['cache_hit']}")

    metrics = engine.get_metrics()
    print(f"Metrics: {metrics}")

    # Test StreamingInferenceEngine
    print("\nTesting StreamingInferenceEngine...")
    streaming_engine = StreamingInferenceEngine(model, device='cpu')

    for i in range(10):
        features = np.random.randn(5).astype(np.float32)
        result = streaming_engine.process_tick(features)
        print(f"Tick {i+1}: latency={result['latency_ms']:.2f}ms, context={result['context_length']}")
