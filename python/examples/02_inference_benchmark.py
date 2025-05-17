"""
Example 02: Inference Benchmarking

This example benchmarks KV-cache inference performance across
different configurations and compares with/without caching.
"""

import torch
import numpy as np
import time
import sys
sys.path.append('..')

from model import KVCacheTrader, KVCacheConfig, benchmark_kv_cache
from inference import OptimizedInferenceEngine, StreamingInferenceEngine


def benchmark_model_sizes():
    """Benchmark different model configurations."""
    print("=" * 70)
    print("Model Size Benchmark")
    print("=" * 70)

    configs = [
        {'d_model': 128, 'n_heads': 4, 'n_layers': 3, 'name': 'Small'},
        {'d_model': 256, 'n_heads': 8, 'n_layers': 6, 'name': 'Medium'},
        {'d_model': 512, 'n_heads': 8, 'n_layers': 8, 'name': 'Large'},
    ]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"Context length: 512 tokens")
    print()

    print(f"{'Model':<10} {'Params':<12} {'No Cache':<15} {'With Cache':<15} {'Speedup':<10}")
    print("-" * 62)

    for cfg in configs:
        model = KVCacheTrader(
            input_dim=5,
            d_model=cfg['d_model'],
            n_heads=cfg['n_heads'],
            n_layers=cfg['n_layers']
        )

        # Count parameters
        params = sum(p.numel() for p in model.parameters())

        results = benchmark_kv_cache(
            model,
            context_length=512,
            num_iterations=20,
            device=device
        )

        print(f"{cfg['name']:<10} {params/1e6:.2f}M{'':<5} {results['no_cache_ms']:<15.2f} "
              f"{results['with_cache_ms']:<15.2f} {results['speedup']:<10.1f}x")


def benchmark_optimized_engine():
    """Benchmark the OptimizedInferenceEngine."""
    print("\n" + "=" * 70)
    print("OptimizedInferenceEngine Benchmark")
    print("=" * 70)

    model = KVCacheTrader(
        input_dim=5,
        d_model=256,
        n_heads=8,
        n_layers=6
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    engine = OptimizedInferenceEngine(model, device=device)

    print(f"\nDevice: {device}")
    print(f"Processing 100 requests...")
    print()

    # Initial requests
    latencies_cold = []
    for i in range(50):
        features = torch.randn(1, 100, 5)
        result = engine.predict_single(f'request_{i}', features)
        latencies_cold.append(result['latency_ms'])

    # Incremental updates (cache hits)
    latencies_warm = []
    for i in range(50):
        # Add more features to existing requests
        features = torch.randn(1, 110, 5)
        result = engine.predict_single(f'request_{i % 10}', features)
        latencies_warm.append(result['latency_ms'])

    metrics = engine.get_metrics()

    print(f"Cold start (new request):")
    print(f"  Mean latency: {np.mean(latencies_cold):.2f} ms")
    print(f"  P50 latency:  {np.percentile(latencies_cold, 50):.2f} ms")
    print(f"  P99 latency:  {np.percentile(latencies_cold, 99):.2f} ms")
    print()
    print(f"Warm (cache hit):")
    print(f"  Mean latency: {np.mean(latencies_warm):.2f} ms")
    print(f"  P50 latency:  {np.percentile(latencies_warm, 50):.2f} ms")
    print(f"  P99 latency:  {np.percentile(latencies_warm, 99):.2f} ms")
    print()
    print(f"Cache hit rate: {metrics.cache_hit_rate:.2%}")
    print(f"Total cache memory: {metrics.cache_memory_mb:.2f} MB")


def benchmark_streaming_throughput():
    """Benchmark streaming inference throughput."""
    print("\n" + "=" * 70)
    print("Streaming Throughput Benchmark")
    print("=" * 70)

    model = KVCacheTrader(
        input_dim=5,
        d_model=256,
        n_heads=8,
        n_layers=6
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    engine = StreamingInferenceEngine(model, device=device)

    print(f"\nDevice: {device}")
    print()

    # Measure throughput at different batch sizes
    tick_counts = [100, 500, 1000]

    for num_ticks in tick_counts:
        engine.reset()

        start = time.time()
        for _ in range(num_ticks):
            features = np.random.randn(5).astype(np.float32)
            _ = engine.process_tick(features)
        total_time = time.time() - start

        throughput = num_ticks / total_time
        avg_latency = total_time / num_ticks * 1000

        print(f"Ticks: {num_ticks}")
        print(f"  Total time: {total_time:.2f} s")
        print(f"  Throughput: {throughput:.1f} ticks/s")
        print(f"  Avg latency: {avg_latency:.2f} ms")
        print(f"  Cache memory: {engine.get_cache_memory_mb():.2f} MB")
        print()


def benchmark_memory_efficiency():
    """Benchmark memory usage with different context lengths."""
    print("=" * 70)
    print("Memory Efficiency Benchmark")
    print("=" * 70)

    model = KVCacheTrader(
        input_dim=5,
        d_model=256,
        n_heads=8,
        n_layers=6
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print()

    print(f"{'Context Length':<18} {'Cache Memory (MB)':<20} {'Memory/Token (KB)':<18}")
    print("-" * 56)

    context_lengths = [128, 256, 512, 1024, 2048]

    for ctx_len in context_lengths:
        engine = StreamingInferenceEngine(model, device=device)

        # Process tokens
        for _ in range(ctx_len):
            features = np.random.randn(5).astype(np.float32)
            _ = engine.process_tick(features)

        memory_mb = engine.get_cache_memory_mb()
        memory_per_token = memory_mb * 1024 / ctx_len

        print(f"{ctx_len:<18} {memory_mb:<20.2f} {memory_per_token:<18.2f}")


def benchmark_batch_vs_single():
    """Compare batch and single request processing."""
    print("\n" + "=" * 70)
    print("Batch vs Single Request Benchmark")
    print("=" * 70)

    model = KVCacheTrader(
        input_dim=5,
        d_model=256,
        n_heads=8,
        n_layers=6
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    engine = OptimizedInferenceEngine(model, device=device)

    print(f"\nDevice: {device}")
    print()

    num_requests = 10

    # Single request processing
    engine.clear_cache()
    start = time.time()
    for i in range(num_requests):
        features = torch.randn(1, 100, 5)
        _ = engine.predict_single(f'single_{i}', features)
    single_time = time.time() - start

    # Batch processing
    engine.clear_cache()
    requests = [
        {'request_id': f'batch_{i}', 'features': torch.randn(100, 5)}
        for i in range(num_requests)
    ]

    start = time.time()
    _ = engine.predict_batch(requests)
    batch_time = time.time() - start

    print(f"Processing {num_requests} requests:")
    print(f"  Single request mode: {single_time*1000:.2f} ms total ({single_time*1000/num_requests:.2f} ms/req)")
    print(f"  Batch mode: {batch_time*1000:.2f} ms total ({batch_time*1000/num_requests:.2f} ms/req)")
    print(f"  Speedup: {single_time/batch_time:.2f}x")


def main():
    print("\n" + "=" * 70)
    print("   KV-CACHE INFERENCE BENCHMARKS")
    print("=" * 70)

    benchmark_model_sizes()
    benchmark_memory_efficiency()
    benchmark_streaming_throughput()
    benchmark_optimized_engine()
    benchmark_batch_vs_single()

    print("\n" + "=" * 70)
    print("   Benchmarks completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()
