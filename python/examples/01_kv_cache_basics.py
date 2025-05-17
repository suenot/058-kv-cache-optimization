"""
Example 01: KV-Cache Basics

This example demonstrates the fundamental concepts of KV-Cache
and how it speeds up Transformer inference.
"""

import torch
import time
import sys
sys.path.append('..')

from model import KVCacheTrader, KVCache, benchmark_kv_cache


def demonstrate_kv_cache_concept():
    """Show how KV-cache eliminates redundant computation."""
    print("=" * 60)
    print("KV-Cache Concept Demonstration")
    print("=" * 60)

    # Create a simple model
    model = KVCacheTrader(
        input_dim=5,
        d_model=128,
        n_heads=4,
        n_layers=3,
        max_seq_len=512
    )
    model.eval()

    # Initial sequence (e.g., first 50 time steps of market data)
    initial_seq = torch.randn(1, 50, 5)

    print("\n1. Initial Forward Pass (Full Sequence)")
    print("-" * 40)

    with torch.no_grad():
        start = time.time()
        output, kv_cache = model(initial_seq, use_cache=True)
        initial_time = (time.time() - start) * 1000

    print(f"   Sequence length: {initial_seq.shape[1]}")
    print(f"   Output shape: {output.shape}")
    print(f"   Time: {initial_time:.2f} ms")
    print(f"   Cache created: {len(kv_cache)} layers")

    # New single token (e.g., new market data point)
    new_token = torch.randn(1, 1, 5)

    print("\n2. Incremental Update (Single New Token)")
    print("-" * 40)

    with torch.no_grad():
        start = time.time()
        output2, kv_cache2 = model(new_token, past_kv_cache=kv_cache, use_cache=True)
        incremental_time = (time.time() - start) * 1000

    print(f"   New tokens: 1")
    print(f"   Output shape: {output2.shape}")
    print(f"   Time: {incremental_time:.2f} ms")
    print(f"   Cache updated: {kv_cache2[0][0].shape[2]} total tokens")

    # Compare with full recomputation
    full_seq = torch.cat([initial_seq, new_token], dim=1)

    print("\n3. Comparison: Full Recomputation vs Incremental")
    print("-" * 40)

    with torch.no_grad():
        start = time.time()
        output3, _ = model(full_seq, use_cache=False)
        full_time = (time.time() - start) * 1000

    print(f"   Full recomputation time: {full_time:.2f} ms")
    print(f"   Incremental update time: {incremental_time:.2f} ms")
    print(f"   Speedup: {full_time / incremental_time:.1f}x")

    # Verify outputs are similar
    print(f"\n   Output difference: {torch.abs(output2 - output3).max().item():.6f}")
    print("   (Small differences due to floating point precision)")


def benchmark_different_context_lengths():
    """Benchmark KV-cache performance at different context lengths."""
    print("\n" + "=" * 60)
    print("Context Length Benchmark")
    print("=" * 60)

    model = KVCacheTrader(
        input_dim=5,
        d_model=256,
        n_heads=8,
        n_layers=6,
        max_seq_len=4096
    )

    context_lengths = [64, 128, 256, 512, 1024]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"{'Context':<10} {'No Cache':<15} {'With Cache':<15} {'Speedup':<10}")
    print("-" * 50)

    for ctx_len in context_lengths:
        results = benchmark_kv_cache(
            model,
            context_length=ctx_len,
            num_iterations=20,
            device=device
        )
        print(f"{ctx_len:<10} {results['no_cache_ms']:<15.2f} {results['with_cache_ms']:<15.2f} {results['speedup']:<10.1f}x")


def demonstrate_streaming_inference():
    """Simulate streaming market data processing."""
    print("\n" + "=" * 60)
    print("Streaming Inference Simulation")
    print("=" * 60)

    model = KVCacheTrader(
        input_dim=5,
        d_model=128,
        n_heads=4,
        n_layers=3
    )
    model.eval()

    print("\nSimulating 20 streaming market data updates...")
    print(f"{'Tick':<8} {'Prediction':<15} {'Latency (ms)':<15} {'Cache Size':<12}")
    print("-" * 50)

    kv_cache = None

    for tick in range(20):
        # Simulate new market data
        features = torch.randn(1, 1, 5)

        start = time.time()
        with torch.no_grad():
            output, kv_cache = model(
                features,
                past_kv_cache=kv_cache,
                use_cache=True
            )
        latency = (time.time() - start) * 1000

        cache_tokens = kv_cache[0][0].shape[2] if kv_cache else 0
        prediction = output[0, 0].item()

        print(f"{tick + 1:<8} {prediction:<15.4f} {latency:<15.2f} {cache_tokens:<12}")


def main():
    print("\n" + "=" * 60)
    print("   KV-CACHE BASICS - Educational Examples")
    print("=" * 60)

    demonstrate_kv_cache_concept()
    benchmark_different_context_lengths()
    demonstrate_streaming_inference()

    print("\n" + "=" * 60)
    print("   Examples completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
