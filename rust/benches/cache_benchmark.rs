//! Benchmarks for KV-Cache implementations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ndarray::Array4;

// Note: These benchmarks require the kv_cache_trading crate to be built
// Run with: cargo bench

fn benchmark_standard_cache(c: &mut Criterion) {
    use kv_cache_trading::KVCache;

    let mut group = c.benchmark_group("standard_cache");

    for num_tokens in [10, 50, 100, 500].iter() {
        group.bench_with_input(
            BenchmarkId::new("update", num_tokens),
            num_tokens,
            |b, &num_tokens| {
                let mut cache = KVCache::new(6, None);

                b.iter(|| {
                    for _ in 0..num_tokens {
                        let keys = Array4::zeros((1, 8, 1, 64));
                        let values = Array4::zeros((1, 8, 1, 64));
                        cache.update(0, black_box(keys), black_box(values));
                    }
                    cache.clear();
                });
            },
        );
    }

    group.finish();
}

fn benchmark_streaming_engine(c: &mut Criterion) {
    use kv_cache_trading::{KVCacheTrader, KVCacheConfig, StreamingEngine};

    let mut group = c.benchmark_group("streaming_inference");

    // Create model once
    let config = KVCacheConfig::default();
    let model = KVCacheTrader::new(5, 128, 4, 3, config).unwrap();
    let mut engine = StreamingEngine::new(model, 1000);

    group.bench_function("single_tick", |b| {
        b.iter(|| {
            let features = vec![0.001, 0.02, 1.0, 0.05, 0.5];
            let _ = engine.process_tick(black_box(&features));
        });
    });

    group.finish();
}

fn benchmark_context_lengths(c: &mut Criterion) {
    use kv_cache_trading::{KVCacheTrader, KVCacheConfig, StreamingEngine};

    let mut group = c.benchmark_group("context_length");

    for ctx_len in [100, 500, 1000, 2000].iter() {
        group.bench_with_input(
            BenchmarkId::new("fill_context", ctx_len),
            ctx_len,
            |b, &ctx_len| {
                b.iter(|| {
                    let config = KVCacheConfig::default();
                    let model = KVCacheTrader::new(5, 128, 4, 3, config).unwrap();
                    let mut engine = StreamingEngine::new(model, ctx_len);

                    for _ in 0..ctx_len {
                        let features = vec![0.001, 0.02, 1.0, 0.05, 0.5];
                        let _ = engine.process_tick(black_box(&features));
                    }
                });
            },
        );
    }

    group.finish();
}

fn benchmark_model_sizes(c: &mut Criterion) {
    use kv_cache_trading::{KVCacheTrader, KVCacheConfig, StreamingEngine};

    let mut group = c.benchmark_group("model_size");

    let configs = vec![
        ("tiny", 64, 2, 2),
        ("small", 128, 4, 3),
        ("medium", 256, 8, 6),
    ];

    for (name, d_model, n_heads, n_layers) in configs {
        group.bench_function(name, |b| {
            let config = KVCacheConfig::default();
            let model = KVCacheTrader::new(5, d_model, n_heads, n_layers, config).unwrap();
            let mut engine = StreamingEngine::new(model, 100);

            // Warmup
            for _ in 0..10 {
                let features = vec![0.001, 0.02, 1.0, 0.05, 0.5];
                let _ = engine.process_tick(&features);
            }

            b.iter(|| {
                let features = vec![0.001, 0.02, 1.0, 0.05, 0.5];
                let _ = engine.process_tick(black_box(&features));
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_standard_cache,
    benchmark_streaming_engine,
    benchmark_context_lengths,
    benchmark_model_sizes,
);

criterion_main!(benches);
