//! Example: Streaming market data processing
//!
//! This example demonstrates real-time streaming inference
//! with KV-cache optimization.

use kv_cache_trading::{KVCacheTrader, KVCacheConfig, StreamingEngine, OptimizedInferenceEngine};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("Streaming Inference Example");
    println!("============================\n");

    // Create model
    let config = KVCacheConfig::default();
    let model = KVCacheTrader::new(5, 256, 8, 6, config)?;

    println!("Testing different inference modes...\n");

    // Test 1: Streaming Engine
    println!("1. StreamingEngine (continuous stream)");
    println!("-" .repeat(40));

    let mut streaming_engine = StreamingEngine::new(
        KVCacheTrader::new(5, 256, 8, 6, KVCacheConfig::default())?,
        4096
    );

    let start = Instant::now();
    let num_ticks = 100;

    for _ in 0..num_ticks {
        let features = vec![0.001, 0.02, 1.0, 0.05, 0.5];
        let _ = streaming_engine.process_tick(&features)?;
    }

    let streaming_time = start.elapsed();
    let streaming_throughput = num_ticks as f64 / streaming_time.as_secs_f64();

    println!("   Ticks processed: {}", num_ticks);
    println!("   Total time: {:.2} ms", streaming_time.as_secs_f64() * 1000.0);
    println!("   Throughput: {:.1} ticks/sec", streaming_throughput);
    println!("   Cache memory: {:.2} KB", streaming_engine.cache_memory_bytes() as f64 / 1024.0);

    // Test 2: Optimized Inference Engine
    println!("\n2. OptimizedInferenceEngine (request-based)");
    println!("{}", "-".repeat(40));

    let mut opt_engine = OptimizedInferenceEngine::new(
        KVCacheTrader::new(5, 256, 8, 6, KVCacheConfig::default())?,
        100.0  // max cache memory MB
    );

    let start = Instant::now();
    let num_requests = 50;

    // First round: cold requests
    for i in 0..num_requests {
        let features = vec![0.001, 0.02, 1.0, 0.05, 0.5];
        let _ = opt_engine.predict(&format!("req_{}", i), &features)?;
    }

    // Second round: warm requests (cache hits)
    for i in 0..num_requests {
        let features = vec![0.002, 0.025, 1.1, 0.06, 0.55];
        let _ = opt_engine.predict(&format!("req_{}", i % 10), &features)?;
    }

    let opt_time = start.elapsed();

    println!("   Total requests: {}", num_requests * 2);
    println!("   Total time: {:.2} ms", opt_time.as_secs_f64() * 1000.0);
    println!("   Cache hit rate: {:.1}%", opt_engine.cache_hit_rate() * 100.0);

    // Test 3: Context length impact
    println!("\n3. Context Length Impact");
    println!("{}", "-".repeat(40));

    let context_lengths = [10, 50, 100, 200, 500];

    for &ctx_len in &context_lengths {
        let mut engine = StreamingEngine::new(
            KVCacheTrader::new(5, 128, 4, 3, KVCacheConfig::default())?,
            ctx_len
        );

        let start = Instant::now();

        for _ in 0..ctx_len {
            let features = vec![0.001, 0.02, 1.0, 0.05, 0.5];
            let _ = engine.process_tick(&features)?;
        }

        let time = start.elapsed();
        let avg_latency = time.as_secs_f64() * 1000.0 / ctx_len as f64;

        println!(
            "   Context {}: avg latency = {:.3} ms, memory = {:.1} KB",
            ctx_len,
            avg_latency,
            engine.cache_memory_bytes() as f64 / 1024.0
        );
    }

    // Test 4: Throughput benchmark
    println!("\n4. Throughput Benchmark");
    println!("{}", "-".repeat(40));

    let mut engine = StreamingEngine::new(
        KVCacheTrader::new(5, 256, 8, 6, KVCacheConfig::default())?,
        1000
    );

    let warmup = 50;
    let benchmark_ticks = 500;

    // Warmup
    for _ in 0..warmup {
        let features = vec![0.001, 0.02, 1.0, 0.05, 0.5];
        let _ = engine.process_tick(&features)?;
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..benchmark_ticks {
        let features = vec![0.001, 0.02, 1.0, 0.05, 0.5];
        let _ = engine.process_tick(&features)?;
    }
    let benchmark_time = start.elapsed();

    let throughput = benchmark_ticks as f64 / benchmark_time.as_secs_f64();
    let avg_latency = benchmark_time.as_secs_f64() * 1000.0 / benchmark_ticks as f64;

    println!("   Throughput: {:.1} predictions/sec", throughput);
    println!("   Average latency: {:.3} ms", avg_latency);
    println!("   P99 estimate: {:.3} ms (approx)", avg_latency * 1.5);

    println!("\n{}", "=".repeat(50));
    println!("Streaming example completed successfully!");

    Ok(())
}
