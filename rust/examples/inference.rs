//! Example: Basic KV-Cache inference
//!
//! This example demonstrates how to use the KV-cache Transformer
//! for efficient inference in trading applications.

use kv_cache_trading::{KVCacheTrader, KVCacheConfig, StreamingEngine, Prediction};

fn main() -> anyhow::Result<()> {
    println!("KV-Cache Inference Example");
    println!("===========================\n");

    // Create model configuration
    let config = KVCacheConfig::default();

    // Create model
    println!("Creating KV-Cache Transformer...");
    let model = KVCacheTrader::new(
        5,    // input_dim (features)
        128,  // d_model
        4,    // n_heads
        3,    // n_layers
        config,
    )?;

    println!("Model created with {} layers, {} heads\n", model.num_layers(), model.num_heads());

    // Create streaming engine
    let mut engine = StreamingEngine::new(model, 4096);

    // Simulate streaming market data
    println!("Processing 20 ticks of market data...\n");
    println!("{:<8} {:<15} {:<15} {:<12}", "Tick", "Prediction", "Latency (ms)", "Cache (KB)");
    println!("{}", "-".repeat(50));

    for tick in 0..20 {
        // Simulated market features
        let features = vec![
            0.001 * (tick as f32 % 5 - 2.0),  // log_return
            0.02,                               // volatility
            1.0 + 0.1 * (tick as f32 % 3 - 1.0), // volume_ratio
            0.05 * (tick as f32 % 4 - 2.0),   // momentum
            0.5 + 0.1 * (tick as f32 % 5 - 2.0), // rsi (normalized)
        ];

        let result = engine.process_tick(&features)?;

        println!(
            "{:<8} {:<15.6} {:<15.3} {:<12.2}",
            tick + 1,
            result.value,
            result.latency_ms,
            engine.cache_memory_bytes() as f64 / 1024.0
        );
    }

    println!("\n{}", "=".repeat(50));
    println!("Final context length: {}", 20);
    println!("Total cache memory: {:.2} KB", engine.cache_memory_bytes() as f64 / 1024.0);

    // Demonstrate cache reset
    println!("\nResetting cache...");
    engine.reset();
    println!("Cache memory after reset: {} bytes", engine.cache_memory_bytes());

    println!("\nExample completed successfully!");

    Ok(())
}
