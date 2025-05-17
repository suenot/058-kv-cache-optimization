"""
KV-Cache Optimization for Algorithmic Trading

This module provides efficient KV-cache implementations for Transformer-based
trading systems, enabling low-latency inference for real-time market prediction.

Key components:
- KVCacheTrader: Transformer model with KV-cache support
- OptimizedInferenceEngine: Efficient batch inference
- StreamingInferenceEngine: Real-time streaming predictions
- KVCacheBacktester: Backtesting with realistic inference simulation
"""

from .model import (
    KVCache,
    KVCacheConfig,
    KVCacheAttention,
    KVCacheTransformerBlock,
    KVCacheTrader,
    benchmark_kv_cache,
)

from .data_loader import (
    fetch_bybit_klines,
    create_streaming_generator,
    prepare_kv_cache_benchmark_data,
)

from .inference import (
    InferenceMetrics,
    OptimizedInferenceEngine,
    StreamingInferenceEngine,
)

from .predict import (
    TradingSignal,
    RealTimePredictor,
)

from .strategy import (
    BacktestResult,
    KVCacheBacktester,
    calculate_metrics,
    compare_cache_strategies,
    plot_backtest_results,
)

__version__ = "0.1.0"
__all__ = [
    # Model
    "KVCache",
    "KVCacheConfig",
    "KVCacheAttention",
    "KVCacheTransformerBlock",
    "KVCacheTrader",
    "benchmark_kv_cache",
    # Data
    "fetch_bybit_klines",
    "create_streaming_generator",
    "prepare_kv_cache_benchmark_data",
    # Inference
    "InferenceMetrics",
    "OptimizedInferenceEngine",
    "StreamingInferenceEngine",
    # Prediction
    "TradingSignal",
    "RealTimePredictor",
    # Strategy
    "BacktestResult",
    "KVCacheBacktester",
    "calculate_metrics",
    "compare_cache_strategies",
    "plot_backtest_results",
]
