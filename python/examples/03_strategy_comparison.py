"""
Example 03: Strategy Comparison

This example compares trading performance across different
KV-cache optimization strategies.
"""

import numpy as np
import pandas as pd
import torch
import sys
sys.path.append('..')

from model import KVCacheTrader, KVCacheConfig
from strategy import (
    KVCacheBacktester,
    calculate_metrics,
    compare_cache_strategies,
    plot_backtest_results,
    prepare_test_data
)
from data_loader import fetch_bybit_klines


def create_synthetic_data(symbols, num_points=500):
    """Create synthetic market data for testing."""
    test_data = {}

    for symbol in symbols:
        dates = pd.date_range(start='2024-01-01', periods=num_points, freq='h')
        prices = 100 * np.cumprod(1 + np.random.randn(num_points) * 0.01)
        volumes = np.random.rand(num_points) * 1000

        df = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'volume': volumes
        })

        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(24).std()
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(24).mean()
        df['momentum'] = df['close'] / df['close'].shift(24) - 1
        df['rsi'] = 50 + np.random.randn(num_points) * 10

        df = df.dropna()
        test_data[symbol] = df

    return test_data


def run_basic_backtest():
    """Run a basic backtest with KV-cache enabled model."""
    print("=" * 70)
    print("Basic Backtest with KV-Cache")
    print("=" * 70)

    symbols = ['BTCUSDT', 'ETHUSDT']

    # Try to fetch real data, fall back to synthetic
    print("\nFetching market data...")
    try:
        test_data = prepare_test_data(symbols, limit=300)
        print("Using real market data from Bybit")
    except Exception as e:
        print(f"Could not fetch real data: {e}")
        print("Using synthetic data instead")
        test_data = create_synthetic_data(symbols, num_points=300)

    # Create model
    model = KVCacheTrader(
        input_dim=5,
        d_model=256,
        n_heads=8,
        n_layers=6,
        cache_config=KVCacheConfig()
    )

    # Run backtest
    print("\nRunning backtest...")
    backtester = KVCacheBacktester(
        model,
        symbols,
        initial_capital=100000,
        transaction_cost=0.001,
        position_size=0.1
    )

    result = backtester.run_backtest(test_data)

    print("\nResults:")
    print("-" * 40)
    print(f"  Total Return:  {result.total_return:.2%}")
    print(f"  Sharpe Ratio:  {result.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {result.sortino_ratio:.2f}")
    print(f"  Max Drawdown:  {result.max_drawdown:.2%}")
    print(f"  Win Rate:      {result.win_rate:.2%}")
    print(f"  Total Trades:  {result.total_trades}")
    print(f"  Avg Latency:   {result.avg_latency_ms:.2f} ms")

    return result


def compare_cache_configurations():
    """Compare different KV-cache configurations."""
    print("\n" + "=" * 70)
    print("Cache Configuration Comparison")
    print("=" * 70)

    symbols = ['BTCUSDT', 'ETHUSDT']

    print("\nPreparing data...")
    test_data = create_synthetic_data(symbols, num_points=300)

    # Different cache configurations
    configs = [
        ('Standard', KVCacheConfig(cache_type='standard')),
        ('Limited (512)', KVCacheConfig(cache_type='standard', max_cache_size=512)),
        ('Limited (256)', KVCacheConfig(cache_type='standard', max_cache_size=256)),
    ]

    results = []

    for name, config in configs:
        print(f"\nTesting {name}...")

        model = KVCacheTrader(
            input_dim=5,
            d_model=256,
            n_heads=8,
            n_layers=6,
            cache_config=config
        )

        backtester = KVCacheBacktester(model, symbols)
        result = backtester.run_backtest(test_data)

        results.append({
            'Configuration': name,
            'Return': f"{result.total_return:.2%}",
            'Sharpe': f"{result.sharpe_ratio:.2f}",
            'Max DD': f"{result.max_drawdown:.2%}",
            'Latency (ms)': f"{result.avg_latency_ms:.2f}",
            'Trades': result.total_trades
        })

    print("\n" + "-" * 80)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))


def compare_model_architectures():
    """Compare different model architectures."""
    print("\n" + "=" * 70)
    print("Model Architecture Comparison")
    print("=" * 70)

    symbols = ['BTCUSDT']

    print("\nPreparing data...")
    test_data = create_synthetic_data(symbols, num_points=300)

    # Different architectures
    architectures = [
        ('Tiny (2 layers)', {'d_model': 64, 'n_heads': 2, 'n_layers': 2}),
        ('Small (4 layers)', {'d_model': 128, 'n_heads': 4, 'n_layers': 4}),
        ('Medium (6 layers)', {'d_model': 256, 'n_heads': 8, 'n_layers': 6}),
        ('Large (8 layers)', {'d_model': 512, 'n_heads': 8, 'n_layers': 8}),
    ]

    results = []

    for name, arch in architectures:
        print(f"\nTesting {name}...")

        model = KVCacheTrader(
            input_dim=5,
            d_model=arch['d_model'],
            n_heads=arch['n_heads'],
            n_layers=arch['n_layers']
        )

        params = sum(p.numel() for p in model.parameters()) / 1e6

        backtester = KVCacheBacktester(model, symbols)
        result = backtester.run_backtest(test_data)

        results.append({
            'Architecture': name,
            'Params (M)': f"{params:.2f}",
            'Latency (ms)': f"{result.avg_latency_ms:.2f}",
            'Sharpe': f"{result.sharpe_ratio:.2f}",
            'Max DD': f"{result.max_drawdown:.2%}"
        })

    print("\n" + "-" * 80)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))


def analyze_latency_distribution():
    """Analyze latency distribution during backtest."""
    print("\n" + "=" * 70)
    print("Latency Distribution Analysis")
    print("=" * 70)

    symbols = ['BTCUSDT']
    test_data = create_synthetic_data(symbols, num_points=500)

    model = KVCacheTrader(
        input_dim=5,
        d_model=256,
        n_heads=8,
        n_layers=6
    )

    # Run with latency tracking
    from inference import StreamingInferenceEngine

    engine = StreamingInferenceEngine(model, device='cpu')
    latencies = []

    print("\nProcessing 500 ticks...")

    df = test_data[symbols[0]]
    for i in range(len(df)):
        features = np.array([
            df['log_return'].iloc[i],
            df['volatility'].iloc[i],
            df['volume_ratio'].iloc[i],
            df['momentum'].iloc[i],
            df['rsi'].iloc[i] / 100
        ], dtype=np.float32)

        features = np.nan_to_num(features, 0)
        result = engine.process_tick(features)
        latencies.append(result['latency_ms'])

    latencies = np.array(latencies)

    print("\nLatency Statistics:")
    print("-" * 40)
    print(f"  Mean:    {np.mean(latencies):.2f} ms")
    print(f"  Std:     {np.std(latencies):.2f} ms")
    print(f"  Min:     {np.min(latencies):.2f} ms")
    print(f"  Max:     {np.max(latencies):.2f} ms")
    print(f"  P50:     {np.percentile(latencies, 50):.2f} ms")
    print(f"  P90:     {np.percentile(latencies, 90):.2f} ms")
    print(f"  P99:     {np.percentile(latencies, 99):.2f} ms")

    # Show latency trend
    print("\nLatency Trend (every 100 ticks):")
    for i in range(0, len(latencies), 100):
        chunk = latencies[i:i+100]
        print(f"  Ticks {i}-{i+100}: mean={np.mean(chunk):.2f} ms")


def main():
    print("\n" + "=" * 70)
    print("   KV-CACHE STRATEGY COMPARISON")
    print("=" * 70)

    # Basic backtest
    result = run_basic_backtest()

    # Try to plot results
    try:
        plot_backtest_results(result, save_path='basic_backtest.png')
    except Exception as e:
        print(f"\nCould not create plot: {e}")

    # Compare configurations
    compare_cache_configurations()

    # Compare architectures
    compare_model_architectures()

    # Latency analysis
    analyze_latency_distribution()

    print("\n" + "=" * 70)
    print("   Strategy comparison completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()
