"""
Trading Strategy and Backtesting

This module provides backtesting capabilities for trading strategies
that use KV-cache optimized models.
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Optional
from dataclasses import dataclass

from .model import KVCacheTrader, KVCacheConfig
from .inference import StreamingInferenceEngine
from .data_loader import fetch_bybit_klines


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
    if len(returns) == 0:
        return {
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'total_return': 0
        }

    excess_returns = returns - risk_free_rate / 252

    # Sharpe Ratio
    std = returns.std()
    sharpe = np.sqrt(252) * excess_returns.mean() / (std + 1e-8)

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
        'total_return': cumulative[-1] - 1 if len(cumulative) > 0 else 0
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

        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Streaming engines for realistic KV-cache behavior
        self.engines = {
            symbol: StreamingInferenceEngine(model, device=device)
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
        # Reset all engines
        for engine in self.engines.values():
            engine.reset()

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
                    df['volatility'].iloc[i] if 'volatility' in df.columns else 0.01,
                    df['volume_ratio'].iloc[i] if 'volume_ratio' in df.columns else 1.0,
                    df['momentum'].iloc[i] if 'momentum' in df.columns else 0,
                    df['rsi'].iloc[i] / 100 if 'rsi' in df.columns else 0.5
                ], dtype=np.float32)

                # Handle NaN values
                features = np.nan_to_num(features, 0)

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
                if not np.isnan(actual_return):
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
            avg_latency_ms=np.mean(latencies) if latencies else 0,
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
        else:
            config = KVCacheConfig()

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
    title: str = 'KV-Cache Trading Strategy Backtest',
    save_path: str = 'backtest_results.png'
):
    """Plot backtest results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot.")
        return

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
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Results saved to {save_path}")


def prepare_test_data(symbols: List[str], limit: int = 2000) -> Dict[str, pd.DataFrame]:
    """Prepare test data with technical features."""
    test_data = {}

    for symbol in symbols:
        df = fetch_bybit_klines(symbol, limit=limit)
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

    return test_data


def main():
    """Run backtest example."""
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

    # Fetch data
    print("Fetching data...")
    try:
        test_data = prepare_test_data(symbols, limit=500)
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Using synthetic data instead...")

        # Create synthetic data
        test_data = {}
        for symbol in symbols:
            n = 500
            dates = pd.date_range(start='2024-01-01', periods=n, freq='h')
            prices = 100 * np.cumprod(1 + np.random.randn(n) * 0.01)

            df = pd.DataFrame({
                'timestamp': dates,
                'close': prices,
                'volume': np.random.rand(n) * 1000
            })

            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['log_return'].rolling(24).std()
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(24).mean()
            df['momentum'] = df['close'] / df['close'].shift(24) - 1
            df['rsi'] = 50 + np.random.randn(n) * 10

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

    # Try to plot results
    try:
        plot_backtest_results(result)
    except Exception as e:
        print(f"Could not plot results: {e}")

    return result


if __name__ == '__main__':
    main()
