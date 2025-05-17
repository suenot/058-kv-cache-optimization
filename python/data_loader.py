"""
Data Loading Utilities for KV-Cache Trading

This module provides utilities for fetching market data from Bybit
and preparing datasets for KV-cache benchmarking.
"""

import numpy as np
import pandas as pd
import requests
from typing import List, Dict, Iterator


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


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators commonly used in trading.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with added features
    """
    df = df.copy()

    # Log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Volatility (rolling std of returns)
    df['volatility'] = df['log_return'].rolling(24).std()

    # Volume ratio
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(24).mean()

    # Momentum
    df['momentum'] = df['close'] / df['close'].shift(24) - 1

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_upper'] = sma20 + 2 * std20
    df['bb_lower'] = sma20 - 2 * std20
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    return df


def create_feature_matrix(
    df: pd.DataFrame,
    feature_cols: List[str] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Create feature matrix for model input.

    Args:
        df: DataFrame with features
        feature_cols: List of column names to include
        normalize: Whether to normalize features

    Returns:
        Feature matrix as numpy array
    """
    if feature_cols is None:
        feature_cols = ['log_return', 'volatility', 'volume_ratio', 'momentum', 'rsi']

    features = df[feature_cols].values.astype(np.float32)

    if normalize:
        # Z-score normalization
        mean = np.nanmean(features, axis=0)
        std = np.nanstd(features, axis=0) + 1e-8
        features = (features - mean) / std

    # Replace any NaN with 0
    features = np.nan_to_num(features, 0)

    return features


if __name__ == '__main__':
    # Example usage
    print("Fetching BTCUSDT data...")
    df = fetch_bybit_klines('BTCUSDT', limit=100)
    print(f"Fetched {len(df)} candles")
    print(df.head())

    # Add features
    df = add_technical_features(df)
    print("\nWith features:")
    print(df.columns.tolist())

    # Create feature matrix
    features = create_feature_matrix(df.dropna())
    print(f"\nFeature matrix shape: {features.shape}")

    # Streaming generator example
    print("\nStreaming generator test:")
    stream = create_streaming_generator('BTCUSDT', lookback_days=7)
    for i, data in enumerate(stream):
        if i >= 5:
            break
        print(f"  {data['timestamp']}: close={data['close']:.2f}, return={data['log_return']:.4f}")
