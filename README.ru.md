# Глава 60: Оптимизация KV-Cache для Алгоритмической Торговли

В этой главе рассматривается **оптимизация KV-Cache (Key-Value Cache)** — критически важная техника для эффективного инференса в торговых системах на основе трансформеров. Мы применяем стратегии оптимизации KV-cache к прогнозированию финансовых данных в реальном времени, демонстрируя, как эффективное использование памяти позволяет принимать торговые решения с низкой задержкой при более длинных контекстных окнах.

<p align="center">
<img src="https://i.imgur.com/kVcache.png" width="70%">
</p>

## Содержание

1. [Введение в KV-Cache](#введение-в-kv-cache)
    * [Узкое место инференса](#узкое-место-инференса)
    * [Что такое KV-Cache?](#что-такое-kv-cache)
    * [Почему это важно для трейдинга](#почему-это-важно-для-трейдинга)
2. [Основы KV-Cache](#основы-kv-cache)
    * [Авторегрессивная генерация](#авторегрессивная-генерация)
    * [Проблема роста памяти](#проблема-роста-памяти)
    * [Структура кэша](#структура-кэша)
3. [Техники оптимизации](#техники-оптимизации)
    * [PagedAttention](#pagedattention)
    * [Квантизация KV-Cache](#квантизация-kv-cache)
    * [Селективное сохранение](#селективное-сохранение)
    * [Кэширование префиксов](#кэширование-префиксов)
4. [Применение в трейдинге](#применение-в-трейдинге)
    * [Прогнозирование цен в реальном времени](#прогнозирование-цен-в-реальном-времени)
    * [Потоковый анализ книги ордеров](#потоковый-анализ-книги-ордеров)
    * [Инференс для мультиактивного портфеля](#инференс-для-мультиактивного-портфеля)
5. [Практические примеры](#практические-примеры)
6. [Реализация на Python](#реализация-на-python)
7. [Реализация на Rust](#реализация-на-rust)
8. [Бенчмарки производительности](#бенчмарки-производительности)
9. [Лучшие практики](#лучшие-практики)
10. [Ресурсы](#ресурсы)

## Введение в KV-Cache

### Узкое место инференса

В продакшн торговых системах скорость инференса критична. В то время как обучение происходит офлайн, инференс должен выполняться в реальном времени — часто за миллисекунды. Модели-трансформеры сталкиваются с фундаментальной проблемой при авторегрессивной генерации: им нужно пересчитывать attention по всем предыдущим токенам для каждого нового предсказания.

```
Традиционный инференс трансформера:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   Токен 1: Вычисляем attention для [Токен 1]                                │
│   Токен 2: Вычисляем attention для [Токен 1, Токен 2]                       │
│   Токен 3: Вычисляем attention для [Токен 1, Токен 2, Токен 3]              │
│   ...                                                                        │
│   Токен N: Вычисляем attention для [Токен 1, Токен 2, ... Токен N]          │
│                                                                              │
│   Проблема: Избыточное вычисление Q, K, V для токенов 1 до N-1!             │
│   Каждый шаг пересчитывает всё с нуля.                                      │
│                                                                              │
│   Сложность: O(N²) на последовательность                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Что такое KV-Cache?

**KV-Cache** сохраняет тензоры Key и Value, вычисленные на предыдущих шагах инференса, избегая избыточных пересчётов:

```
Механизм KV-Cache:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   Шаг 1: Обрабатываем Токен 1                                               │
│          Вычисляем K₁, V₁ → Сохраняем в кэш                                 │
│          Выход: Предсказание следующего токена                              │
│                                                                              │
│   Шаг 2: Обрабатываем Токен 2                                               │
│          Загружаем K₁, V₁ из кэша (без пересчёта!)                          │
│          Вычисляем K₂, V₂ → Добавляем в кэш                                 │
│          Выход: Предсказание следующего токена                              │
│                                                                              │
│   Шаг N: Обрабатываем Токен N                                               │
│          Загружаем K₁...K_{N-1}, V₁...V_{N-1} из кэша                       │
│          Вычисляем только K_N, V_N → Добавляем в кэш                        │
│          Выход: Предсказание следующего токена                              │
│                                                                              │
│   Результат: O(N) вычислений вместо O(N²) на каждый новый токен!            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Почему это важно для трейдинга

| Сценарий | Без KV-Cache | С KV-Cache | Улучшение |
|----------|--------------|------------|-----------|
| Прогнозирование цен в реальном времени | 50мс задержка | 5мс задержка | 10x быстрее |
| Потоковая книга ордеров | Не успевает | В реальном времени | Делает возможным |
| Длинный контекст (1 год данных) | Не хватает памяти | Выполнимо | Открывает возможность |
| Пакетное обслуживание (100 запросов) | 20 запр/сек | 200 запр/сек | 10x пропускная способность |

Для торговых приложений:
- **Задержка важна**: Каждая миллисекунда имеет значение в высокочастотной торговле
- **Эффективность памяти**: Позволяет обрабатывать более длинную историю рынка
- **Пропускная способность**: Обслуживание большего количества запросов на предсказания
- **Снижение затрат**: Меньше GPU ресурсов для той же производительности

## Основы KV-Cache

### Авторегрессивная генерация

Модели-трансформеры генерируют предсказания авторегрессивно — каждый новый токен зависит от всех предыдущих:

```python
def autoregressive_inference_naive(model, initial_context):
    """
    Наивный авторегрессивный инференс (неэффективный).

    Для трейдинга: предсказание следующего движения цены
    на основе исторической последовательности цен.
    """
    sequence = initial_context.copy()

    for step in range(prediction_horizon):
        # Проблема: Пересчитывает K, V для ВСЕХ токенов каждый шаг
        output = model.forward(sequence)  # O(N²) каждый раз!
        next_prediction = output[-1]
        sequence.append(next_prediction)

    return sequence

def autoregressive_inference_with_cache(model, initial_context):
    """
    Эффективный инференс с KV-cache.
    """
    sequence = initial_context.copy()
    kv_cache = None

    for step in range(prediction_horizon):
        if kv_cache is None:
            # Первый шаг: вычисляем и кэшируем K, V для всех токенов
            output, kv_cache = model.forward(sequence, use_cache=True)
        else:
            # Последующие шаги: вычисляем K, V только для нового токена
            output, kv_cache = model.forward(
                [sequence[-1]],  # Только последний токен!
                past_kv_cache=kv_cache,
                use_cache=True
            )

        next_prediction = output[-1]
        sequence.append(next_prediction)

    return sequence
```

### Проблема роста памяти

Сложность KV-cache в том, что потребление памяти растёт линейно с длиной последовательности:

```
Использование памяти KV-Cache:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   Память на токен = 2 × num_layers × num_heads × head_dim × bytes_per_val   │
│                                                                              │
│   Пример: LLaMA-2 13B параметров                                            │
│   - 40 слоёв, 40 голов, 128 head_dim, FP16 (2 байта)                        │
│   - На токен: 2 × 40 × 40 × 128 × 2 = 819,200 байт ≈ 0.8 МБ                │
│                                                                              │
│   Для трейдинга с разной длиной контекста:                                  │
│   ─────────────────────────────────────────────────────────────────────────  │
│   Длина контекста       Память на последовательность   Годовые часовые данные│
│   ─────────────────────────────────────────────────────────────────────────  │
│   256 токенов           ~200 МБ                        ~10 дней почасовых   │
│   1,024 токена          ~800 МБ                        ~6 недель почасовых  │
│   4,096 токенов         ~3.2 ГБ                        ~6 месяцев почасовых │
│   8,760 токенов         ~7 ГБ                          1 год почасовых      │
│                                                                              │
│   Проблема: Память растёт линейно, ограничивая batch size и контекст!       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Структура кэша

```python
class KVCache:
    """
    Key-Value Cache для эффективного инференса трансформера.

    Структура: [batch_size, num_heads, seq_len, head_dim]

    Для торговых моделей:
    - batch_size: Количество разных активов или сценариев
    - num_heads: Головы attention (захватывают разные паттерны)
    - seq_len: Длина исторического контекста (растёт при инференсе)
    - head_dim: Размерность на голову attention
    """

    def __init__(self, num_layers, batch_size, num_heads, head_dim, dtype=torch.float16):
        self.num_layers = num_layers
        self.keys = [None] * num_layers
        self.values = [None] * num_layers

        # Предварительное выделение для известной максимальной длины (опциональная оптимизация)
        self.max_seq_len = None
        self.current_seq_len = 0

    def update(self, layer_idx, new_keys, new_values):
        """
        Добавление новых ключей и значений в кэш.

        Args:
            layer_idx: Какой слой трансформера
            new_keys: [batch, num_heads, new_tokens, head_dim]
            new_values: [batch, num_heads, new_tokens, head_dim]
        """
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = new_keys
            self.values[layer_idx] = new_values
        else:
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], new_keys], dim=2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], new_values], dim=2)

        self.current_seq_len = self.keys[layer_idx].shape[2]

    def get(self, layer_idx):
        """Получение кэшированных ключей и значений для слоя."""
        return self.keys[layer_idx], self.values[layer_idx]

    def memory_usage(self):
        """Расчёт общего использования памяти в байтах."""
        total = 0
        for k, v in zip(self.keys, self.values):
            if k is not None:
                total += k.numel() * k.element_size()
                total += v.numel() * v.element_size()
        return total
```

## Техники оптимизации

### PagedAttention

**PagedAttention** (представлен в vLLM) применяет концепции страничной памяти операционных систем к управлению KV-cache:

```
Концепция PagedAttention:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│   Традиционный KV-Cache (непрерывная память):                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ KV запроса 1 │   ПОТЕРЯНО   │ KV запроса 2 │   ПОТЕРЯНО   │ ...    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│   Проблема: Нужно выделять максимальную длину → 60-80% памяти теряется!     │
│                                                                              │
│   PagedAttention (страничная память):                                        │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐            │
│   │ Блок 1  │ │ Блок 2  │ │ Блок 3  │ │ Блок 4  │ │ Блок 5  │            │
│   │ Запр 1  │ │ Запр 1  │ │ Запр 2  │ │ Запр 1  │ │ Запр 2  │            │
│   └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘            │
│                                                                              │
│   Таблица блоков (связывает логические и физические блоки):                 │
│   Запрос 1: [Блок 1, Блок 2, Блок 4]                                        │
│   Запрос 2: [Блок 3, Блок 5]                                                │
│                                                                              │
│   Результат: Почти нулевая потеря памяти, динамическое выделение!           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Квантизация KV-Cache

Снижение использования памяти путём квантизации кэшированных значений:

```python
class QuantizedKVCache:
    """
    Квантизованный KV-Cache для эффективного по памяти инференса.

    Квантизация FP8 снижает память на 50% по сравнению с FP16 с минимальной потерей качества.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        quantization: str = 'fp8'  # Варианты: 'fp8', 'int8', 'int4'
    ):
        self.quantization = quantization
        self.num_layers = num_layers

        # Тип хранения в зависимости от квантизации
        if quantization == 'fp8':
            self.storage_dtype = torch.float8_e4m3fn
            self.scale_dtype = torch.float16
        elif quantization == 'int8':
            self.storage_dtype = torch.int8
            self.scale_dtype = torch.float16
        elif quantization == 'int4':
            self.storage_dtype = torch.int8  # Упаковка двух значений int4
            self.scale_dtype = torch.float16

    def quantize(self, tensor: torch.Tensor) -> tuple:
        """Квантизация тензора и возврат квантизованных значений + масштаба."""
        if self.quantization == 'fp8':
            scale = tensor.abs().max() / 448.0  # Максимальное значение FP8 E4M3
            quantized = (tensor / scale).to(self.storage_dtype)
            return quantized, scale

        elif self.quantization == 'int8':
            scale = tensor.abs().max() / 127.0
            quantized = torch.round(tensor / scale).to(torch.int8)
            return quantized, scale

    def dequantize(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Деквантизация значений для вычисления attention."""
        return quantized.to(torch.float16) * scale
```

### Селективное сохранение

Сохранение только наиболее важных KV-пар для ограничения роста памяти:

```python
class SelectiveKVCache:
    """
    Селективный KV-Cache с сохранением на основе важности.

    Для трейдинга: Сохраняет KV-пары для критических рыночных событий,
    отбрасывая менее релевантные исторические данные.
    """

    def __init__(
        self,
        num_layers: int,
        max_cache_size: int = 2048,
        retention_strategy: str = 'attention_score'
    ):
        self.num_layers = num_layers
        self.max_cache_size = max_cache_size
        self.retention_strategy = retention_strategy

    def compute_importance(
        self,
        attention_weights: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Вычисление оценок важности для каждой кэшированной позиции.

        Стратегии:
        - attention_score: На основе полученных весов attention
        - recency: Более новые = более важные
        - entropy: Позиции с высокой энтропией более информативны
        - hybrid: Комбинация вышеперечисленного
        """
        if self.retention_strategy == 'attention_score':
            # Сумма полученного attention от всех позиций запроса
            importance = attention_weights.sum(dim=-2).mean(dim=1)

        elif self.retention_strategy == 'recency':
            seq_len = attention_weights.shape[-1]
            importance = torch.arange(seq_len, device=attention_weights.device).float()
            importance = importance / seq_len

        return importance

    def evict_if_needed(self, layer_idx: int):
        """Вытеснение наименее важных записей при превышении максимального размера."""
        if self.keys[layer_idx] is None:
            return

        current_size = self.keys[layer_idx].shape[2]

        if current_size > self.max_cache_size:
            # Сохраняем top-k наиболее важных позиций
            importance = self.importance_scores[layer_idx]
            _, keep_indices = torch.topk(importance, self.max_cache_size, dim=-1)
            keep_indices = keep_indices.sort(dim=-1).values  # Сохраняем временной порядок

            # Собираем сохраняемые записи
            # ... (код сбора)
```

### Кэширование префиксов

Кэширование общих префиксов для избежания повторных вычислений:

```python
class PrefixCache:
    """
    Кэширование префиксов для общего контекста между запросами.

    Для трейдинга: Кэширование рыночного контекста, общего для нескольких
    предсказаний активов (например, макро-индикаторы, рыночный режим).
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.prefix_store = {}  # хэш -> (keys, values, length)

    def store_prefix(
        self,
        prefix_tokens: torch.Tensor,
        keys: list,
        values: list
    ):
        """Сохранение вычисленного KV cache для префикса."""
        prefix_hash = self.hash_prefix(prefix_tokens)
        self.prefix_store[prefix_hash] = {
            'keys': [k.clone() for k in keys],
            'values': [v.clone() for v in values],
            'length': prefix_tokens.shape[1]
        }

    def lookup_prefix(self, prefix_tokens: torch.Tensor) -> dict:
        """Поиск кэшированных KV значений префикса."""
        prefix_hash = self.hash_prefix(prefix_tokens)
        return self.prefix_store.get(prefix_hash)
```

## Применение в трейдинге

### Прогнозирование цен в реальном времени

```python
class RealTimePricePredictor:
    """
    Прогнозирование цен в реальном времени с оптимизированным KV-cache.

    Применение: Прогнозирование следующего движения цены на основе потоковых рыночных данных.
    """

    def __init__(
        self,
        model: nn.Module,
        kv_cache_type: str = 'paged',  # 'standard', 'paged', 'quantized', 'selective'
        max_context: int = 4096
    ):
        self.model = model
        self.max_context = max_context

        if kv_cache_type == 'paged':
            self.kv_cache = PagedKVCache(...)
        elif kv_cache_type == 'quantized':
            self.kv_cache = QuantizedKVCache(quantization='fp8')
        # ...

    def predict_stream(
        self,
        data_stream: Iterator[dict],
        symbol: str
    ) -> Iterator[dict]:
        """
        Потоковые предсказания для торговли в реальном времени.

        Args:
            data_stream: Итератор рыночных данных
            symbol: Торговый символ (например, 'BTCUSDT')

        Yields:
            Предсказания с метриками уверенности и задержки
        """
        for data_point in data_stream:
            start_time = time.time()

            features = self.extract_features(data_point)

            with torch.no_grad():
                output, self.kv_cache = self.model(
                    torch.tensor([[features]]),
                    past_kv_cache=self.kv_cache,
                    use_cache=True
                )

            latency = time.time() - start_time
            prediction = output[0, -1].item()

            yield {
                'timestamp': data_point['timestamp'],
                'symbol': symbol,
                'prediction': prediction,
                'direction': 'ВВЕРХ' if prediction > 0 else 'ВНИЗ',
                'latency_ms': latency * 1000,
                'cache_memory_mb': self.kv_cache.memory_usage() / (1024 * 1024)
            }
```

### Потоковый анализ книги ордеров

```python
class StreamingOrderBookAnalyzer:
    """
    Анализ обновлений книги ордеров с эффективным KV-кэшированием.

    Книги ордеров генерируют высокочастотные обновления (100-1000/сек),
    требуя очень эффективного инференса.
    """

    def __init__(
        self,
        model: nn.Module,
        num_levels: int = 20,
        update_buffer_size: int = 100
    ):
        self.model = model
        self.num_levels = num_levels

        # Используем квантизованный кэш для эффективности памяти
        self.kv_cache = QuantizedKVCache(
            num_layers=model.num_layers,
            num_heads=model.num_heads,
            head_dim=model.head_dim,
            quantization='int8'
        )
```

### Инференс для мультиактивного портфеля

```python
class MultiAssetPortfolioInference:
    """
    Эффективный инференс для мультиактивных портфелей.

    Использует кэширование префиксов для общего рыночного контекста между активами.
    """

    def __init__(
        self,
        model: nn.Module,
        assets: list,
        shared_context_length: int = 256
    ):
        self.model = model
        self.assets = assets
        self.shared_context_length = shared_context_length

        # Кэш префиксов для общего рыночного контекста
        self.prefix_cache = PrefixCache(num_layers=model.num_layers)

        # Отдельные KV-кэши для каждого актива
        self.asset_caches = {
            asset: KVCache(...)
            for asset in assets
        }

    def predict_all_assets(
        self,
        market_data: dict,
        asset_data: dict
    ) -> dict:
        """
        Генерация предсказаний для всех активов эффективно.

        Args:
            market_data: Общие рыночные данные
            asset_data: Данные специфичные для каждого актива

        Returns:
            Предсказания для каждого актива с весами аллокации
        """
        # Вычисляем общий контекст один раз
        shared_context = self.compute_shared_context(market_data)

        # Проверяем кэш префикса
        cached_prefix = self.prefix_cache.lookup_prefix(shared_context)

        if cached_prefix is None:
            # Вычисляем и кэшируем префикс
            _, prefix_kv = self.model(shared_context, use_cache=True)
            self.prefix_cache.store_prefix(shared_context, prefix_kv.keys, prefix_kv.values)
            cached_prefix = prefix_kv

        # Предсказываем каждый актив с общим префиксом
        predictions = {}
        for asset in self.assets:
            asset_features = self.extract_asset_features(asset_data.get(asset, {}))

            output, self.asset_caches[asset] = self.model(
                asset_features,
                past_kv_cache=cached_prefix,  # Переиспользуем общий контекст
                use_cache=True
            )

            predictions[asset] = {
                'return_prediction': output[0, -1, 0].item(),
            }

        return predictions
```

## Реализация на Python

```
python/
├── __init__.py
├── model.py                # KV-Cache трансформер
├── data_loader.py          # Загрузка данных Bybit
├── inference.py            # Оптимизированный движок инференса
├── predict.py              # Прогнозирование в реальном времени
├── strategy.py             # Торговая стратегия и бэктестинг
├── requirements.txt        # Зависимости
└── examples/
    ├── 01_kv_cache_basics.py
    ├── 02_inference_benchmark.py
    └── 03_strategy_comparison.py
```

### Быстрый старт (Python)

```bash
# Установка зависимостей
cd python
pip install -r requirements.txt

# Запуск бенчмарка инференса
python -c "
from model import KVCacheTrader, benchmark_kv_cache
model = KVCacheTrader(input_dim=5)
results = benchmark_kv_cache(model, context_length=1024)
print(f'Ускорение с KV-cache: {results[\"speedup\"]:.2f}x')
"

# Запуск бэктеста
python strategy.py
```

## Реализация на Rust

Смотрите [rust/](rust/) для production-ready реализации на Rust.

```
rust/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs
│   ├── cache/
│   │   ├── mod.rs
│   │   ├── standard.rs       # Базовый KV-cache
│   │   ├── paged.rs          # PagedAttention-стиль кэша
│   │   └── quantized.rs      # Квантизованный кэш
│   ├── model/
│   │   ├── mod.rs
│   │   ├── attention.rs      # Attention с кэшем
│   │   └── transformer.rs    # Полная модель
│   ├── data/
│   │   ├── mod.rs
│   │   └── bybit.rs          # Клиент Bybit
│   └── strategy/
│       ├── mod.rs
│       └── backtest.rs       # Бэктестинг
├── benches/
│   └── cache_benchmark.rs
└── examples/
    ├── inference.rs
    └── streaming.rs
```

## Бенчмарки производительности

### Экономия памяти KV-Cache

| Длина контекста | Без кэша | Стандартный кэш | Квантизованный (FP8) | Снижение |
|-----------------|----------|-----------------|----------------------|----------|
| 256 | Пересчёт | 50 МБ | 25 МБ | 50% |
| 1,024 | Пересчёт | 200 МБ | 100 МБ | 50% |
| 4,096 | Пересчёт | 800 МБ | 400 МБ | 50% |
| 8,192 | Пересчёт | 1.6 ГБ | 800 МБ | 50% |

### Сравнение задержки инференса

| Длина контекста | Без кэша | С кэшем | Ускорение |
|-----------------|----------|---------|-----------|
| 256 | 15 мс | 2 мс | 7.5x |
| 512 | 35 мс | 2 мс | 17.5x |
| 1,024 | 120 мс | 3 мс | 40x |
| 2,048 | 450 мс | 4 мс | 112x |
| 4,096 | 1,800 мс | 6 мс | 300x |

### Бенчмарки торговых приложений

| Сценарий | Задержка | Пропускная способность | Память |
|----------|----------|------------------------|--------|
| Потоковый один актив | 2-5 мс | 200-500 пред/с | 100 МБ |
| Мульти-актив (10 символов) | 10-20 мс | 50-100 пред/с | 500 МБ |
| Анализ книги ордеров | 1-3 мс | 300-1000 пред/с | 200 МБ |

## Лучшие практики

### Когда использовать KV-Cache

**Рекомендуемые сценарии:**
- Торговля в реальном времени с потоковыми данными
- Авторегрессивные многошаговые предсказания
- Длинные контекстные окна (>256 токенов)
- Пакетное обслуживание множества запросов

**Может не понадобиться:**
- Одиночные предсказания
- Очень короткие последовательности
- Обучение (используйте только при инференсе)

### Советы по управлению памятью

```python
# 1. Предварительное выделение для известной длины последовательности
cache = KVCache(
    num_layers=6,
    batch_size=1,
    num_heads=8,
    head_dim=64,
    max_seq_len=4096  # Предвыделение
)

# 2. Используйте квантизацию для ограниченных по памяти развёртываний
cache = QuantizedKVCache(
    quantization='fp8'  # 50% экономия памяти
)

# 3. Реализуйте скользящее окно для бесконечных потоков
if cache_length > max_length:
    cache.truncate(keep_last=max_length)
```

### Оптимизация задержки

```python
# 1. Держите модель и кэш на GPU
model = model.cuda()

# 2. Используйте torch.inference_mode() для минимальных накладных расходов
with torch.inference_mode():
    output, cache = model(x, past_kv_cache=cache, use_cache=True)

# 3. Группируйте несколько запросов когда возможно
# (амортизирует накладные расходы между запросами)
```

## Ресурсы

### Статьи

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — статья vLLM (2023)
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691) — Дополнительная оптимизация
- [MiniCache: KV Cache Compression across Layers](https://arxiv.org/abs/2405.14366) — Послойное сжатие (2024)
- [SnapKV: LLM Knows What You are Looking for Before Generation](https://arxiv.org/abs/2404.14469) — Селективное сохранение (2024)

### Реализации

- [vLLM](https://github.com/vllm-project/vllm) — Высокопроизводительное обслуживание LLM с PagedAttention
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — Оптимизированный инференс LLM от NVIDIA
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/main/en/kv_cache) — Документация по KV-cache

### Связанные главы

- [Глава 58: FlashAttention для трейдинга](../58_flash_attention_trading) — Эффективное по памяти вычисление attention
- [Глава 59: Grouped Query Attention](../59_grouped_query_attention) — Уменьшение размера KV-cache
- [Глава 50: Трансформеры с расширенной памятью](../50_memory_augmented_transformers) — Системы внешней памяти

---

## Уровень сложности

**Продвинутый**

Предварительные требования:
- Архитектура трансформера и self-attention
- Концепции авторегрессивной генерации
- Управление памятью GPU
- PyTorch или аналогичный фреймворк глубокого обучения
- Базовые знания торговых стратегий
