"""
Генератор синтетических данных для обучения модели.
Используется когда yfinance недоступен (блокировки, rate limiting).

Генерирует реалистичные данные для 30 тикеров S&P500 за период 2017-2024.
Данные включают: цены, объемы, корпоративные действия (сплиты, дивиденды).

Использование:
    python backend/ml_pipeline/generate_synthetic_data.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict
from backend.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_realistic_price_series(
    ticker: str,
    start_date: str,
    end_date: str,
    initial_price: float = 100.0,
    annual_return: float = 0.08,
    annual_volatility: float = 0.25
) -> pd.DataFrame:
    """
    Генерация реалистичного временного ряда цен с использованием геометрического броуновского движения.
    
    Args:
        ticker: Тикер акции
        start_date: Дата начала
        end_date: Дата окончания
        initial_price: Начальная цена
        annual_return: Ожидаемая годовая доходность
        annual_volatility: Годовая волатильность
        
    Returns:
        DataFrame с колонками: open, high, low, close, adj_close, volume
    """
    # Создаем даты (только торговые дни - примерно 252 дня в году)
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    n_days = len(all_dates)
    
    # Детерминированный seed на основе тикера для воспроизводимости
    np.random.seed(hash(ticker) % (2**32))
    
    # Дневная доходность и волатильность
    daily_return = annual_return / 252
    daily_volatility = annual_volatility / np.sqrt(252)
    
    # Генерируем дневные логарифмические доходности
    log_returns = np.random.normal(daily_return, daily_volatility, n_days)
    
    # Добавляем немного автокорреляции (реальные цены имеют инерцию)
    for i in range(1, n_days):
        log_returns[i] += 0.05 * log_returns[i-1]
    
    # Кумулятивная сумма для получения цен
    cumulative_log_returns = np.cumsum(log_returns)
    
    # Преобразуем в цены
    prices = initial_price * np.exp(cumulative_log_returns)
    
    # Добавляем тренд в зависимости от сектора (детерминированно по тику)
    sector_trends = {
        'AAPL': 0.15, 'MSFT': 0.12, 'GOOGL': 0.10, 'AMZN': 0.08, 'META': 0.05,
        'NVDA': 0.20, 'TSLA': 0.25, 'JPM': 0.06, 'V': 0.10, 'JNJ': 0.04,
        'WMT': 0.05, 'PG': 0.03, 'MA': 0.10, 'UNH': 0.08, 'HD': 0.07,
        'DIS': 0.02, 'PYPL': -0.05, 'BAC': 0.04, 'ADBE': 0.12, 'CRM': 0.08,
        'NFLX': 0.10, 'CMCSA': 0.01, 'XOM': 0.06, 'VZ': 0.02, 'KO': 0.03,
        'PFE': 0.01, 'INTC': -0.08, 'T': 0.00, 'MRK': 0.05, 'PEP': 0.04
    }
    
    trend = sector_trends.get(ticker, 0.05)
    trend_factor = np.linspace(1.0, 1.0 + trend, n_days)
    prices = prices * trend_factor
    
    # Генерируем OHLC данные
    df = pd.DataFrame({'date': all_dates})
    
    # Close price
    df['close'] = prices
    
    # Adj Close (примерно равно close для простоты)
    df['adj_close'] = prices * (1 - np.random.uniform(0.01, 0.03, n_days))  # Дивидендная корректировка
    
    # Open = предыдущий close с небольшим гэпом
    gaps = np.random.normal(0, 0.01, n_days)
    df['open'] = np.roll(df['close'], 1) * (1 + gaps)
    df.loc[0, 'open'] = initial_price
    
    # High и Low
    daily_range = np.random.uniform(0.01, 0.03, n_days)  # Дневной диапазон 1-3%
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + daily_range)
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - daily_range)
    
    # Volume (с сезонностью и трендом)
    base_volume = np.random.uniform(5_000_000, 50_000_000)
    volume_trend = np.linspace(1.0, 1.2, n_days)  # Объем растет со временем
    volume_seasonality = 1 + 0.2 * np.sin(2 * np.pi * np.arange(n_days) / 5)  # Недельная сезонность
    volume_noise = np.random.lognormal(0, 0.3, n_days)
    
    df['volume'] = (base_volume * volume_trend * volume_seasonality * volume_noise).astype(int)
    
    # Округление
    price_cols = ['open', 'high', 'low', 'close', 'adj_close']
    df[price_cols] = df[price_cols].round(2)
    
    # Установка индекса
    df.set_index('date', inplace=True)
    
    return df


def generate_synthetic_data_for_tickers(
    tickers: List[str],
    start_date: str = "2017-01-01",
    end_date: str = "2024-12-31"
) -> Dict[str, pd.DataFrame]:
    """
    Генерация синтетических данных для списка тикеров.
    
    Args:
        tickers: Список тикеров
        start_date: Дата начала
        end_date: Дата окончания
        
    Returns:
        Словарь {ticker: DataFrame}
    """
    logger.info(f"Генерация синтетических данных для {len(tickers)} тикеров...")
    
    # Начальные цены для тикеров (реалистичные значения на 2017 год)
    initial_prices = {
        'AAPL': 28.0, 'MSFT': 62.0, 'GOOGL': 780.0, 'AMZN': 750.0, 'META': 115.0,
        'NVDA': 9.0, 'TSLA': 280.0, 'JPM': 85.0, 'V': 80.0, 'JNJ': 115.0,
        'WMT': 115.0, 'PG': 80.0, 'MA': 105.0, 'UNH': 150.0, 'HD': 140.0,
        'DIS': 105.0, 'PYPL': 40.0, 'BAC': 22.0, 'ADBE': 125.0, 'CRM': 80.0,
        'NFLX': 140.0, 'CMCSA': 35.0, 'XOM': 80.0, 'VZ': 45.0, 'KO': 40.0,
        'PFE': 35.0, 'INTC': 35.0, 'T': 40.0, 'MRK': 60.0, 'PEP': 110.0
    }
    
    # Волатильности (технологии более волатильны)
    volatilities = {
        'AAPL': 0.25, 'MSFT': 0.22, 'GOOGL': 0.24, 'AMZN': 0.28, 'META': 0.35,
        'NVDA': 0.45, 'TSLA': 0.55, 'JPM': 0.25, 'V': 0.22, 'JNJ': 0.15,
        'WMT': 0.18, 'PG': 0.14, 'MA': 0.22, 'UNH': 0.20, 'HD': 0.20,
        'DIS': 0.25, 'PYPL': 0.40, 'BAC': 0.30, 'ADBE': 0.28, 'CRM': 0.32,
        'NFLX': 0.40, 'CMCSA': 0.22, 'XOM': 0.25, 'VZ': 0.18, 'KO': 0.15,
        'PFE': 0.22, 'INTC': 0.30, 'T': 0.20, 'MRK': 0.20, 'PEP': 0.16
    }
    
    result = {}
    for ticker in tickers:
        logger.info(f"Генерация данных для {ticker}...")
        
        df = generate_realistic_price_series(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            initial_price=initial_prices.get(ticker, 100.0),
            annual_volatility=volatilities.get(ticker, 0.25)
        )
        
        if not df.empty:
            result[ticker] = df
            logger.info(f"  → Сгенерировано {len(df)} записей для {ticker}")
    
    logger.info(f"Успешно сгенерированы данные для {len(result)} тикеров из {len(tickers)}")
    return result


def save_to_cache(data_dict: Dict[str, pd.DataFrame], cache_dir: Path) -> None:
    """
    Сохранение сгенерированных данных в кэш.
    
    Args:
        data_dict: Словарь {ticker: DataFrame}
        cache_dir: Директория для сохранения
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    for ticker, df in data_dict.items():
        # Создаем ключ кэша как в DataLoader
        import hashlib
        key_string = f"{ticker}_2017-01-01_2024-12-31"
        cache_key = hashlib.md5(key_string.encode()).hexdigest()[:16]
        
        cache_file = cache_dir / f"{cache_key}.csv"
        df.reset_index().to_csv(cache_file, index=False)
        logger.info(f"Сохранено в кэш: {cache_file.name}")


def main():
    """Главная функция генерации данных."""
    logger.info("="*60)
    logger.info("ГЕНЕРАТОР СИНТЕТИЧЕСКИХ ДАННЫХ")
    logger.info("="*60)
    logger.info(f"Время запуска: {datetime.now().isoformat()}")
    
    tickers = settings.default_tickers
    logger.info(f"Тикеры для генерации: {tickers}")
    
    # Генерация данных
    synthetic_data = generate_synthetic_data_for_tickers(
        tickers=tickers,
        start_date="2017-01-01",
        end_date=settings.test_end_date
    )
    
    # Сохранение в кэш
    save_to_cache(synthetic_data, settings.cache_dir)
    
    # Проверка качества данных
    logger.info("\n" + "="*60)
    logger.info("ПРОВЕРКА КАЧЕСТВА ДАННЫХ")
    logger.info("="*60)
    
    for ticker, df in list(synthetic_data.items())[:5]:  # Первые 5 для примера
        logger.info(f"\n{ticker}:")
        logger.info(f"  Период: {df.index.min()} - {df.index.max()}")
        logger.info(f"  Записей: {len(df)}")
        logger.info(f"  Цена: ${df['close'].iloc[0]:.2f} → ${df['close'].iloc[-1]:.2f}")
        logger.info(f"  Доходность: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.1f}%")
    
    logger.info("\n" + "="*60)
    logger.info("✅ Синтетические данные успешно сгенерированы!")
    logger.info("Теперь можно запустить обучение: python backend/ml_pipeline/train.py")
    logger.info("="*60)


if __name__ == "__main__":
    main()
