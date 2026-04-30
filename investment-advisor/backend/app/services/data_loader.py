"""
Сервис загрузки данных: цены акций, макроэкономические показатели, новости.
Использует yfinance для получения исторических данных и кэширование для оптимизации.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json
import hashlib
from typing import List, Optional, Dict, Tuple
from functools import lru_cache

from backend.config import settings

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Загрузчик данных для ML модели.
    
    Отвечает за:
    - Загрузку исторических цен через yfinance
    - Кэширование данных для ускорения повторных запросов
    - Загрузку макроэкономических показателей (с заглушками при недоступности)
    - Получение новостей для анализа сентимента
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Инициализация загрузчика данных.
        
        Args:
            cache_dir: Директория для кэширования данных. По умолчанию из config.
        """
        self.cache_dir = cache_dir or settings.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_key(self, ticker: str, start_date: str, end_date: str) -> str:
        """
        Генерация уникального ключа для кэша на основе параметров запроса.
        Использует MD5 хеш для создания короткого имени файла.
        """
        key_string = f"{ticker}_{start_date}_{end_date}"
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        Загрузка данных из кэша.
        
        Args:
            cache_key: Ключ кэша
            
        Returns:
            DataFrame с данными или None если кэш не найден
        """
        cache_file = self.cache_dir / f"{cache_key}.csv"
        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file, parse_dates=['Date'], index_col='Date')
                logger.info(f"Данные загружены из кэша: {cache_key}")
                return df
            except Exception as e:
                logger.warning(f"Ошибка чтения кэша {cache_key}: {e}")
        return None
    
    def _save_to_cache(self, df: pd.DataFrame, cache_key: str) -> None:
        """
        Сохранение данных в кэш.
        
        Args:
            df: DataFrame для сохранения
            cache_key: Ключ кэша
        """
        cache_file = self.cache_dir / f"{cache_key}.csv"
        try:
            df.to_csv(cache_file)
            logger.info(f"Данные сохранены в кэш: {cache_key}")
        except Exception as e:
            logger.warning(f"Ошибка записи в кэш {cache_key}: {e}")
    
    def download_stock_data(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Загрузка исторических данных по акции.
        
        Args:
            ticker: Тикер акции (например, "AAPL")
            start_date: Дата начала в формате YYYY-MM-DD
            end_date: Дата окончания в формате YYYY-MM-DD
            use_cache: Использовать ли кэш
            
        Returns:
            DataFrame с колонками: Open, High, Low, Close, Adj Close, Volume
        """
        cache_key = self._get_cache_key(ticker, start_date, end_date)
        
        # Проверка кэша
        if use_cache:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Загрузка через yfinance
        logger.info(f"Загрузка данных для {ticker} с {start_date} по {end_date}")
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                logger.warning(f"Нет данных для тикера {ticker}")
                return pd.DataFrame()
            
            # Стандартизация имен колонок
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',
                'Volume': 'volume'
            })
            
            # Сохранение в кэш
            if use_cache:
                self._save_to_cache(df.reset_index(), cache_key)
            
            logger.info(f"Успешно загружено {len(df)} записей для {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Ошибка загрузки данных для {ticker}: {e}")
            return pd.DataFrame()
    
    def download_multiple_tickers(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Загрузка данных по нескольким тикерам.
        
        Args:
            tickers: Список тикеров
            start_date: Дата начала
            end_date: Дата окончания
            use_cache: Использовать ли кэш
            
        Returns:
            Словарь {ticker: DataFrame}
        """
        result = {}
        for ticker in tickers:
            df = self.download_stock_data(ticker, start_date, end_date, use_cache)
            if not df.empty:
                result[ticker] = df
        
        logger.info(f"Загружены данные для {len(result)} тикеров из {len(tickers)}")
        return result
    
    def get_macro_data(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Загрузка макроэкономических показателей.
        
        В учебной версии используется заглушка с реалистичными значениями.
        В продакшене можно подключить FRED API или другие источники.
        
        Args:
            start_date: Дата начала
            end_date: Дата окончания
            
        Returns:
            DataFrame с макро-факторами: inflation_rate, interest_rate, vix
        """
        logger.info("Генерация макроэкономических данных (заглушка)")
        
        # Создаем даты
        dates = pd.date_range(start=start_date, end=end_date, freq='B')  # бизнес-дни
        
        # Генерируем реалистичные значения с использованием случайного блуждания
        np.random.seed(42)  # Для воспроизводимости
        
        n_days = len(dates)
        
        # Инфляция (годовая, %)
        inflation = np.cumsum(np.random.normal(0, 0.1, n_days)) + 2.5
        inflation = np.clip(inflation, 0, 10)
        
        # Процентная ставка ФРС (%)
        interest_rate = np.cumsum(np.random.normal(0, 0.05, n_days)) + 2.0
        interest_rate = np.clip(interest_rate, 0, 10)
        
        # VIX (индекс волатильности)
        vix = np.cumsum(np.random.normal(0, 0.3, n_days)) + 20
        vix = np.clip(vix, 10, 80)
        
        macro_df = pd.DataFrame({
            'date': dates,
            'inflation_rate': inflation,
            'interest_rate': interest_rate,
            'vix': vix
        })
        macro_df.set_index('date', inplace=True)
        
        logger.info(f"Сгенерированы макро-данные для {n_days} дней")
        return macro_df
    
    def get_news_sentiment_data(
        self,
        ticker: str,
        limit: int = 10
    ) -> List[Dict[str, str]]:
        """
        Получение новостей для анализа сентимента.
        
        В учебной версии используются заглушки.
        В продакшене можно подключить NewsAPI, AlphaVantage или другие источники.
        
        Args:
            ticker: Тикер компании
            limit: Количество новостей
            
        Returns:
            Список словарей с заголовками новостей
        """
        logger.info(f"Получение новостей для {ticker} (лимит: {limit})")
        
        # Заглушка: генерируем реалистичные заголовки на основе тикера
        # В реальности здесь был бы вызов NewsAPI или парсинг RSS
        news_templates = [
            f"{ticker} reports strong quarterly earnings",
            f"Analysts upgrade {ticker} price target",
            f"{ticker} announces new product launch",
            f"Market volatility affects {ticker} stock price",
            f"{ticker} CEO makes strategic announcement",
            f"Industry trends impact {ticker} outlook",
            f"{ticker} faces regulatory challenges",
            f"Investor sentiment shifts on {ticker}",
            f"{ticker} expands into new markets",
            f"Technical analysis suggests {ticker} breakout"
        ]
        
        # Выбираем случайные новости
        np.random.seed(hash(ticker) % 2**32)  # Детерминированный выбор для тикера
        selected_indices = np.random.choice(
            len(news_templates), 
            size=min(limit, len(news_templates)), 
            replace=False
        )
        
        news_list = [
            {"title": news_templates[i], "source": "Financial News"}
            for i in selected_indices
        ]
        
        logger.info(f"Найдено {len(news_list)} новостей для {ticker}")
        return news_list
    
    def get_latest_prices(
        self,
        tickers: List[str]
    ) -> Dict[str, float]:
        """
        Получение текущих цен по тикерам.
        
        Args:
            tickers: Список тикеров
            
        Returns:
            Словарь {ticker: current_price}
        """
        prices = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                # Быстрая загрузка последней цены
                data = stock.history(period="1d")
                if not data.empty:
                    prices[ticker] = data['Close'].iloc[-1]
                else:
                    # Fallback: используем предыдущее закрытие
                    prices[ticker] = stock.info.get('previousClose', 0.0)
            except Exception as e:
                logger.warning(f"Не удалось получить цену для {ticker}: {e}")
                prices[ticker] = 0.0
        
        return prices
