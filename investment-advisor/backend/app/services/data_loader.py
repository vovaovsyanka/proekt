"""
Сервис загрузки данных: цены акций, макроэкономические показатели, новости.
Использует MOEX API для получения исторических данных по российским акциям и кэширование для оптимизации.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
from scripts.collect_data import MOEXDataCollector

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Загрузчик данных для ML модели.
    
    Отвечает за:
    - Загрузку исторических цен через MOEX API (для российских акций)
    - Кэширование данных для ускорения повторных запросов
    - Загрузку макроэкономических показателей
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
        self.moex_collector = MOEXDataCollector()
        
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
                # Читаем CSV и парсим дату как колонку, затем устанавливаем как индекс без TZ
                df = pd.read_csv(cache_file, parse_dates=['Date'])
                
                # Устанавливаем Date как индекс и гарантируем что он без timezone
                df.set_index('Date', inplace=True)
                
                # Если индекс имеет timezone - убираем её (конвертируем в naive datetime)
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    df.index = df.index.tz_convert('UTC').tz_localize(None)
                else:
                    # Уже naive datetime, просто убеждаемся что это datetime индекс
                    df.index = pd.to_datetime(df.index)
                
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
            # Сбрасываем индекс чтобы Date стал колонкой
            df_reset = df.reset_index()
            
            # Если индекс был timezone-aware, конвертируем в naive перед сохранением
            if 'Date' in df_reset.columns:
                # Конвертируем любую tz-aware дату в naive UTC
                if hasattr(df_reset['Date'].dt, 'tz') and df_reset['Date'].dt.tz is not None:
                    df_reset['Date'] = df_reset['Date'].dt.tz_convert('UTC').dt.tz_localize(None)
            
            # Сохраняем CSV
            df_reset.to_csv(cache_file, index=False)
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
        Приоритет: MOEX API для российских тикеров
        
        Args:
            ticker: Тикер акции (например, "SBER")
            start_date: Дата начала в формате YYYY-MM-DD
            end_date: Дата окончания в формате YYYY-MM-DD
            use_cache: Использовать ли кэш
            
        Returns:
            DataFrame с колонками: open, high, low, close, adj_close, volume
        """
        cache_key = self._get_cache_key(ticker, start_date, end_date)
        
        # Проверка кэша
        if use_cache:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Загрузка через MOEX API (приоритет для российских акций)
        logger.info(f"Загрузка данных для {ticker} с {start_date} по {end_date}")
        try:
            # Пробуем MOEX API
            df = self.moex_collector.download_ticker_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is None or df.empty:
                # Fallback на yfinance
                logger.warning(f"MOEX API не вернул данные для {ticker}, пробуем yfinance...")
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
            
            # Убираем timezone из индекса для совместимости
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
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
        Загрузка макроэкономических показателей РФ.
        
        Использует реальные исторические данные для ключевых показателей:
        - Ключевая ставка ЦБ РФ
        - Инфляция (ИПЦ, % г/г)
        - Курс USD/RUB
        - Цена нефти Brent
        
        Args:
            start_date: Дата начала
            end_date: Дата окончания
        
        Returns:
            DataFrame с макро-факторами: key_rate, inflation, usd_rub, brent
        """
        return self.moex_collector.get_macro_data(start_date, end_date)

    def get_news_sentiment_data(
        self,
        ticker: str,
        limit: int = 10
    ) -> List[Dict[str, str]]:
        """
        Получение новостей для анализа сентимента.
        Загружает новости из локального файла и фильтрует по тикеру.
        
        Args:
            ticker: Тикер компании
            limit: Количество новостей
            
        Returns:
            Список словарей с заголовками новостей
        """
        logger.info(f"Получение новостей для {ticker} (лимит: {limit})")
        
        # Путь к файлу с новостями
        news_file = Path(__file__).parent.parent.parent / "data" / "features" / "rbk_news.csv"
        
        if news_file.exists():
            try:
                df = pd.read_csv(news_file, parse_dates=['published'])
                
                # Берем последние новости
                news_list = []
                for _, row in df.head(limit).iterrows():
                    news_list.append({
                        'title': row.get('title', ''),
                        'summary': row.get('summary', ''),
                        'published_at': row.get('published', ''),
                        'link': row.get('link', '')
                    })
                
                logger.info(f"Найдено {len(news_list)} новостей")
                return news_list
                
            except Exception as e:
                logger.warning(f"Ошибка загрузки новостей: {e}")
        
        # Fallback: генерируем заглушки
        news_templates = [
            f"{ticker} reports strong quarterly earnings",
            f"Analysts upgrade {ticker} price target",
            f"{ticker} announces new product launch",
            f"Market volatility affects {ticker} stock price",
            f"{ticker} CEO makes strategic announcement"
        ]
        
        news_list = [{"title": t, "source": "Financial News"} for t in news_templates[:limit]]
        logger.info(f"Сгенерировано {len(news_list)} fallback новостей для {ticker}")
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
                # Используем MOEX API
                df = self.moex_collector.download_ticker_data(
                    ticker=ticker,
                    start_date=(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                    end_date=datetime.now().strftime("%Y-%m-%d")
                )
                
                if df is not None and not df.empty:
                    prices[ticker] = df['close'].iloc[-1]
                else:
                    # Fallback
                    stock = yf.Ticker(ticker)
                    data = stock.history(period="5d")
                    if not data.empty:
                        prices[ticker] = data['Close'].iloc[-1]
                    else:
                        prices[ticker] = 0.0
            except Exception as e:
                logger.warning(f"Не удалось получить цену для {ticker}: {e}")
                prices[ticker] = 0.0
        
        return prices
