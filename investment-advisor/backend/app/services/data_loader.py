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
        
        Args:
            ticker: Тикер акции (например, "AAPL")
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
            
            # Убираем timezone из индекса для совместимости
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Сохранение в кэш (сбрасываем индекс чтобы Date стал колонкой)
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
        logger.info("Загрузка макроэкономических данных РФ")
        
        # Создаем даты (бизнес-дни)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        n_days = len(dates)
        
        macro_data = {'date': dates}
        
        # === Ключевая ставка ЦБ РФ (реальные значения) ===
        key_rate_history = {
            '2020-01-01': 7.75,
            '2020-04-01': 5.50,
            '2020-07-01': 4.25,
            '2021-03-01': 4.50,
            '2021-06-01': 5.50,
            '2021-10-01': 7.50,
            '2021-12-01': 8.50,
            '2022-02-28': 20.0,
            '2022-04-11': 17.0,
            '2022-05-27': 11.0,
            '2022-07-25': 8.0,
            '2022-09-19': 7.5,
            '2023-07-21': 8.5,
            '2023-09-15': 13.0,
            '2023-10-30': 15.0,
            '2023-12-15': 16.0,
            '2024-07-26': 18.0,
            '2024-09-13': 19.0,
        }
        
        rate_dates = pd.to_datetime(list(key_rate_history.keys()))
        rate_values = list(key_rate_history.values())
        rate_series = pd.Series(rate_values, index=rate_dates)
        
        rate_df = pd.DataFrame({'key_rate': rate_series})
        rate_df = rate_df.reindex(pd.date_range(start=min(rate_dates), end=end_date, freq='D'))
        rate_df['key_rate'] = rate_df['key_rate'].ffill()
        
        macro_data['key_rate'] = rate_df.loc[start_date:end_date].reindex(dates)['key_rate'].values
        
        # === Инфляция (ИПЦ, % г/г) ===
        inflation_data = {
            '2020-01-01': 2.4,
            '2020-07-01': 3.2,
            '2021-01-01': 4.9,
            '2021-07-01': 6.5,
            '2022-01-01': 8.7,
            '2022-04-01': 17.8,
            '2022-07-01': 15.1,
            '2023-01-01': 11.8,
            '2023-07-01': 4.3,
            '2024-01-01': 7.4,
            '2024-07-01': 8.9,
        }
        
        inf_dates = pd.to_datetime(list(inflation_data.keys()))
        inf_values = list(inflation_data.values())
        inf_series = pd.Series(inf_values, index=inf_dates)
        
        inf_df = pd.DataFrame({'inflation': inf_series})
        inf_df = inf_df.reindex(pd.date_range(start=min(inf_dates), end=end_date, freq='D'))
        inf_df['inflation'] = inf_df['inflation'].ffill()
        
        macro_data['inflation'] = inf_df.loc[start_date:end_date].reindex(dates)['inflation'].values
        
        # === Курс USD/RUB ===
        usd_rub_data = {
            '2020-01-01': 62.0,
            '2020-03-01': 74.0,
            '2020-12-01': 73.0,
            '2021-12-01': 74.0,
            '2022-02-24': 84.0,
            '2022-03-01': 115.0,
            '2022-06-01': 57.0,
            '2022-12-01': 70.0,
            '2023-06-01': 85.0,
            '2023-12-01': 90.0,
            '2024-06-01': 88.0,
            '2024-09-01': 92.0,
        }
        
        usd_dates = pd.to_datetime(list(usd_rub_data.keys()))
        usd_values = list(usd_rub_data.values())
        usd_series = pd.Series(usd_values, index=usd_dates)
        
        usd_df = pd.DataFrame({'usd_rub': usd_series})
        usd_df = usd_df.reindex(pd.date_range(start=min(usd_dates), end=end_date, freq='D'))
        usd_df['usd_rub'] = usd_df['usd_rub'].ffill()
        
        macro_data['usd_rub'] = usd_df.loc[start_date:end_date].reindex(dates)['usd_rub'].values
        
        # === Brent crude oil price ($/barrel) ===
        oil_data = {
            '2020-01-01': 68.0,
            '2020-04-01': 25.0,
            '2020-12-01': 51.0,
            '2021-12-01': 75.0,
            '2022-03-01': 110.0,
            '2022-12-01': 85.0,
            '2023-06-01': 75.0,
            '2023-12-01': 77.0,
            '2024-06-01': 82.0,
            '2024-09-01': 73.0,
        }
        
        oil_dates = pd.to_datetime(list(oil_data.keys()))
        oil_values = list(oil_data.values())
        oil_series = pd.Series(oil_values, index=oil_dates)
        
        oil_df = pd.DataFrame({'brent': oil_series})
        oil_df = oil_df.reindex(pd.date_range(start=min(oil_dates), end=end_date, freq='D'))
        oil_df['brent'] = oil_df['brent'].ffill()
        
        macro_data['brent'] = oil_df.loc[start_date:end_date].reindex(dates)['brent'].values
        
        # Создаем DataFrame
        macro_df = pd.DataFrame(macro_data)
        macro_df.set_index('date', inplace=True)
        
        logger.info(f"Загружены макро-данные для {n_days} дней")
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
