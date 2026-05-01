"""
Скрипт сбора данных с Московской Биржи (MOEX).
Собирает исторические данные OHLCV, макроэкономические показатели и новости.

Использование:
    python scripts/collect_data.py --tickers SBER,GAZP,LKOH --start-date 2020-01-01 --end-date 2024-12-31
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import logging
import json
import hashlib
import time

# Добавляем корень проекта в path для импортов
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import requests
from typing import List, Dict, Optional
from backend.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MOEXDataCollector:
    """
    Сборщик данных с Московской Биржи через ISS API.
    
    Источники:
    - MOEX ISS API для котировок
    - ЦБ РФ API для макроэкономики
    - RSS ленты для новостей
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or settings.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Список российских тикеров по умолчанию
        self.default_tickers = [
            "SBER",   # Сбербанк
            "GAZP",   # Газпром
            "LKOH",   # Лукойл
            "NVTK",   # Новатэк
            "YNDX",   # Яндекс
            "TCSG",   # Тинькофф
            "VTBR",   # ВТБ
            "ROSN",   # Роснефть
            "GMKN",   # Норникель
            "NLMK",   # НЛМК
            "SNGS",   # Сургутнефтегаз
            "HYDR",   # РусГидро
            "FEES",   # ФСК ЕЭС
            "TRNFP",  # Транснефть
            "MTSS",   # МТС
            "AFKS",   # АФК Система
            "PIKK",   # ПИК
            "CHMF",   # Северсталь
            "MAGN",   # Магнитка
            "RTKM",   # Ростелеком
            "BSPB",   # Банк Санкт-Петербург
            "VKCO",   # VK
            "OZON",   # Ozon
            "SGZH"    # Сахалинская энергия
        ]
    
    def _get_cache_key(self, ticker: str, start_date: str, end_date: str) -> str:
        """Генерация уникального ключа для кэша."""
        key_string = f"{ticker}_{start_date}_{end_date}"
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Загрузка данных из кэша."""
        cache_file = self.cache_dir / f"{cache_key}.csv"
        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file, parse_dates=['Date'])
                df.set_index('Date', inplace=True)
                logger.info(f"Данные загружены из кэша: {cache_key}")
                return df
            except Exception as e:
                logger.warning(f"Ошибка чтения кэша {cache_key}: {e}")
        return None
    
    def _save_to_cache(self, df: pd.DataFrame, cache_key: str) -> None:
        """Сохранение данных в кэш."""
        cache_file = self.cache_dir / f"{cache_key}.csv"
        try:
            df_reset = df.reset_index()
            if 'Date' in df_reset.columns:
                if hasattr(df_reset['Date'].dt, 'tz') and df_reset['Date'].dt.tz is not None:
                    df_reset['Date'] = df_reset['Date'].dt.tz_convert('UTC').dt.tz_localize(None)
            df_reset.to_csv(cache_file, index=False)
            logger.info(f"Данные сохранены в кэш: {cache_key}")
        except Exception as e:
            logger.warning(f"Ошибка записи в кэш {cache_key}: {e}")
    
    def get_moex_candles(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str,
        interval: int = 24,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Загружает исторические свечи (OHLCV) с MOEX ISS API.
        
        Args:
            ticker: Тикер акции (например, 'SBER')
            start_date: Дата начала в формате YYYY-MM-DD
            end_date: Дата окончания в формате YYYY-MM-DD
            interval: 24 (день), 60 (час), 10 (10 минут), 1 (1 минута)
            use_cache: Использовать ли кэш
        
        Returns:
            DataFrame с колонками: open, high, low, close, volume
        """
        cache_key = self._get_cache_key(ticker, start_date, end_date)
        
        if use_cache:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        url = f'https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json'
        params = {
            'from': start_date,
            'till': end_date,
            'interval': interval,
            'start': 0
        }
        
        all_data = []
        page = 0
        
        while True:
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                candles = data.get('candles', {}).get('data', [])
                if not candles:
                    break
                
                all_data.extend(candles)
                params['start'] += len(candles)
                page += 1
                
                # Лимит на количество страниц для защиты от бесконечного цикла
                if len(candles) < 500 or page > 100:
                    break
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Ошибка запроса для {ticker}: {e}")
                break
        
        if not all_data:
            logger.warning(f"Нет данных для тикера {ticker}")
            return pd.DataFrame()
        
        # Получаем названия колонок из ответа
        # MOEX API может возвращать columns как:
        # 1. Список словарей: [{'name': 'begin', 'id': 0}, ...]
        # 2. Список кортежей/списков: [['begin', 0], ['open', 1], ...]
        # 3. Простой список строк: ['open', 'close', 'high', ...] - текущий формат MOEX
        columns_info = data['candles']['columns']
        
        # Проверяем формат columns
        if isinstance(columns_info[0], dict):
            # Формат: [{'name': 'begin', 'id': 0}, ...]
            column_names = [col['name'] for col in columns_info]
        elif isinstance(columns_info[0], (list, tuple)):
            # Формат: [['begin', 0], ['open', 1], ...]
            column_names = [col[0] for col in columns_info]
        elif isinstance(columns_info[0], str):
            # Формат: ['open', 'close', 'high', 'low', 'value', 'volume', 'begin', 'end']
            column_names = columns_info
        else:
            logger.error(f"Неизвестный формат колонок: {columns_info}")
            return pd.DataFrame()
        
        # Создаем DataFrame
        df = pd.DataFrame(all_data, columns=column_names)
        
        # Переименовываем колонки в стандартный формат
        rename_map = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'begin': 'Date',
            'end': 'end_time'
        }
        
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        # Выбираем только нужные колонки
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if available_cols:
            df = df[available_cols]
            
            # Сохраняем в кэш
            if use_cache:
                self._save_to_cache(df, cache_key)
            
            logger.info(f"Загружено {len(df)} записей для {ticker}")
            return df
        else:
            logger.warning(f"Не найдены нужные колонки в данных для {ticker}")
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
        total = len(tickers)
        
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"[{i}/{total}] Загрузка {ticker}...")
            df = self.get_moex_candles(ticker, start_date, end_date, use_cache=use_cache)
            if not df.empty:
                result[ticker] = df
            
            # Небольшая пауза между запросами для соблюдения rate limits
            if i < total:
                time.sleep(0.2)
        
        logger.info(f"Успешно загружены данные для {len(result)}/{total} тикеров")
        return result
    
    def get_macro_data(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Загрузка макроэкономических показателей РФ.
        
        Источники:
        - API Банка России (ключевая ставка)
        - Росстат (инфляция, ВВП)
        
        Args:
            start_date: Дата начала
            end_date: Дата окончания
        
        Returns:
            DataFrame с макро-показателями
        """
        logger.info("Загрузка макроэкономических данных...")
        
        # Создаем даты
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        n_days = len(dates)
        
        macro_data = {'date': dates}
        
        # === Ключевая ставка ЦБ РФ ===
        # Реальные значения ставки (примерные, для демонстрации)
        # В продакшене нужно парсить с cbr.ru
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
        
        # Интерполяция ставки по датам
        rate_dates = pd.to_datetime(list(key_rate_history.keys()))
        rate_values = list(key_rate_history.values())
        
        # Создаем Series с ставкой
        rate_series = pd.Series(rate_values, index=rate_dates)
        
        # Приводим к нашим датам с forward fill
        rate_df = pd.DataFrame({'key_rate': rate_series})
        rate_df = rate_df.reindex(pd.date_range(start=min(rate_dates), end=end_date, freq='D'))
        rate_df['key_rate'] = rate_df['key_rate'].ffill()
        
        # Берем только бизнес-дни
        macro_data['key_rate'] = rate_df.loc[start_date:end_date].reindex(dates)['key_rate'].values
        
        # === Инфляция (ИПЦ, % г/г) ===
        # Примерные значения
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
        # Можно брать с cbr.ru или moex
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
    
    def parse_rbk_news(self, limit: int = 50) -> pd.DataFrame:
        """
        Парсинг финансовых новостей через альтернативные источники.
        
        Args:
            limit: Максимальное количество новостей
        
        Returns:
            DataFrame с новостями
        """
        try:
            import feedparser
            import requests
            
            # Используем работающие RSS ленты
            rss_urls = [
                'https://smart-lab.ru/rss/',
                'https://www.rbc.ru/quote/news/exportNewsXml/?symbol=',
            ]
            
            news_list = []
            for url in rss_urls:
                try:
                    # Скачиваем через requests с таймаутом
                    response = requests.get(url, timeout=5)
                    response.raise_for_status()
                    feed = feedparser.parse(response.content)
                    
                    for entry in feed.entries[:limit]:
                        published = None
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            published = datetime(*entry.published_parsed[:6])
                        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                            published = datetime(*entry.updated_parsed[:6])
                        
                        news_list.append({
                            'title': entry.title if hasattr(entry, 'title') else '',
                            'published': published,
                            'link': entry.link if hasattr(entry, 'link') else '',
                            'summary': entry.summary if hasattr(entry, 'summary') else ''
                        })
                except Exception as e:
                    logger.warning(f"Не удалось спарсить {url}: {e}")
                    continue
            
            df = pd.DataFrame(news_list)
            logger.info(f"Спарсено {len(df)} новостей")
            return df
            
        except ImportError:
            logger.warning("feedparser не установлен. Пропускаем парсинг новостей.")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Ошибка парсинга новостей: {e}")
            return pd.DataFrame()


def main():
    """Главная функция сбора данных."""
    parser = argparse.ArgumentParser(description='Сбор данных с MOEX')
    parser.add_argument(
        '--tickers', 
        type=str, 
        default=None,
        help='Список тикеров через запятую (по умолчанию все из конфига)'
    )
    parser.add_argument(
        '--start-date', 
        type=str, 
        default='2020-01-01',
        help='Дата начала в формате YYYY-MM-DD'
    )
    parser.add_argument(
        '--end-date', 
        type=str, 
        default='2024-12-31',
        help='Дата окончания в формате YYYY-MM-DD'
    )
    parser.add_argument(
        '--no-cache', 
        action='store_true',
        help='Не использовать кэш'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Директория для сохранения данных'
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("СБОР ДАННЫХ С МОСКОВСКОЙ БИРЖИ")
    logger.info("="*60)
    logger.info(f"Время запуска: {datetime.now().isoformat()}")
    
    # Определяем тикеры
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',')]
    else:
        collector = MOEXDataCollector()
        tickers = collector.default_tickers
    
    logger.info(f"Тикеры для сбора: {tickers}")
    logger.info(f"Период: {args.start_date} - {args.end_date}")
    
    # Инициализация сборщика
    collector = MOEXDataCollector()
    
    # Сбор данных по акциям
    logger.info("\n=== Сбор данных по акциям ===")
    price_data = collector.download_multiple_tickers(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        use_cache=not args.no_cache
    )
    
    # Сбор макро-данных
    logger.info("\n=== Сбор макроэкономических данных ===")
    macro_df = collector.get_macro_data(args.start_date, args.end_date)
    
    # Сохранение макро-данных
    output_dir = Path(args.output_dir) if args.output_dir else settings.features_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    macro_path = output_dir / "macro_data.csv"
    macro_df.to_csv(macro_path)
    logger.info(f"Макро-данные сохранены в {macro_path}")
    
    # Парсинг новостей
    logger.info("\n=== Парсинг новостей ===")
    news_df = collector.parse_rbk_news(limit=100)
    if not news_df.empty:
        news_path = output_dir / "rbk_news.csv"
        news_df.to_csv(news_path, index=False)
        logger.info(f"Новости сохранены в {news_path}")
    
    # Статистика
    logger.info("\n" + "="*60)
    logger.info("ИТОГИ СБОРА ДАННЫХ")
    logger.info("="*60)
    logger.info(f"Загружено тикеров: {len(price_data)}/{len(tickers)}")
    logger.info(f"Макро-данных: {len(macro_df)} записей")
    logger.info(f"Новостей: {len(news_df)} записей")
    
    for ticker, df in list(price_data.items())[:5]:
        logger.info(f"\n{ticker}:")
        logger.info(f"  Период: {df.index.min()} - {df.index.max()}")
        logger.info(f"  Записей: {len(df)}")
        logger.info(f"  Цена: {df['close'].iloc[0]:.2f} → {df['close'].iloc[-1]:.2f}")
    
    logger.info("\n✅ Сбор данных завершен!")
    logger.info("Для обучения модели запустите: python scripts/train_model.py")
    

if __name__ == "__main__":
    main()
