"""
Скрипт сбора данных для бота-трейдера.
Собирает исторические данные OHLCV, макроэкономические показатели, новости и фундаментальные данные.

Приоритет источников:
1. Готовые датасеты (Kaggle, HuggingFace, GitHub) - основной источник исторических данных
2. MOEX ISS API - для актуализации и дополнения
3. API ЦБ РФ - для макроэкономических показателей
4. RSS ленты - для новостей

Использование:
    python scripts/collect_data.py --tickers SBER,GAZP,LKOH --start-date 2020-01-01 --end-date 2024-12-31

Все данные сохраняются в data/raw/
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import logging
import json
import time
import glob
import xml.etree.ElementTree as ET

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


# Рабочие RSS ленты финансовых новостей
NEWS_SOURCES = [
    {
        'name': 'Smart-Lab',
        'url': 'https://smart-lab.ru/rss/',
        'type': 'rss'
    },
    {
        'name': 'RBC Finance',
        'url': 'https://www.rbc.ru/quote/news/index.xml',
        'type': 'rss'
    },
    {
        'name': 'Investing.com Russia',
        'url': 'https://ru.investing.com/rss/news.rss',
        'type': 'rss'
    }
]

# Датасеты для загрузки
DATASETS = {
    'kaggle_stocks': {
        'name': 'Russia Stocks Prices OHLCV',
        'url': 'https://www.kaggle.com/datasets/olegshpagin/russia-stocks-prices-ohlcv',
        'description': 'Данные по крупнейшим российским тикерам OHLCV',
        'type': 'stocks'
    },
    'moex_dataset': {
        'name': 'MOEX Dataset',
        'url': 'https://github.com/foykes/moex-dataset',
        'description': 'Данные по тикерам MOEX',
        'type': 'stocks'
    },
    'russian_financial_news': {
        'name': 'Russian Financial News',
        'url': 'https://huggingface.co/datasets/Kasymkhan/RussianFinancialNews',
        'description': 'Датасет с финансовыми новостями',
        'type': 'news'
    },
    'financial_sentiment': {
        'name': 'Financial News Sentiment',
        'url': 'https://github.com/WebOfRussia/financial-news-sentiment',
        'description': 'Набор данных для анализа тональности финансовых новостей',
        'type': 'sentiment'
    },
    'rfsd': {
        'name': 'RFSD - Russian Financial Statements Database',
        'url': 'https://github.com/irlcode/RFSD',
        'description': 'Российская база данных финансовой отчетности',
        'type': 'fundamentals'
    },
    'kaggle_macro': {
        'name': 'Russian Investment Activity',
        'url': 'https://www.kaggle.com/datasets/demirtry/russian-investment-activity',
        'description': 'Макроэкономические показатели: ставка ЦБ, инфляция, ВВП и др.',
        'type': 'macro'
    }
}


class MOEXDataCollector:
    """
    Сборщик данных с Московской Биржи через ISS API.
    
    Источники:
    - MOEX ISS API для котировок
    - ЦБ РФ API для макроэкономики
    - RSS ленты для новостей
    """
    
    def __init__(self, raw_data_dir: Optional[Path] = None):
        self.raw_data_dir = raw_data_dir or settings.raw_data_dir
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
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
    
    def save_dataframe(self, df: pd.DataFrame, filename: str) -> Path:
        """Сохранение DataFrame в CSV файл."""
        filepath = self.raw_data_dir / filename
        df.to_csv(filepath, index=True)
        logger.info(f"Данные сохранены в {filepath}")
        return filepath
    
    def get_moex_candles(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str,
        interval: int = 24
    ) -> pd.DataFrame:
        """
        Загружает исторические свечи (OHLCV) с MOEX ISS API.
        
        Args:
            ticker: Тикер акции (например, 'SBER')
            start_date: Дата начала в формате YYYY-MM-DD
            end_date: Дата окончания в формате YYYY-MM-DD
            interval: 24 (день), 60 (час), 10 (10 минут), 1 (1 минута)
        
        Returns:
            DataFrame с колонками: open, high, low, close, volume
        """
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
        columns_info = data['candles']['columns']
        
        # Проверяем формат columns
        if isinstance(columns_info[0], dict):
            column_names = [col['name'] for col in columns_info]
        elif isinstance(columns_info[0], (list, tuple)):
            column_names = [col[0] for col in columns_info]
        elif isinstance(columns_info[0], str):
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
            logger.info(f"Загружено {len(df)} записей для {ticker}")
            return df
        else:
            logger.warning(f"Не найдены нужные колонки в данных для {ticker}")
            return pd.DataFrame()
    
    def download_ticker_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Обёртка над get_moex_candles для совместимости с data_loader.
        
        Args:
            ticker: Тикер акции (например, 'SBER')
            start_date: Дата начала в формате YYYY-MM-DD
            end_date: Дата окончания в формате YYYY-MM-DD
            
        Returns:
            DataFrame с колонками: open, high, low, close, volume
        """
        return self.get_moex_candles(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval=24  # дневные свечи
        )
    
    def download_multiple_tickers(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Загрузка данных по нескольким тикерам.
        
        Args:
            tickers: Список тикеров
            start_date: Дата начала
            end_date: Дата окончания
        
        Returns:
            Словарь {ticker: DataFrame}
        """
        result = {}
        total = len(tickers)
        
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"[{i}/{total}] Загрузка {ticker}...")
            df = self.get_moex_candles(ticker, start_date, end_date)
            if not df.empty:
                result[ticker] = df
            
            # Небольшая пауза между запросами для соблюдения rate limits
            if i < total:
                time.sleep(0.2)
        
        logger.info(f"Успешно загружены данные для {len(result)}/{total} тикеров")
        return result
    
    def get_macro_data_from_cbr(
        self,
        series_id: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Загружает макроэкономические данные из API Банка России.
        
        Args:
            series_id - идентификатор ряда (например, 'RU_CPI_M' для инфляции)
        
        Returns:
            DataFrame с колонками date и value
        """
        url = f'https://www.cbr.ru/statistics/data-service/api/v1/data/{series_id}'
        params = {
            'from': start_date,
            'to': end_date
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            data = []
            for obs in root.findall('.//Obs'):
                data.append({
                    'date': obs.get('TIME_PERIOD'),
                    'value': obs.get('OBS_VALUE')
                })
            df = pd.DataFrame(data)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            return df
        except Exception as e:
            logger.warning(f"Ошибка загрузки данных ЦБ РФ ({series_id}): {e}")
            return pd.DataFrame()
    
    def get_macro_data(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Загрузка макроэкономических показателей РФ из реальных источников.
        
        Источники:
        - API Банка России (ключевая ставка, курсы валют)
        
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
        # Загружаем реальные данные с API ЦБ РФ
        key_rate_df = self.get_macro_data_from_cbr('RU_KEY_RATE', start_date, end_date)
        if not key_rate_df.empty:
            key_rate_df = key_rate_df.reindex(pd.date_range(start=start_date, end=end_date, freq='D'))
            key_rate_df['key_rate'] = key_rate_df['value'].ffill()
            macro_data['key_rate'] = key_rate_df.reindex(dates)['key_rate'].values
        else:
            logger.warning("Не удалось загрузить ключевую ставку, используем значения по умолчанию")
            macro_data['key_rate'] = np.nan
        
        # === Курс USD/RUB ===
        usd_rub_df = self.get_macro_data_from_cbr('RU_USDRUSD', start_date, end_date)
        if not usd_rub_df.empty:
            usd_rub_df = usd_rub_df.reindex(pd.date_range(start=start_date, end=end_date, freq='D'))
            usd_rub_df['usd_rub'] = usd_rub_df['value'].ffill()
            macro_data['usd_rub'] = usd_rub_df.reindex(dates)['usd_rub'].values
        else:
            logger.warning("Не удалось загрузить курс USD/RUB")
            macro_data['usd_rub'] = np.nan
        
        # === Инфляция (ИПЦ, % г/г) ===
        inflation_df = self.get_macro_data_from_cbr('RU_CPI_M', start_date, end_date)
        if not inflation_df.empty:
            inflation_df = inflation_df.reindex(pd.date_range(start=start_date, end=end_date, freq='D'))
            inflation_df['inflation'] = inflation_df['value'].ffill()
            macro_data['inflation'] = inflation_df.reindex(dates)['inflation'].values
        else:
            logger.warning("Не удалось загрузить инфляцию")
            macro_data['inflation'] = np.nan
        
        # === Brent crude oil price ===
        # API ЦБ может не иметь прямых данных по нефти, оставляем NaN или можно добавить другой источник
        macro_data['brent'] = np.nan
        
        # Создаем DataFrame
        macro_df = pd.DataFrame(macro_data)
        macro_df.set_index('date', inplace=True)
        
        logger.info(f"Загружены макро-данные для {n_days} дней")
        return macro_df
    
    def parse_rbk_news(self, limit: int = 100) -> pd.DataFrame:
        """
        Парсинг финансовых новостей через RSS ленты.
        
        Args:
            limit: Максимальное количество новостей
        
        Returns:
            DataFrame с новостями
        """
        try:
            import feedparser
            
            news_list = []
            for source in NEWS_SOURCES:
                try:
                    logger.info(f"Парсинг новостей из {source['name']} ({source['url']})...")
                    feed = feedparser.parse(source['url'])
                    
                    if not feed.entries:
                        logger.warning(f"Нет записей в ленте {source['name']}")
                        continue
                    
                    for entry in feed.entries[:limit]:
                        published = None
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            try:
                                published = datetime(*entry.published_parsed[:6])
                            except (TypeError, ValueError):
                                published = datetime.now()
                        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                            try:
                                published = datetime(*entry.updated_parsed[:6])
                            except (TypeError, ValueError):
                                published = datetime.now()
                        
                        title = entry.title if hasattr(entry, 'title') else ''
                        link = entry.link if hasattr(entry, 'link') else ''
                        summary = entry.summary if hasattr(entry, 'summary') else ''
                        
                        # Пропускаем пустые новости
                        if not title and not summary:
                            continue
                        
                        news_list.append({
                            'title': title,
                            'published': published,
                            'link': link,
                            'summary': summary,
                            'source': source['name']
                        })
                    
                    logger.info(f"Загружено {min(limit, len(feed.entries))} новостей из {source['name']}")
                    
                except Exception as e:
                    logger.warning(f"Не удалось спарсить {source['name']}: {e}")
                    continue
            
            if not news_list:
                logger.warning("Новости не загружены ни из одного источника")
                return pd.DataFrame()
            
            df = pd.DataFrame(news_list)
            
            # Удаляем дубликаты по заголовкам
            df = df.drop_duplicates(subset=['title'], keep='first')
            
            # Сортируем по дате
            if 'published' in df.columns:
                df = df.sort_values('published', ascending=False)
            
            # Ограничиваем количество
            df = df.head(limit * len(NEWS_SOURCES))
            
            logger.info(f"Итого загружено {len(df)} уникальных новостей")
            return df
            
        except ImportError:
            logger.warning("feedparser не установлен. Пропускаем парсинг новостей.")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Ошибка парсинга новостей: {e}")
            return pd.DataFrame()
    
    def load_kaggle_dataset(self, dataset_path: str) -> Optional[pd.DataFrame]:
        """
        Загрузка готового датасета с Kaggle.
        
        Поддерживаемые датасеты:
        - olegshpagin/russia-stocks-prices-ohlcv
        
        Args:
            dataset_path: Путь к файлу датасета
        
        Returns:
            DataFrame с данными или None если файл не найден
        """
        path = Path(dataset_path)
        
        if not path.exists():
            logger.warning(f"Файл датасета не найден: {dataset_path}")
            return None
        
        try:
            if path.suffix == '.csv':
                df = pd.read_csv(path, parse_dates=['Date'])
                if 'Date' in df.columns:
                    df.set_index('Date', inplace=True)
                logger.info(f"Загружен Kaggle датасет: {dataset_path}, {len(df)} записей")
                return df
            else:
                logger.warning(f"Неподдерживаемый формат файла: {path.suffix}")
                return None
        except Exception as e:
            logger.error(f"Ошибка загрузки датасета: {e}")
            return None
    
    def merge_datasets(self, primary_df: pd.DataFrame, secondary_df: pd.DataFrame) -> pd.DataFrame:
        """
        Объединение двух датасетов с удалением дубликатов.
        
        Приоритет отдается primary_df. Данные из secondary_df добавляются
        только для тех дат, которых нет в primary_df.
        
        Args:
            primary_df: Основной датасет (приоритетный)
            secondary_df: Дополнительный датасет
        
        Returns:
            Объединенный DataFrame
        """
        if primary_df.empty:
            return secondary_df.copy()
        if secondary_df.empty:
            return primary_df.copy()
        
        # Определяем даты которые есть только во вторичном датасете
        primary_dates = set(primary_df.index)
        secondary_dates = set(secondary_df.index)
        new_dates = secondary_dates - primary_dates
        
        if not new_dates:
            logger.info("Дублирование данных: все даты уже присутствуют в основном датасете")
            return primary_df.copy()
        
        # Фильтруем вторичный датасет
        secondary_new = secondary_df[secondary_df.index.isin(new_dates)]
        
        # Объединяем
        merged = pd.concat([primary_df, secondary_new])
        merged = merged.sort_index()
        
        logger.info(f"Объединено датасетов: {len(primary_df)} + {len(secondary_new)} = {len(merged)} записей")
        return merged


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
    parser.add_argument(
        '--kaggle-dataset',
        type=str,
        default=None,
        help='Путь к Kaggle датасету для приоритетной загрузки'
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
    
    # Попытка загрузки готового датасета (приоритет №1)
    price_data = {}
    if args.kaggle_dataset:
        logger.info("\n=== Загрузка готового Kaggle датасета ===")
        kaggle_df = collector.load_kaggle_dataset(args.kaggle_dataset)
        if kaggle_df is not None:
            # Разделяем по тикерам если есть колонка 'Ticker'
            if 'Ticker' in kaggle_df.columns or 'ticker' in kaggle_df.columns:
                ticker_col = 'Ticker' if 'Ticker' in kaggle_df.columns else 'ticker'
                for ticker in tickers:
                    ticker_df = kaggle_df[kaggle_df[ticker_col] == ticker].copy()
                    if not ticker_df.empty:
                        if 'Date' in ticker_df.columns:
                            ticker_df.set_index('Date', inplace=True)
                        price_data[ticker] = ticker_df
                        logger.info(f"Загружен {ticker}: {len(ticker_df)} записей из Kaggle")
            else:
                # Если тикеров нет, считаем что это данные одного тикера
                price_data[tickers[0]] = kaggle_df
                logger.info(f"Загружены данные для {tickers[0]}: {len(kaggle_df)} записей из Kaggle")
    
    # Сбор данных по акциям из MOEX API (приоритет №2 - дополнение)
    logger.info("\n=== Сбор данных по акциям (MOEX API) ===")
    moex_data = collector.download_multiple_tickers(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Объединяем датасеты с удалением дубликатов
    for ticker in tickers:
        if ticker in price_data and ticker in moex_data:
            price_data[ticker] = collector.merge_datasets(price_data[ticker], moex_data[ticker])
        elif ticker in moex_data and ticker not in price_data:
            price_data[ticker] = moex_data[ticker]
    
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
