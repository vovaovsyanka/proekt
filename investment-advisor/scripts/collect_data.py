"""
Скрипт сбора данных для бота-трейдера.
Собирает исторические данные OHLCV, макроэкономические показатели, новости и фундаментальные данные.

Пайплайн сбора данных:
1. Проверяем наличие готовых датасетов в data/raw/ и data/features/
2. Если файлы существуют - загружаем их и определяем тикеры
3. Актуализируем данные через MOEX API с момента последней записи
4. Сохраняем все данные в единую структуру:
   - data/raw/stocks_combined.csv - объединенные OHLCV данные по всем тикерам
   - data/features/macro_data.csv - макроэкономические показатели
   - data/features/news.csv - новости с тональностью

Использование:
    python scripts/collect_data.py --start-date 2020-01-01 --end-date 2024-12-31

Все данные сохраняются без генерации, только реальные значения из API.
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import logging
import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Set, Tuple

# Добавляем корень проекта в path для импортов
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import requests
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

# Начальный список тикеров (используется ТОЛЬКО при первой загрузке)
DEFAULT_TICKERS = [
    "SBER", "GAZP", "LKOH", "NVTK", "YNDX", "TCSG", "VTBR", "ROSN",
    "GMKN", "NLMK", "SNGS", "HYDR", "FEES", "TRNFP", "MTSS", "AFKS",
    "PIKK", "CHMF", "MAGN", "RTKM", "BSPB", "VKCO", "OZON", "SGZH"
]


class MOEXDataCollector:
    """
    Сборщик данных с Московской Биржи через ISS API.
    
    Источники:
    - MOEX ISS API для котировок
    - ЦБ РФ API для макроэкономики
    - RSS ленты для новостей
    
    Пайплайн:
    1. Загрузка готовых данных из data/raw/stocks_combined.csv (если существует)
    2. Определение тикеров из загруженных данных
    3. Актуализация через MOEX API с последней даты
    4. Сохранение в единый файл data/raw/stocks_combined.csv
    """
    
    def __init__(self, raw_data_dir: Optional[Path] = None, features_dir: Optional[Path] = None):
        self.raw_data_dir = raw_data_dir or settings.raw_data_dir
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir = features_dir or settings.features_dir
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
        # Список тикеров определяется из загруженных данных
        self.tickers: List[str] = []
    
    def load_existing_stocks_data(self) -> pd.DataFrame:
        """
        Загрузить существующие данные по акциям из data/raw/stocks_combined.csv
        
        Returns:
            DataFrame с колонками: Date, Ticker, open, high, low, close, volume
        """
        stocks_file = self.raw_data_dir / "stocks_combined.csv"
        
        if not stocks_file.exists():
            logger.info("Файл stocks_combined.csv не найден, начинаем сбор с нуля")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(stocks_file, parse_dates=['Date'])
            logger.info(f"Загружено {len(df)} записей из stocks_combined.csv")
            
            # Определяем тикеры из данных
            if 'Ticker' in df.columns:
                self.tickers = df['Ticker'].unique().tolist()
                logger.info(f"Найдены тикеры: {self.tickers}")
            
            return df
        except Exception as e:
            logger.warning(f"Ошибка загрузки stocks_combined.csv: {e}")
            return pd.DataFrame()
    
    def get_last_date_for_ticker(self, df: pd.DataFrame, ticker: str) -> Optional[str]:
        """Получить последнюю дату для тикера в данных."""
        if df.empty:
            return None
        
        ticker_df = df[df['Ticker'] == ticker]
        if ticker_df.empty:
            return None
        
        last_date = ticker_df['Date'].max()
        if pd.notna(last_date):
            return last_date.strftime('%Y-%m-%d')
        return None
    
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
            DataFrame с колонками: Date, open, high, low, close, volume, Ticker
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
            'begin': 'Date',
        }
        
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Добавляем колонку с тикером
        df['Ticker'] = ticker
        
        # Выбираем только нужные колонки
        required_cols = ['Date', 'Ticker', 'open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if available_cols:
            df = df[available_cols]
            logger.info(f"Загружено {len(df)} записей для {ticker}")
            return df
        else:
            logger.warning(f"Не найдены нужные колонки в данных для {ticker}")
            return pd.DataFrame()
    
    def get_macro_data_from_cbr(
        self,
        series_id: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Загружает макроэкономические данные из API Банка России.
        
        Args:
            series_id - идентификатор ряда (например, 'RU_KEY_RATE' для ключевой ставки)
        
        Returns:
            DataFrame с колонками date и value
        """
        # Используем правильный формат URL для API ЦБ РФ
        url = f'https://cbr.ru/statistics/data-service/api/v1/data/{series_id}'
        params = {
            'from': start_date,
            'to': end_date
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 404:
                logger.warning(f"Серия данных {series_id} не найдена в API ЦБ РФ (404)")
                return pd.DataFrame()
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
        except requests.exceptions.HTTPError as e:
            logger.warning(f"HTTP ошибка загрузки данных ЦБ РФ ({series_id}): {e.response.status_code}")
            return pd.DataFrame()
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
        - API Банка России (ключевая ставка, курсы валют, инфляция)
        
        Args:
            start_date: Дата начала
            end_date: Дата окончания
        
        Returns:
            DataFrame с макро-показателями (без генерации данных, только NaN если API недоступно)
        """
        logger.info("Загрузка макроэкономических данных...")
        
        # Создаем даты
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)
        
        macro_data = {'date': dates}
        
        # === Ключевая ставка ЦБ РФ ===
        key_rate_df = self.get_macro_data_from_cbr('RU_KEY_RATE', start_date, end_date)
        if not key_rate_df.empty:
            key_rate_df = key_rate_df.reindex(pd.date_range(start=start_date, end=end_date, freq='D'))
            key_rate_df['key_rate'] = key_rate_df['value'].ffill()
            macro_data['key_rate'] = key_rate_df.reindex(dates)['key_rate'].values
        else:
            logger.warning("Не удалось загрузить ключевую ставку из API ЦБ РФ")
            macro_data['key_rate'] = np.nan
        
        # === Курс USD/RUB ===
        usd_rub_df = self.get_macro_data_from_cbr('USD_RUB', start_date, end_date)
        if not usd_rub_df.empty:
            usd_rub_df = usd_rub_df.reindex(pd.date_range(start=start_date, end=end_date, freq='D'))
            usd_rub_df['usd_rub'] = usd_rub_df['value'].ffill()
            macro_data['usd_rub'] = usd_rub_df.reindex(dates)['usd_rub'].values
        else:
            logger.warning("Не удалось загрузить курс USD/RUB из API ЦБ РФ")
            macro_data['usd_rub'] = np.nan
        
        # === Инфляция (ИПЦ, % г/г) ===
        inflation_df = self.get_macro_data_from_cbr('CPI_IY', start_date, end_date)
        if not inflation_df.empty:
            inflation_df = inflation_df.reindex(pd.date_range(start=start_date, end=end_date, freq='D'))
            inflation_df['inflation'] = inflation_df['value'].ffill()
            macro_data['inflation'] = inflation_df.reindex(dates)['inflation'].values
        else:
            logger.warning("Не удалось загрузить инфляцию из API ЦБ РФ")
            macro_data['inflation'] = np.nan
        
        # Цены на нефть Brent - оставляем NaN (требуется другой источник)
        macro_data['brent'] = np.nan
        
        # Создаем DataFrame
        macro_df = pd.DataFrame(macro_data)
        macro_df.set_index('date', inplace=True)
        
        logger.info(f"Загружены макро-данные для {n_days} дней (NaN если API недоступно)")
        return macro_df
    
    def parse_news(self, limit: int = 100) -> pd.DataFrame:
        """
        Парсинг финансовых новостей через RSS ленты.
        
        Args:
            limit: Максимальное количество новостей из каждого источника
        
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
            
            logger.info(f"Итого загружено {len(df)} уникальных новостей")
            return df
            
        except ImportError:
            logger.warning("feedparser не установлен. Пропускаем парсинг новостей.")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Ошибка парсинга новостей: {e}")
            return pd.DataFrame()
    
    def analyze_sentiment(self, texts: List[str]) -> List[float]:
        """
        Анализ тональности текстов с помощью предобученной модели.
        
        Args:
            texts: Список текстов для анализа
        
        Returns:
            Список оценок тональности (от -1 до 1)
        """
        try:
            from transformers import pipeline
            
            # Загрузка модели для русского финансового рынка
            sentiment_pipeline = pipeline(
                "sentiment-analysis", 
                model="serguntsov/rubert-tiny2-russian-financial-sentiment",
                return_all_scores=False
            )
            
            results = []
            for text in texts:
                try:
                    result = sentiment_pipeline(text[:512])[0]  # Ограничиваем длину текста
                    # Преобразуем LABEL в числовое значение
                    score = result['score']
                    if result['label'] == 'NEGATIVE':
                        score = -score
                    results.append(round(score, 4))
                except Exception:
                    results.append(0.0)  # Нейтральная оценка при ошибке
            
            return results
            
        except ImportError:
            logger.warning("transformers не установлен. Пропускаем анализ тональности.")
            return [0.0] * len(texts)
        except Exception as e:
            logger.warning(f"Ошибка анализа тональности: {e}")
            return [0.0] * len(texts)
    
    def save_stocks_data(self, df: pd.DataFrame):
        """Сохранение данных по акциям в единый файл."""
        if df.empty:
            logger.warning("Пустой DataFrame, сохранение пропущено")
            return
        
        filepath = self.raw_data_dir / "stocks_combined.csv"
        df.to_csv(filepath, index=False)
        logger.info(f"Данные по акциям сохранены в {filepath} ({len(df)} записей)")
    
    def save_macro_data(self, df: pd.DataFrame):
        """Сохранение макро-данных."""
        if df.empty:
            logger.warning("Пустой DataFrame макро-данных")
            return
        
        filepath = self.features_dir / "macro_data.csv"
        df.to_csv(filepath)
        logger.info(f"Макро-данные сохранены в {filepath}")
    
    def save_news_data(self, df: pd.DataFrame):
        """Сохранение новостей."""
        if df.empty:
            logger.warning("Пустой DataFrame новостей")
            return
        
        filepath = self.features_dir / "news.csv"
        df.to_csv(filepath, index=False)
        logger.info(f"Новости сохранены в {filepath}")


def main():
    """Главная функция сбора данных."""
    parser = argparse.ArgumentParser(description='Сбор данных с MOEX')
    parser.add_argument(
        '--tickers', 
        type=str, 
        default=None,
        help='Список тикеров через запятую (по умолчанию все из существующих данных)'
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
        '--force-refresh', 
        action='store_true',
        help='Игнорировать существующие данные и загрузить заново'
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
    
    # Инициализация сборщика
    collector = MOEXDataCollector()
    
    # Шаг 1: Загружаем существующие данные (если есть и не force-refresh)
    logger.info("\n=== Проверка существующих данных ===")
    stocks_df = pd.DataFrame()
    
    if not args.force_refresh:
        stocks_df = collector.load_existing_stocks_data()
    
    # Определяем список тикеров
    if args.tickers:
        # Пользователь явно указал тикеры
        tickers = [t.strip().upper() for t in args.tickers.split(',')]
        logger.info(f"Используем тикеры из аргументов: {tickers}")
    elif collector.tickers:
        # Используем тикеры из загруженных данных
        tickers = collector.tickers.copy()
        logger.info(f"Используем тикеры из существующих данных: {tickers}")
    else:
        # Данные отсутствуют - используем дефолтный список для первичной загрузки
        # Это единственный случай использования дефолтных тикеров
        tickers = DEFAULT_TICKERS.copy()
        logger.info(f"Данные отсутствуют, используем начальный набор тикеров: {tickers}")
    
    logger.info(f"Период: {args.start_date} - {args.end_date}")
    logger.info(f"Всего тикеров: {len(tickers)}")
    
    # Шаг 2: Для каждого тикера проверяем последнюю дату и добираем данные через API
    logger.info("\n=== Сбор/актуализация данных по акциям (MOEX API) ===")
    
    all_tickers_data = []
    
    for i, ticker in enumerate(tickers, 1):
        logger.info(f"[{i}/{len(tickers)}] Обработка {ticker}...")
        
        # Получаем последнюю дату в существующих данных для этого тикера
        last_date = collector.get_last_date_for_ticker(stocks_df, ticker)
        
        if last_date and not args.force_refresh:
            # Есть данные - добираем с последней даты
            fetch_from = (pd.to_datetime(last_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            logger.info(f"  Последняя запись: {last_date}, добираем с {fetch_from}")
            
            if fetch_from <= args.end_date:
                new_data = collector.get_moex_candles(ticker, fetch_from, args.end_date)
                if not new_data.empty:
                    all_tickers_data.append(new_data)
                    logger.info(f"  Добавлено {len(new_data)} новых записей")
                else:
                    logger.info(f"  Новых данных нет")
                
                # Добавляем существующие данные для этого тикера
                existing_ticker_data = stocks_df[stocks_df['Ticker'] == ticker].copy()
                if not existing_ticker_data.empty:
                    all_tickers_data.append(existing_ticker_data)
            else:
                logger.info(f"  Данные актуальны")
                # Добавляем существующие данные
                existing_ticker_data = stocks_df[stocks_df['Ticker'] == ticker].copy()
                if not existing_ticker_data.empty:
                    all_tickers_data.append(existing_ticker_data)
        else:
            # Нет данных или force-refresh - загружаем весь период
            logger.info(f"  {'Принудительная перезагрузка' if args.force_refresh else 'Нет данных, загружаем весь период'}")
            full_data = collector.get_moex_candles(ticker, args.start_date, args.end_date)
            if not full_data.empty:
                all_tickers_data.append(full_data)
                logger.info(f"  Загружено {len(full_data)} записей")
    
    # Объединяем все данные
    if all_tickers_data:
        combined_df = pd.concat(all_tickers_data, ignore_index=True)
        # Удаляем дубликаты (по Date и Ticker)
        combined_df = combined_df.drop_duplicates(subset=['Date', 'Ticker'], keep='first')
        # Сортируем
        combined_df = combined_df.sort_values(['Ticker', 'Date'])
        logger.info(f"\nОбъединено данных: {len(combined_df)} записей")
    else:
        combined_df = pd.DataFrame()
        logger.warning("\nНет данных для сохранения")
    
    # Сохраняем объединенные данные
    if not combined_df.empty:
        collector.save_stocks_data(combined_df)
    
    # Сбор макро-данных
    logger.info("\n=== Сбор макроэкономических данных ===")
    macro_df = collector.get_macro_data(args.start_date, args.end_date)
    collector.save_macro_data(macro_df)
    
    # Парсинг новостей
    logger.info("\n=== Парсинг новостей ===")
    news_df = collector.parse_news(limit=100)
    
    # Анализ тональности новостей (если transformers установлен)
    if not news_df.empty and 'title' in news_df.columns:
        logger.info("Анализ тональности новостей...")
        titles = news_df['title'].fillna('').tolist()
        sentiments = collector.analyze_sentiment(titles)
        news_df['sentiment'] = sentiments
        logger.info(f"Тональность проанализирована для {len(news_df)} новостей")
    
    collector.save_news_data(news_df)
    
    # Статистика
    logger.info("\n" + "="*60)
    logger.info("ИТОГИ СБОРА ДАННЫХ")
    logger.info("="*60)
    
    if not combined_df.empty:
        unique_tickers = combined_df['Ticker'].unique()
        logger.info(f"Загружено тикеров: {len(unique_tickers)}")
        logger.info(f"Всего записей OHLCV: {len(combined_df)}")
        
        # Показываем статистику по первым 5 тикерам
        for ticker in list(unique_tickers)[:5]:
            ticker_df = combined_df[combined_df['Ticker'] == ticker]
            logger.info(f"\n{ticker}:")
            logger.info(f"  Период: {ticker_df['Date'].min()} - {ticker_df['Date'].max()}")
            logger.info(f"  Записей: {len(ticker_df)}")
            logger.info(f"  Цена: {ticker_df['close'].iloc[0]:.2f} → {ticker_df['close'].iloc[-1]:.2f}")
    else:
        logger.info("Данные по акциям не загружены")
    
    logger.info(f"\nМакро-данных: {len(macro_df)} записей")
    logger.info(f"Новостей: {len(news_df)} записей")
    
    logger.info("\n✅ Сбор данных завершен!")
    logger.info(f"Данные сохранены:")
    logger.info(f"  - {collector.raw_data_dir / 'stocks_combined.csv'}")
    logger.info(f"  - {collector.features_dir / 'macro_data.csv'}")
    logger.info(f"  - {collector.features_dir / 'news.csv'}")
    logger.info("\nДля обучения модели запустите: python scripts/train_model.py")


if __name__ == "__main__":
    main()
