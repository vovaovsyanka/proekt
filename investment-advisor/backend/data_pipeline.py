#!/usr/bin/env python3
"""
Единый пайплайн сбора и обработки данных для инвестиционного советника.

Источники данных:
- OHLCV: Kaggle (olegshpagin/russia-stocks-prices-ohlcv) + MOEX API для актуализации
- Новости: HuggingFace (Kasymkhan/RussianFinancialNews)
- Фундаментальные: HuggingFace (irlspbru/RFSD)
- Макроэкономика: Kaggle (demirtry/russian-investment-activity)

Использование:
    python data_pipeline.py --start-date 2020-01-01 --end-date 2024-12-31
    python data_pipeline.py --tickers SBER GAZP LKOH
"""
import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import pandas as pd
import requests
import warnings
warnings.filterwarnings('ignore')

from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Единый пайплайн для сбора, обработки и объединения всех данных.
    
    Этапы:
    1. Загрузка OHLCV из локальных файлов (Kaggle) + дозагрузка через MOEX API
    2. Загрузка фундаментальных данных (RFSD)
    3. Обработка новостей и расчет сентимента
    4. Загрузка макроэкономических показателей
    5. Объединение всех данных в единый формат для ML
    """
    
    def __init__(self):
        self.raw_dir = settings.raw_data_dir
        self.features_dir = settings.features_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)

    def run_pipeline(self, tickers: List[str], start_date: str, end_date: str):
        logger.info(f"Запуск пайплайна с {start_date} по {end_date} для {len(tickers)} тикеров")

        ohlcv_df = self.get_ohlcv_data(tickers, start_date, end_date)
        fundamental_df = self.get_fundamental_data(tickers)
        ohlcv_fund_df = self.merge_ohlcv_fundamentals(ohlcv_df, fundamental_df)
        self.save_parquet(ohlcv_fund_df, "ohlcv_fundamentals", start_date, end_date)

        news_df = self.get_news_sentiment(start_date, end_date)
        self.save_parquet(news_df, "news_sentiment", start_date, end_date)

        macro_df = self.get_macro_data(start_date, end_date)
        self.save_parquet(macro_df, "macro_indicators", start_date, end_date)

        logger.info("Пайплайн завершён успешно")
        return ohlcv_fund_df, news_df, macro_df

    # ----------------------------------------------------------------
    #  OHLCV
    # ----------------------------------------------------------------
    def get_ohlcv_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        logger.info("Сбор OHLCV данных...")
        all_dfs = []

        # Соответствие оригинальных имён колонок (с угловыми скобками) и стандартных
        rename_map = {
            '<date>': 'date', '<datetime>': 'date', 'date': 'date',
            '<open>': 'open', 'open': 'open',
            '<high>': 'high', 'high': 'high',
            '<low>': 'low', 'low': 'low',
            '<close>': 'close', 'close': 'close',
            '<vol>': 'volume', 'volume': 'volume', 'vol': 'volume',
            '<ticker>': 'ticker', 'ticker': 'ticker'
        }

        # Файлы OHLCV из Kaggle (вида TICKER_D1.csv)
        kaggle_files = list(self.raw_dir.glob("*_D1.csv"))
        # API-файлы, если остались с предыдущих запусков
        api_files = list(self.raw_dir.glob("moex_api_*.csv"))

        # Сначала загружаем Kaggle
        for file in kaggle_files:
            ticker = file.stem.replace("_D1", "").upper()
            if ticker not in tickers:
                continue
            try:
                df = pd.read_csv(file)
                df.columns = [c.strip().lower() for c in df.columns]

                date_col = None
                for c in df.columns:
                    if c in ['<date>', 'date', 'datetime', '<datetime>']:
                        date_col = c
                        break
                if not date_col:
                    first_col = df.columns[0]
                    try:
                        pd.to_datetime(df[first_col])
                        date_col = first_col
                    except:
                        logger.warning(f"Не найден столбец даты в {file.name}, пропускаем")
                        continue

                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
                df = df.rename(columns=rename_map)
                df['ticker'] = ticker
                need_cols = ['open', 'high', 'low', 'close', 'volume', 'ticker']
                for c in need_cols:
                    if c not in df.columns:
                        df[c] = None
                df = df[need_cols]
                all_dfs.append(df)
                logger.info(f"  -> загружен {file.name}: {len(df)} строк")
            except Exception as e:
                logger.warning(f"Ошибка загрузки {file.name}: {e}")

        # Загружаем moex_api_* файлы
        for file in api_files:
            try:
                df = pd.read_csv(file, index_col=0, parse_dates=True)
                ticker_col = df['ticker'].iloc[0] if 'ticker' in df.columns else file.stem.replace("moex_api_", "").upper()
                if ticker_col not in tickers:
                    continue
                df = df[['open', 'high', 'low', 'close', 'volume', 'ticker']]
                all_dfs.append(df)
                logger.info(f"  -> загружен {file.name}: {len(df)} строк")
            except Exception as e:
                logger.warning(f"Ошибка загрузки {file.name}: {e}")

        if not all_dfs:
            logger.warning("Локальных OHLCV нет. Полная загрузка через MOEX API (может занять время)...")
            for ticker in tickers:
                try:
                    df = self._fetch_moex_candles(ticker, start_date, end_date, max_pages=50)
                    if not df.empty:
                        all_dfs.append(df)
                except Exception as e:
                    logger.error(f"Не удалось загрузить {ticker}: {e}")
        else:
            combined = pd.concat(all_dfs).sort_index()
            last_date = combined.index.max()
            if last_date < pd.Timestamp(end_date):
                api_start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
                logger.info(f"Дозагрузка OHLCV с {api_start} по {end_date} через MOEX API")
                needed_tickers = list(set(combined['ticker'].unique()))
                for ticker in needed_tickers:
                    try:
                        df_new = self._fetch_moex_candles(ticker, api_start, end_date, max_pages=10)
                        if not df_new.empty:
                            all_dfs.append(df_new)
                    except Exception as e:
                        logger.error(f"Ошибка дозагрузки {ticker}: {e}")

        if not all_dfs:
            raise RuntimeError("Не удалось получить ни одного OHLCV датафрейма")

        result = pd.concat(all_dfs).sort_index()
        result = result[~result.index.duplicated(keep='first')]
        logger.info(f"Итоговый OHLCV объём: {len(result)} записей")
        return result

    def _fetch_moex_candles(self, ticker: str, start: str, end: str, interval=24, max_pages=50) -> pd.DataFrame:
        """Загружает свечи с MOEX ISS API."""
        url = f'https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json'
        params = {'from': start, 'till': end, 'interval': interval, 'start': 0}
        all_data = []
        columns = []
        
        for page in range(max_pages):
            try:
                resp = requests.get(url, params=params, timeout=10)
                if resp.status_code != 200:
                    break
                data = resp.json()
                if not columns:
                    columns = data['candles']['columns']
                candles = data['candles']['data']
                if not candles:
                    break
                all_data.extend(candles)
                params['start'] += len(candles)
                time.sleep(0.1)
                if len(candles) < 500:
                    break
            except requests.exceptions.Timeout:
                logger.warning(f"Таймаут для {ticker}")
                break
            except Exception as e:
                logger.error(f"Ошибка API для {ticker}: {e}")
                break
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data, columns=columns)
        df['begin'] = pd.to_datetime(df['begin'])
        df = df.set_index('begin').rename(columns={
            'open': 'open', 'high': 'high', 'low': 'low',
            'close': 'close', 'volume': 'volume'
        })
        df['ticker'] = ticker
        return df[['open', 'high', 'low', 'close', 'volume', 'ticker']]

    # ----------------------------------------------------------------
    #  Новости и сентимент
    # ----------------------------------------------------------------
    def get_news_sentiment(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Загружает новости и рассчитывает агрегированный сентимент по дням."""
        logger.info("Сбор новостного сентимента...")
        local_news = self.raw_dir / "Kasymkhan_RussianFinancialNews.parquet"
        
        if not local_news.exists():
            logger.warning(f"Файл {local_news} не найден.")
            return pd.DataFrame()

        news_df = pd.read_parquet(local_news)
        date_col = next((c for c in ['published', 'date', 'pub_date'] if c in news_df.columns), None)
        
        if not date_col:
            logger.error("В новостях не найдена колонка с датой")
            return pd.DataFrame()

        news_df[date_col] = pd.to_datetime(news_df[date_col], format='mixed')
        news_df = news_df[(news_df[date_col] >= start_date) & (news_df[date_col] <= end_date)].copy()

        if news_df.empty:
            logger.warning(f"Нет новостей в диапазоне {start_date} – {end_date}")
            return pd.DataFrame()

        # Расчет сентимента если отсутствует
        if 'sentiment_score' not in news_df.columns:
            news_df['sentiment_score'] = self._calculate_sentiment(news_df)

        # Агрегация по дням
        daily = news_df.groupby(news_df[date_col].dt.date).agg(
            daily_sentiment_mean=('sentiment_score', 'mean'),
            daily_news_count=('sentiment_score', 'count'),
            daily_sentiment_std=('sentiment_score', 'std')
        ).reset_index()
        daily['date'] = pd.to_datetime(daily['date'])
        logger.info(f"Агрегировано за {len(daily)} дней")
        return daily

    def _calculate_sentiment(self, news_df: pd.DataFrame) -> pd.Series:
        """Рассчитывает сентимент для новостей."""
        try:
            from transformers import pipeline
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="blanchefort/rubert-base-cased-sentiment",
                truncation=True, max_length=512, device=-1
            )
            text_col = 'title' if 'title' in news_df.columns else 'text'
            texts = news_df[text_col].fillna('').tolist()
            scores = []
            
            for i in range(0, len(texts), 32):
                batch = texts[i:i+32]
                results = sentiment_pipeline(batch)
                for res in results:
                    label = res['label'].lower()
                    score = res['score']
                    if label == 'positive':
                        scores.append(score)
                    elif label == 'negative':
                        scores.append(-score)
                    else:
                        scores.append(0.0)
            return pd.Series(scores, index=news_df.index)
        except Exception as e:
            logger.error(f"Ошибка анализа тональности: {e}")
            return pd.Series(0.0, index=news_df.index)

    # ----------------------------------------------------------------
    #  Макроэкономика
    # ----------------------------------------------------------------
    def get_macro_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Загружает макроэкономические данные из локального CSV."""
        macro_csv = self.raw_dir / "russian_investment.csv"
        if not macro_csv.exists():
            logger.warning("Макроэкономический файл не найден")
            return pd.DataFrame()

        logger.info(f"Чтение макро из {macro_csv}")
        try:
            df = pd.read_csv(macro_csv)
            first_col = df.columns[0]
            
            # Пытаемся распарсить дату разными способами
            for date_format in [None, '%Y-%m-%d', '%d.%m.%Y']:
                try:
                    dates = pd.to_datetime(df[first_col], format=date_format)
                    break
                except:
                    continue
            else:
                # Если не получилось, пробуем добавить день/месяц
                try:
                    dates = pd.to_datetime(df[first_col].astype(str) + '-01-01')
                except:
                    logger.error("Не удалось определить дату в макро CSV")
                    return pd.DataFrame()
            
            df['date'] = dates
            df = df.drop(columns=[first_col]).set_index('date')
            
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                logger.error("В макро CSV нет числовых данных")
                return pd.DataFrame()
            
            logger.info(f"Макро: {df.shape} – показатели: {list(df.columns)}")
            return df[numeric_cols]
        except Exception as e:
            logger.error(f"Ошибка чтения макро: {e}")
            return pd.DataFrame()

    # ----------------------------------------------------------------
    #  Фундаментальные (RFSD)
    # ----------------------------------------------------------------
    def get_fundamental_data(self, tickers: List[str]) -> pd.DataFrame:
        rfsd_path = self.raw_dir / "rfsd_2023.parquet"
        if rfsd_path.exists():
            logger.info(f"Чтение фундаментальных данных из {rfsd_path}")
            df = pd.read_parquet(rfsd_path)
            if 'ticker' in df.columns:
                df['ticker'] = df['ticker'].str.upper()
                df = df[df['ticker'].isin(tickers)]
            return df
        else:
            logger.warning("RFSD файл не найден. Фундаментальные данные не будут добавлены.")
            return pd.DataFrame()

    # ----------------------------------------------------------------
    #  Объединение OHLCV с фундаментальными
    # ----------------------------------------------------------------
    def merge_ohlcv_fundamentals(self, ohlcv: pd.DataFrame, fund: pd.DataFrame) -> pd.DataFrame:
        if ohlcv.empty or fund.empty:
            return ohlcv
        ohlcv = ohlcv.reset_index().rename(columns={'index': 'date'})
        date_col = None
        for col in ['date', 'financial_year_end', 'report_date']:
            if col in fund.columns:
                date_col = col
                break
        if not date_col:
            logger.warning("Не найдена колонка с датой в фундаментальных данных")
            return ohlcv.set_index('date')
        fund = fund.copy()
        fund['date_parsed'] = pd.to_datetime(fund[date_col])
        merged = pd.merge_asof(
            ohlcv.sort_values('date'),
            fund.sort_values('date_parsed'),
            left_on='date', right_on='date_parsed', by='ticker',
            direction='backward', allow_exact_matches=True
        )
        return merged.set_index('date')

    # ----------------------------------------------------------------
    #  Сохранение
    # ----------------------------------------------------------------
    def save_parquet(self, df: pd.DataFrame, prefix: str, start: str, end: str):
        if df.empty:
            logger.warning(f"Нет данных для сохранения ({prefix})")
            return
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        fname = f"{prefix}_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.parquet"
        path = self.features_dir / fname
        df.to_parquet(path)
        logger.info(f"Сохранено: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сбор и подготовка данных")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--tickers", nargs="+", default=None)
    args = parser.parse_args()

    start_date = args.start_date or (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    tickers = args.tickers if args.tickers else settings.load_tickers()

    pipeline = DataPipeline()
    pipeline.run_pipeline(tickers, start_date, end_date)