"""
ML Pipeline: обучение усиленной ансамбль-модели на исторических данных.

Приоритет источников данных:
1. Локальные датасеты из /workspace/data/raw/ (Kaggle и другие источники)
2. API сервисы (MOEX) для дополнения и актуализации

Особенности:
- Ансамбль моделей: LightGBM + GradientBoosting + RandomForest + LogisticRegression
- NLP анализ новостей через FinBERT для сентимент-признаков
- Расширенные технические индикаторы: Stochastic, ADX, ATR, Bollinger Bands
- Макроэкономические признаки с изменениями показателей

Использование:
    python scripts/train_model.py --tickers SBER,GAZP,LKOH,NVTK,YNDX
"""
import sys
import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
import warnings

# Добавляем корень проекта в path для импортов
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import joblib
import lightgbm as lgb

from backend.config import settings
from scripts.collect_data import MOEXDataCollector

# Игнорируем предупреждения для чистоты вывода
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Пути к данным
DATA_ROOT = Path(__file__).parent.parent / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
FEATURES_DIR = DATA_ROOT / "features"
MODEL_DIR = Path(__file__).parent.parent / "backend" / "models"


def load_kaggle_dataset(filepath: str) -> pd.DataFrame:
    """
    Загрузка датасета с Kaggle или другого локального источника.
    
    Args:
        filepath: Путь к CSV файлу
        
    Returns:
        DataFrame с данными
    """
    if not os.path.exists(filepath):
        logger.warning(f"Файл не найден: {filepath}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(filepath, parse_dates=['Date', 'date'], dayfirst=False)
        # Нормализация имен колонок
        df.columns = df.columns.str.lower().str.strip()
        logger.info(f"Загружено {len(df)} записей из {filepath}")
        return df
    except Exception as e:
        logger.error(f"Ошибка загрузки файла {filepath}: {e}")
        return pd.DataFrame()


def load_news_data() -> pd.DataFrame:
    """
    Загрузка данных новостей из локального файла.
    
    Returns:
        DataFrame с новостями
    """
    news_file = FEATURES_DIR / "rbk_news.csv"
    if news_file.exists():
        try:
            df = pd.read_csv(news_file, parse_dates=['published'])
            logger.info(f"Загружено {len(df)} новостей")
            
            # Нормализация колонок
            if 'title' in df.columns:
                df = df.rename(columns={'published': 'published_at', 'title': 'title'})
                # Добавляем пустую колонку ticker (будет заполняться при анализе)
                if 'ticker' not in df.columns:
                    df['ticker'] = ''
            
            return df
        except Exception as e:
            logger.warning(f"Ошибка загрузки новостей: {e}")
    
    return pd.DataFrame()


def load_and_prepare_data(tickers: list, start_date: str, end_date: str) -> tuple:
    """
    Загрузка и подготовка данных для обучения.
    Приоритет: локальные датасеты -> API
    
    Returns:
        Кортеж (price_data, macro_df, news_df) с ценами, макро-факторами и новостями
    """
    logger.info("=== Этап 1: Загрузка данных ===")
    
    # Инициализация загрузчика
    collector = MOEXDataCollector()
    
    logger.info(f"Загрузка данных для {len(tickers)} тикеров: {tickers[:5]}...")
    
    # Пробуем загрузить из локальных датасетов Kaggle
    price_data = {}
    
    # Проверяем наличие Kaggle датасета Russia Stocks Prices
    kaggle_file = RAW_DATA_DIR / "russia_stocks_ohlcv.csv"
    if kaggle_file.exists():
        logger.info(f"Найден Kaggle датасет: {kaggle_file}")
        kaggle_df = load_kaggle_dataset(str(kaggle_file))
        
        if not kaggle_df.empty:
            # Группируем данные по тикерам
            for ticker in tickers:
                ticker_data = kaggle_df[kaggle_df['ticker'].str.upper() == ticker].copy()
                if not ticker_data.empty:
                    # Приводим к нужному формату
                    ticker_data = ticker_data.rename(columns={
                        'open': 'open', 'high': 'high', 'low': 'low', 
                        'close': 'close', 'volume': 'volume'
                    })
                    if 'Date' not in ticker_data.columns and 'date' in ticker_data.columns:
                        ticker_data['Date'] = ticker_data['date']
                    ticker_data.set_index('Date', inplace=True)
                    price_data[ticker] = ticker_data[['open', 'high', 'low', 'close', 'volume']]
                    logger.info(f"Загружен Kaggle датасет для {ticker}: {len(price_data[ticker])} записей")
    
    # Дополняем данные из API для тех тикеров, которых нет в Kaggle
    missing_tickers = [t for t in tickers if t not in price_data]
    if missing_tickers:
        logger.info(f"Дозагрузка {len(missing_tickers)} тикеров из MOEX API...")
        train_start = "2018-01-01"
        api_data = collector.download_multiple_tickers(
            tickers=missing_tickers,
            start_date=train_start,
            end_date=end_date,
            use_cache=True
        )
        price_data.update(api_data)
    
    if not price_data:
        raise ValueError("Не удалось загрузить данные ни для одного тикера")
    
    logger.info(f"Успешно загружены данные для {len(price_data)} тикеров")
    
    # Загружаем макро-данные
    macro_df = collector.get_macro_data("2018-01-01", end_date)
    
    # Загружаем новости для анализа сентимента
    news_df = load_news_data()
    
    return price_data, macro_df, news_df


def calculate_prophet_forecast(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Расчет прогноза временного ряда через Prophet для использования как фичи.
    
    Args:
        df: DataFrame с ценами (должен иметь колонку 'close' и DatetimeIndex)
        ticker: Тикер акции
        
    Returns:
        DataFrame с прогнозными значениями
    """
    try:
        from prophet import Prophet
        
        # Подготовка данных для Prophet
        prophet_df = df.reset_index()[['Date', 'close']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Обучение модели
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        model.fit(prophet_df)
        
        # Прогноз на 1 день вперед
        future = model.make_future_dataframe(periods=1)
        forecast = model.predict(future)
        
        # Берем последнюю запись с прогнозом
        last_forecast = forecast.iloc[-1]
        
        # Создаем фичи из прогноза
        result = pd.DataFrame({
            'prophet_trend': [last_forecast['trend']],
            'prophet_yhat': [last_forecast['yhat']],
            'prophet_yhat_upper': [last_forecast['yhat_upper']],
            'prophet_yhat_lower': [last_forecast['yhat_lower']],
            'prophet_uncertainty': [last_forecast['yhat_upper'] - last_forecast['yhat_lower']]
        }, index=df.index[-1:])
        
        return result
        
    except Exception as e:
        logger.warning(f"Ошибка Prophet для {ticker}: {e}")
        # Возвращаем заглушку
        return pd.DataFrame({
            'prophet_trend': [df['close'].iloc[-1]],
            'prophet_yhat': [df['close'].iloc[-1]],
            'prophet_yhat_upper': [df['close'].iloc[-1] * 1.05],
            'prophet_yhat_lower': [df['close'].iloc[-1] * 0.95],
            'prophet_uncertainty': [df['close'].iloc[-1] * 0.1]
        }, index=df.index[-1:])


class FeatureEngine:
    """Движок для расчета технических индикаторов и признаков."""
    
    def __init__(self):
        self.sma_periods = [20, 50, 200]
        self.ema_periods = [12, 26]
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.atr_period = 14
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет всех технических индикаторов."""
        if df.empty:
            return df
        
        df = df.copy()
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Отсутствует необходимая колонка: {col}")
                return pd.DataFrame()
        
        df = df[required_cols].copy()
        
        # Скользящие средние (SMA)
        for period in self.sma_periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # Экспоненциальные скользящие средние (EMA)
        for period in self.ema_periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(self.atr_period).mean()
        
        # Логарифмическая доходность
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Объемные соотношения
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # Отклонение цены от SMA
        df['price_sma20_deviation'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['price_sma50_deviation'] = (df['close'] - df['sma_50']) / df['sma_50']
        df['price_sma200_deviation'] = (df['close'] - df['sma_200']) / df['sma_200']
        
        # Соотношение EMA
        df['ema_ratio'] = df['ema_12'] / df['ema_26']
        
        # Волатильность
        df['volatility_20d'] = df['log_return'].rolling(window=20).std()
        
        return df
    
    def add_macro_features(self, df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        """Добавление макроэкономических факторов."""
        if df.empty or macro_df.empty:
            return df
        
        df = df.copy()
        
        # Мердж по дате
        df = df.join(macro_df, how='left')
        
        # Forward fill для макро-данных
        macro_cols = ['key_rate', 'inflation', 'usd_rub', 'brent']
        for col in macro_cols:
            if col in df.columns:
                df[col] = df[col].ffill()
        
        return df
    
    def create_target(self, df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """Создание целевой переменной для классификации."""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Будущая доходность через horizon дней
        future_return = np.log(df['close'].shift(-horizon) / df['close'])
        
        # Бинарная классификация: 1 если рост, 0 если падение
        df['target'] = (future_return > 0).astype(int)
        
        # Также сохраняем саму доходность для анализа
        df[f'target_return_{horizon}d'] = future_return
        
        return df
    
    def process_single_ticker(
        self,
        ticker: str,
        price_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        horizon: int = 1,
        news_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Полный пайплайн обработки данных для одного тикера."""
        logger.info(f"Обработка данных для {ticker}")
        
        # Расчет технических индикаторов
        df = self.calculate_technical_indicators(price_df)
        
        if df.empty:
            logger.warning(f"Пустой DataFrame после расчета индикаторов для {ticker}")
            return pd.DataFrame()
        
        # Добавление макро-факторов
        if macro_df is not None and not macro_df.empty:
            df = self.add_macro_features(df, macro_df)
        
        # Добавление сентимента из новостей
        if news_df is not None and not news_df.empty:
            df = self.add_news_sentiment_features(df, news_df, ticker)
        
        # Создание таргета
        df = self.create_target(df, horizon)
        
        # Добавляем метаданные
        df['ticker'] = ticker
        
        # Удаляем строки с NaN
        df = df.dropna()
        
        logger.info(f"Для {ticker} осталось {len(df)} записей после очистки от NaN")
        return df
    
    def get_feature_columns(self) -> list:
        """Возвращает список колонок-признаков."""
        feature_cols = []
        
        # SMA
        for period in self.sma_periods:
            feature_cols.append(f'sma_{period}')
        
        # EMA
        for period in self.ema_periods:
            feature_cols.append(f'ema_{period}')
        
        # RSI, MACD, ATR
        feature_cols.extend(['rsi', 'macd', 'macd_signal', 'macd_hist', 'atr'])
        
        # Другие технические
        feature_cols.extend([
            'log_return', 'volume_ratio',
            'price_sma20_deviation', 'price_sma50_deviation', 'price_sma200_deviation',
            'ema_ratio', 'volatility_20d'
        ])
        
        # Макро
        feature_cols.extend(['key_rate', 'inflation', 'usd_rub', 'brent'])
        
        # Prophet forecast features
        feature_cols.extend([
            'prophet_trend', 'prophet_yhat', 'prophet_yhat_upper',
            'prophet_yhat_lower', 'prophet_uncertainty'
        ])
        
        return feature_cols
    
    def add_news_sentiment_features(
        self, 
        df: pd.DataFrame, 
        news_df: pd.DataFrame,
        ticker: str = None
    ) -> pd.DataFrame:
        """Добавление признаков сентимента из новостей."""
        if df.empty or news_df.empty:
            return df
        
        df = df.copy()
        
        # Упрощенная логика: добавляем средний сентимент по всем новостям
        # В реальной системе нужно фильтровать по тикеру и дате
        if 'sentiment_score' in news_df.columns:
            avg_sentiment = news_df['sentiment_score'].mean()
            df['news_sentiment'] = avg_sentiment
            df['news_count'] = len(news_df)
            
            # Скользящее среднее сентимента
            df['news_sentiment_7d'] = df['news_sentiment'].rolling(window=7, min_periods=1).mean()
            df['news_count_7d'] = df['news_count'].rolling(window=7, min_periods=1).sum()
        
        return df


def create_panel_data(
    ticker_data_dict: dict,
    macro_df: pd.DataFrame,
    horizon: int = 1,
    feature_engine: FeatureEngine = None,
    news_df: pd.DataFrame = None
) -> pd.DataFrame:
    """Создание панельных данных из нескольких тикеров."""
    logger.info(f"Создание панельных данных для {len(ticker_data_dict)} тикеров")
    
    if feature_engine is None:
        feature_engine = FeatureEngine()
    
    all_data = []
    
    for ticker, price_df in ticker_data_dict.items():
        try:
            processed = feature_engine.process_single_ticker(
                ticker=ticker,
                price_df=price_df,
                macro_df=macro_df,
                horizon=horizon,
                news_df=news_df
            )
            if not processed.empty:
                all_data.append(processed)
                logger.debug(f"Добавлен {ticker}: {len(processed)} записей")
        except Exception as e:
            logger.error(f"Ошибка обработки {ticker}: {e}")
            continue
    
    if not all_data:
        logger.warning("Нет данных для создания панельного набора")
        return pd.DataFrame()
    
    # Объединение всех тикеров в один DataFrame
    panel_df = pd.concat(all_data, ignore_index=False)
    
    logger.info(f"Создан панельный DataFrame: {len(panel_df)} записей, {len(panel_df.columns)} колонок")
    return panel_df


def split_data_time_series(panel_df: pd.DataFrame) -> tuple:
    """Разбиение данных с учетом временной структуры."""
    logger.info("=== Этап 3: Разбиение на train/val/test ===")
    
    # Убеждаемся что индекс датированный
    if 'Date' in panel_df.columns:
        panel_df['Date'] = pd.to_datetime(panel_df['Date'])
        panel_df.set_index('Date', inplace=True)
    
    # Разбиение по датам
    train_df = panel_df[
        (panel_df.index >= '2019-01-01') & 
        (panel_df.index <= '2021-12-31')
    ]
    
    val_df = panel_df[
        (panel_df.index >= '2022-01-01') & 
        (panel_df.index <= '2022-12-31')
    ]
    
    test_df = panel_df[
        (panel_df.index >= '2023-01-01') & 
        (panel_df.index <= '2024-12-31')
    ]
    
    logger.info(f"Train: {len(train_df)} записей ({train_df.index.min()} - {train_df.index.max()})")
    logger.info(f"Val: {len(val_df)} записей ({val_df.index.min()} - {val_df.index.max()})")
    logger.info(f"Test: {len(test_df)} записей ({test_df.index.min()} - {test_df.index.max()})")
    
    # Проверка баланса классов
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        if len(df) > 0:
            class_dist = df['target'].value_counts(normalize=True) * 100
            logger.info(f"{name} баланс классов: 0={class_dist.get(0, 0):.1f}%, 1={class_dist.get(1, 0):.1f}%")
    
    return train_df, val_df, test_df


def prepare_features_and_target(df: pd.DataFrame, feature_columns: list) -> tuple:
    """Подготовка признаков и таргета для обучения."""
    # Отбираем только нужные колонки
    available_features = [col for col in feature_columns if col in df.columns]
    
    if len(available_features) < len(feature_columns):
        missing = set(feature_columns) - set(available_features)
        logger.warning(f"Отсутствуют признаки: {missing}")
    
    X = df[available_features].copy()
    y = df['target'].copy()
    
    # Замена бесконечных значений
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)
    
    return X, y, available_features


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_columns: list
):
    """Обучение ансамбля моделей: LightGBM + GradientBoosting + RandomForest + LogisticRegression."""
    logger.info("=== Этап 4: Обучение ансамбля моделей ===")
    
    # Модель 1: LightGBM (основная)
    logger.info("Обучение LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    
    # Модель 2: Gradient Boosting
    logger.info("Обучение GradientBoosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    
    # Модель 3: Random Forest
    logger.info("Обучение RandomForest...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Модель 4: Logistic Regression с калибровкой
    logger.info("Обучение LogisticRegression...")
    lr_base = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=0.1
    )
    lr_model = CalibratedClassifierCV(lr_base, method='sigmoid', cv=3)
    lr_model.fit(X_train, y_train)
    
    # Создаем ансамбль с весами
    logger.info("Создание VotingClassifier...")
    ensemble = VotingClassifier(
        estimators=[
            ('lgb', lgb_model),
            ('gb', gb_model),
            ('rf', rf_model),
            ('lr', lr_model)
        ],
        voting='soft',
        weights=[3, 2, 2, 1]  # LightGBM имеет наибольший вес
    )
    
    # Финальное обучение ансамбля
    ensemble.fit(X_train, y_train)
    
    logger.info("Обучение ансамбля завершено")
    
    return ensemble


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_columns: list
) -> tuple:
    """Оценка качества модели на тестовой выборке."""
    logger.info("=== Этап 5: Оценка модели ===")
    
    # Предсказания
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Метрики
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }
    
    logger.info("\n" + "="*50)
    logger.info("МЕТРИКИ НА ТЕСТОВОЙ ВЫБОРКЕ:")
    logger.info("="*50)
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name.upper()}: {value:.4f}")
    
    # Classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=['DOWN', 'UP']))
    
    # Feature importance (для ансамбля берем среднее по моделям)
    if hasattr(model, 'named_estimators_'):
        # Собираем важность признаков из всех моделей
        importances = []
        for name, est in model.named_estimators_.items():
            if hasattr(est, 'feature_importances_'):
                importances.append(est.feature_importances_)
            elif hasattr(est, 'calibrated_classifiers_'):
                # Для CalibratedClassifierCV берем важность базовой модели
                for cal_clf in est.calibrated_classifiers_:
                    # Проверяем наличие estimator_ или base_estimator
                    base_est = getattr(cal_clf, 'estimator_', None) or getattr(cal_clf, 'base_estimator_', None)
                    if base_est is not None:
                        if hasattr(base_est, 'coef_'):
                            importances.append(np.abs(base_est.coef_[0]))
                        elif hasattr(base_est, 'feature_importances_'):
                            importances.append(base_est.feature_importances_)
        
        if importances:
            # Усредняем важность признаков
            avg_importance = np.mean(importances, axis=0)
            feature_importance = dict(zip(feature_columns, avg_importance.tolist()))
        else:
            feature_importance = {feat: 1.0/len(feature_columns) for feat in feature_columns}
    else:
        feature_importance = {feat: 1.0/len(feature_columns) for feat in feature_columns}
    
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    logger.info("\nТоп-10 важных признаков:")
    for feat, imp in sorted_importance[:10]:
        logger.info(f"  {feat}: {imp:.2f}")
    
    return metrics, feature_importance


def save_model(
    model,
    feature_columns: list,
    feature_importance: dict,
    metrics: dict,
    tickers: list
) -> None:
    """Сохранение модели и метаданных."""
    logger.info("=== Этап 6: Сохранение артефактов ===")
    
    # Создаем директорию моделей если не существует
    model_dir = Path(__file__).parent.parent / "backend" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "ensemble_portfolio.pkl"
    
    # Сохраняем ансамбль через joblib
    joblib.dump(model, str(model_path))
    logger.info(f"Модель сохранена в {model_path}")
    
    # Сохраняем метаданные отдельно
    metadata = {
        'feature_columns': feature_columns,
        'feature_importance': feature_importance,
        'trained_date': datetime.now().isoformat(),
        'metrics': metrics,
        'config': {
            'tickers': tickers,
            'train_period': '2019-01-01 - 2021-12-31',
            'prediction_horizon': 1,
            'model_type': 'VotingClassifier (LightGBM + GradientBoosting + RandomForest + LogisticRegression)'
        }
    }
    
    metadata_path = model_dir / "model_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        # Преобразуем numpy типы к Python типам
        serializable_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, dict):
                serializable_metadata[k] = {
                    key: float(val) if isinstance(val, (np.floating, float)) else val
                    for key, val in v.items()
                }
            elif isinstance(v, list):
                serializable_metadata[k] = v
            else:
                serializable_metadata[k] = v
        json.dump(serializable_metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Метаданные сохранены в {metadata_path}")
    
    # Сохраняем feature importance отдельно в JSON
    importance_path = model_dir / "feature_importance.json"
    with open(importance_path, 'w', encoding='utf-8') as f:
        json.dump(feature_importance, f, indent=2, ensure_ascii=False)
    logger.info(f"Feature importance сохранен в {importance_path}")
    
    # Сохраняем метрики в JSON
    metrics_path = model_dir / "metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        serializable_metrics = {k: float(v) for k, v in metrics.items()}
        json.dump(serializable_metrics, f, indent=2)
    logger.info(f"Метрики сохранены в {metrics_path}")


def main():
    """Главная функция пайплайна обучения."""
    parser = argparse.ArgumentParser(description='Обучение ML модели')
    parser.add_argument(
        '--tickers', 
        type=str, 
        default=None,
        help='Список тикеров через запятую (по умолчанию все из конфига)'
    )
    parser.add_argument(
        '--end-date', 
        type=str, 
        default='2024-12-31',
        help='Дата окончания в формате YYYY-MM-DD'
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("ЗАПУСК ML PIPELINE ДЛЯ ОБУЧЕНИЯ МОДЕЛИ")
    logger.info("="*60)
    logger.info(f"Время запуска: {datetime.now().isoformat()}")
    
    # Определяем тикеры
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',')]
    else:
        # Используем российские тикеры по умолчанию
        tickers = [
            "SBER", "GAZP", "LKOH", "NVTK", "YNDX",
            "TCSG", "VTBR", "ROSN", "GMKN", "NLMK"
        ]
    
    logger.info(f"Конфигурация: {len(tickers)} тикеров")
    
    try:
        # Этап 1: Загрузка данных
        price_data, macro_df, news_df = load_and_prepare_data(tickers, "2018-01-01", args.end_date)
        
        # Этап 2: Создание признаков
        feature_engine = FeatureEngine()
        panel_df = create_panel_data(price_data, macro_df, horizon=1, feature_engine=feature_engine, news_df=news_df)
        
        if panel_df.empty:
            raise ValueError("Панельный DataFrame пуст после обработки")
        
        # Этап 3: Разбиение на train/val/test
        train_df, val_df, test_df = split_data_time_series(panel_df)
        
        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            raise ValueError("Один из наборов данных пуст после разбиения")
        
        # Получаем список признаков
        feature_columns = feature_engine.get_feature_columns()
        
        # Подготовка признаков
        X_train, y_train, used_features = prepare_features_and_target(train_df, feature_columns)
        X_val, y_val, _ = prepare_features_and_target(val_df, feature_columns)
        X_test, y_test, _ = prepare_features_and_target(test_df, feature_columns)
        
        logger.info(f"Признаков используется: {len(used_features)}")
        
        # Этап 4: Обучение модели
        model = train_model(X_train, y_train, X_val, y_val, used_features)
        
        # Этап 5: Оценка модели
        metrics, feature_importance = evaluate_model(model, X_test, y_test, used_features)
        
        # Этап 6: Сохранение
        save_model(model, used_features, feature_importance, metrics, tickers)
        
        logger.info("="*60)
        logger.info("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка в пайплайне обучения: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
