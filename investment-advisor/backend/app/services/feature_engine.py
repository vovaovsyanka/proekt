"""
Feature Engineering: расчет технических индикаторов и формирование панельных данных.
Использует finta для вычисления индикаторов и создает признаки для ML модели.

Finta - альтернатива pandas-ta с похожим API.
Документация: https://github.com/peerchemist/finta
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import logging
from datetime import datetime

# Импортируем индикаторы из finta
from finta import TA

from backend.config import settings

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Движок для расчета технических индикаторов и признаков.
    
    Рассчитывает:
    - Скользящие средние (SMA, EMA)
    - Индекс относительной силы (RSI)
    - MACD
    - Средний истинный диапазон (ATR)
    - Логарифмическая доходность
    - Объемные соотношения
    - Макро-факторы
    """
    
    def __init__(self):
        """Инициализация движка признаков с параметрами из конфига."""
        self.sma_periods = settings.sma_periods
        self.ema_periods = settings.ema_periods
        self.rsi_period = settings.rsi_period
        self.macd_fast = settings.macd_fast
        self.macd_slow = settings.macd_slow
        self.macd_signal = settings.macd_signal
        self.atr_period = settings.atr_period
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет всех технических индикаторов для одного тикера.
        
        Args:
            df: DataFrame с колонками open, high, low, close, volume
            
        Returns:
            DataFrame с добавленными техническими индикаторами
        """
        if df.empty:
            return df
        
        # Создаем копию чтобы не модифицировать исходные данные
        df = df.copy()
        
        # Убеждаемся, что индекс без timezone (для совместимости с finta)
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Отбираем только нужные колонки для finta (убираем Dividends, Stock Splits и др.)
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Отсутствует необходимая колонка: {col}")
                return pd.DataFrame()
        
        df = df[required_cols].copy()
        
        # === Скользящие средние (SMA) ===
        # SMA - простой индикатор тренда. Цена выше SMA200 = долгосрочный восходящий тренд
        for period in self.sma_periods:
            df[f'sma_{period}'] = TA.SMA(df, period=period)
        
        # === Экспоненциальные скользящие средние (EMA) ===
        # EMA больше весит недавние цены, быстрее реагирует на изменения
        for period in self.ema_periods:
            df[f'ema_{period}'] = TA.EMA(df, period=period)
        
        # === RSI (Relative Strength Index) ===
        # Осциллятор от 0 до 100. >70 = перекупленность, <30 = перепроданность
        df['rsi'] = TA.RSI(df, period=self.rsi_period)
        
        # === MACD (Moving Average Convergence Divergence) ===
        # Показывает изменение импульса. Состоит из линии MACD, сигнальной линии и гистограммы
        # finta использует имена параметров: period_fast, period_slow, signal
        # Возвращает DataFrame с колонками ['MACD', 'SIGNAL'] (заглавными буквами!)
        macd_df = TA.MACD(df, period_fast=self.macd_fast, period_slow=self.macd_slow, signal=self.macd_signal)
        
        # finta возвращает DataFrame с колонками ['MACD', 'SIGNAL'] (заглавные буквы)
        df['macd'] = macd_df['MACD']
        df['macd_signal'] = macd_df['SIGNAL']
        df['macd_hist'] = macd_df['MACD'] - macd_df['SIGNAL']  # Гистограмма = MACD - Signal
        
        # === ATR (Average True Range) ===
        # Мера волатильности. Используется для оценки риска
        df['atr'] = TA.ATR(df, period=self.atr_period)
        
        # === Логарифмическая доходность ===
        # Более стабильная метрика для финансовых временных рядов
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # === Объемные соотношения ===
        # Отношение текущего объема к среднему за период - показывает аномальную активность
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # === Дополнительные признаки ===
        # Отклонение цены от SMA (процентное)
        df['price_sma20_deviation'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['price_sma50_deviation'] = (df['close'] - df['sma_50']) / df['sma_50']
        df['price_sma200_deviation'] = (df['close'] - df['sma_200']) / df['sma_200']
        
        # Соотношение EMA (быстрая к медленной) - индикатор краткосрочного тренда
        df['ema_ratio'] = df['ema_12'] / df['ema_26']
        
        # Волатильность (стандартное отклонение доходности за период)
        df['volatility_20d'] = df['log_return'].rolling(window=20).std()
        
        logger.debug(f"Рассчитано {len(df.columns)} признаков")
        return df
    
    def add_macro_features(
        self, 
        df: pd.DataFrame, 
        macro_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Добавление макроэкономических факторов к данным по акциям.
        
        Args:
            df: DataFrame с данными по акции
            macro_df: DataFrame с макро-показателями
            
        Returns:
            DataFrame с добавленными макро-признаками
        """
        if df.empty or macro_df.empty:
            return df
        
        df = df.copy()
        
        # Убираем timezone информации для совместимости
        # Проверяем тип индекса перед работой с timezone
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_convert('UTC').tz_localize(None)
        
        macro_copy = macro_df.copy()
        if hasattr(macro_copy.index, 'tz') and macro_copy.index.tz is not None:
            macro_copy.index = macro_copy.index.tz_convert('UTC').tz_localize(None)
        
        # Конвертируем индексы в datetime для надежного merge
        df.index = pd.to_datetime(df.index)
        macro_copy.index = pd.to_datetime(macro_copy.index)
        
        # Мердж по дате через join (left join чтобы сохранить все даты акций)
        # Макро-данные могут иметь другие даты, поэтому используем merge_asof или forward fill
        df = df.join(macro_copy, how='left')
        
        # Forward fill для заполнения пропусков (макро-данные обновляются реже ежедневных цен)
        macro_cols = ['inflation_rate', 'interest_rate', 'vix']
        for col in macro_cols:
            if col in df.columns:
                df[col] = df[col].ffill()
        
        logger.debug("Добавлены макро-признаки")
        return df
    
    def create_target(self, df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """
        Создание целевой переменной для классификации.
        
        Target = знак следующей доходности:
        - 1 если цена завтра вырастет
        - 0 если цена завтра упадет
        
        Args:
            df: DataFrame с признаками
            horizon: Горизонт прогноза в днях
            
        Returns:
            DataFrame с добавленной целевой переменной
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Будущая доходность через horizon дней
        future_return = np.log(df['close'].shift(-horizon) / df['close'])
        
        # Бинарная классификация: 1 если рост, 0 если падение
        df['target'] = (future_return > 0).astype(int)
        
        # Также сохраняем саму доходность для анализа
        df[f'target_return_{horizon}d'] = future_return
        
        logger.debug(f"Создана целевая переменная с горизонтом {horizon} дней")
        return df
    
    def process_single_ticker(
        self,
        ticker: str,
        price_df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
        horizon: int = 1
    ) -> pd.DataFrame:
        """
        Полный пайплайн обработки данных для одного тикера.
        
        Args:
            ticker: Тикер акции
            price_df: DataFrame с ценами
            macro_df: DataFrame с макро-данными (опционально)
            horizon: Горизонт прогноза
            
        Returns:
            DataFrame со всеми признаками и таргетом
        """
        logger.info(f"Обработка данных для {ticker}")
        
        # Расчет технических индикаторов
        df = self.calculate_technical_indicators(price_df)
        
        # Добавление макро-факторов
        if macro_df is not None and not macro_df.empty:
            df = self.add_macro_features(df, macro_df)
        
        # Создание таргета
        df = self.create_target(df, horizon)
        
        # Добавляем метаданные
        df['ticker'] = ticker
        
        # Удаляем строки с NaN (появляются при расчете скользящих средних)
        df = df.dropna()
        
        logger.info(f"Для {ticker} осталось {len(df)} записей после очистки от NaN")
        return df
    
    def create_panel_data(
        self,
        ticker_data_dict: Dict[str, pd.DataFrame],
        macro_df: Optional[pd.DataFrame] = None,
        horizon: int = 1
    ) -> pd.DataFrame:
        """
        Создание панельных данных из нескольких тикеров.
        
        Панельные данные = объединение временных рядов разных тикеров
        с сохранением идентификатора тикера для каждой записи.
        
        Args:
            ticker_data_dict: Словарь {ticker: price_dataframe}
            macro_df: DataFrame с макро-данными
            horizon: Горизонт прогноза
            
        Returns:
            Единый DataFrame со всеми данными и признаками
        """
        logger.info(f"Создание панельных данных для {len(ticker_data_dict)} тикеров")
        
        all_data = []
        
        for ticker, price_df in ticker_data_dict.items():
            try:
                processed = self.process_single_ticker(
                    ticker=ticker,
                    price_df=price_df,
                    macro_df=macro_df,
                    horizon=horizon
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
    
    def get_feature_columns(self) -> List[str]:
        """
        Возвращает список колонок-признаков (исключая метаданные и таргет).
        
        Returns:
            Список имен колонок с признаками
        """
        # Базовые колонки которые не являются признаками
        exclude_cols = [
            'ticker', 'target', 'target_return_1d',
            'open', 'high', 'low', 'close', 'adj_close', 'volume',
            'date'
        ]
        
        # Генерируем ожидаемые имена колонок признаков
        feature_cols = []
        
        # SMA
        for period in self.sma_periods:
            feature_cols.append(f'sma_{period}')
        
        # EMA
        for period in self.ema_periods:
            feature_cols.append(f'ema_{period}')
        
        # RSI
        feature_cols.append('rsi')
        
        # MACD
        feature_cols.extend(['macd', 'macd_signal', 'macd_hist'])
        
        # ATR
        feature_cols.append('atr')
        
        # Другие
        feature_cols.extend([
            'log_return', 'volume_ratio',
            'price_sma20_deviation', 'price_sma50_deviation', 'price_sma200_deviation',
            'ema_ratio', 'volatility_20d',
            'inflation_rate', 'interest_rate', 'vix'
        ])
        
        # Фильтруем только существующие колонки
        feature_cols = [col for col in feature_cols if col not in exclude_cols]
        
        return feature_cols
