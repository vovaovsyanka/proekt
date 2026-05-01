"""
Конфигурация системы инвестиционных рекомендаций.
Все настройки вынесены в единый модуль для удобства управления.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
from pathlib import Path
import os


class Settings(BaseSettings):
    """
    Основные настройки приложения.
    Использует pydantic-settings для загрузки из .env файла.
    """
    
    # === API Keys ===
    alpha_vantage_api_key: str = ""
    news_api_key: str = ""
    
    # === Paths ===
    # Базовая директория проекта (родительская от backend/)
    base_dir: Path = Path(__file__).parent.parent.parent
    model_file_path: Path = Path(__file__).parent / "models" / "catboost_portfolio.cbm"
    model_metadata_path: Path = Path(__file__).parent / "models" / "model_metadata.json"
    feature_importance_path: Path = Path(__file__).parent / "models" / "feature_importance.json"
    cache_dir: Path = Path(__file__).parent.parent.parent / "data" / "cache"
    raw_data_dir: Path = Path(__file__).parent.parent.parent / "data" / "raw"
    features_dir: Path = Path(__file__).parent.parent.parent / "data" / "features"
    
    # === Prediction Settings ===
    prediction_horizon: int = 1  # горизонт прогноза в днях
    confidence_threshold: float = 0.5  # минимальный порог уверенности
    # Список российских тикеров по умолчанию для обучения модели
    default_tickers: List[str] = [
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
        "MTSS",   # МТС
        "CHMF",   # Северсталь
        "MAGN"    # Магнитка
    ]
    
    # === Data Settings ===
    # Период обучения модели
    train_start_date: str = "2019-01-01"
    train_end_date: str = "2021-12-31"
    val_start_date: str = "2022-01-01"
    val_end_date: str = "2022-12-31"
    test_start_date: str = "2023-01-01"
    test_end_date: str = "2024-12-31"
    
    # Количество дней данных для инференса
    inference_lookback_days: int = 180
    # Количество новостей для анализа сентимента
    news_count_for_sentiment: int = 10
    
    # === Technical Indicators Settings ===
    # Параметры технических индикаторов
    sma_periods: List[int] = [20, 50, 200]
    ema_periods: List[int] = [12, 26]
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    
    # === Server Settings ===
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        protected_namespaces=('settings_',)  # Исключаем конфликт с model_* полями
    )


# Глобальный экземпляр настроек
settings = Settings()

# Создаем необходимые директории при импорте
for dir_path in [settings.cache_dir, settings.raw_data_dir, settings.features_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)
