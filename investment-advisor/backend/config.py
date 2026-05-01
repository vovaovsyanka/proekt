"""
Конфигурация системы инвестиционных рекомендаций.
Все настройки вынесены в единый модуль для удобства управления.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
from pathlib import Path
import pandas as pd

class Settings(BaseSettings):
    """ Основные настройки приложения. """
    # === API Keys ===
    alpha_vantage_api_key: str = ""
    news_api_key: str = ""

    # === Paths ===
    base_dir: Path = Path(__file__).parent.parent
    model_file_path: Path = Path(__file__).parent / "models" / "catboost_portfolio.cbm"
    model_metadata_path: Path = Path(__file__).parent / "models" / "model_metadata.json"
    feature_importance_path: Path = Path(__file__).parent / "models" / "feature_importance.json"

    # Директория для хранения всех данных
    data_dir: Path = base_dir / "data"
    raw_data_dir: Path = data_dir / "raw"
    features_dir: Path = data_dir / "features"

    # Файл со списком всех тикеров
    ticker_list_file: Path = raw_data_dir / "ticker_list.csv"

    # === Локальные датасеты ===
    kaggle_ohlcv_file: Path = raw_data_dir / "kaggle_ohlcv.csv"
    kaggle_macro_file: Path = raw_data_dir / "russian_investment.csv"
    moex_dataset_file: Path = raw_data_dir / "moex_dataset.csv"
    news_dataset_file: Path = raw_data_dir / "Kasymkhan_RussianFinancialNews.parquet"
    rfsd_dataset_path: str = "irlspbru/RFSD"

    # === Prediction Settings ===
    prediction_horizon: int = 1
    confidence_threshold: float = 0.5

    default_tickers: List[str] = [
        "SBER", "GAZP", "LKOH", "NVTK", "YNDX", "TCSG", "VTBR",
        "ROSN", "GMKN", "NLMK", "SNGS", "HYDR", "MTSS", "CHMF", "MAGN"
    ]

    # === Data Settings ===
    train_start_date: str = "2019-01-01"
    train_end_date: str = "2021-12-31"
    val_start_date: str = "2022-01-01"
    val_end_date: str = "2022-12-31"
    test_start_date: str = "2023-01-01"
    test_end_date: str = "2024-12-31"
    inference_lookback_days: int = 180
    news_count_for_sentiment: int = 10

    # === Technical Indicators ===
    sma_periods: List[int] = [20, 50, 200]
    ema_periods: List[int] = [12, 26]
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14

    # === Server ===
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        protected_namespaces=('settings_',)
    )

    def load_tickers(self) -> List[str]:
        """Загружает список тикеров из файла, если он есть."""
        if self.ticker_list_file.exists():
            df = pd.read_csv(self.ticker_list_file)
            if 'ticker' in df.columns:
                return df['ticker'].tolist()
        return self.default_tickers

# Глобальный экземпляр настроек
settings = Settings()

# Создаём необходимые директории при импорте
for dir_path in [settings.data_dir, settings.raw_data_dir, settings.features_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)