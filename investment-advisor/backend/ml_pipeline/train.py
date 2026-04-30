"""
ML Pipeline: обучение модели на исторических данных.
Скрипт для генерации признаков, обучения LightGBM и сохранения артефактов.

Использование:
    python backend/ml_pipeline/train.py
"""
import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
import warnings

# Добавляем корень проекта в path для импортов
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from backend.config import settings
from backend.app.services.data_loader import DataLoader
from backend.app.services.feature_engine import FeatureEngine

# Игнорируем предупреждения для чистоты вывода
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_prepare_data() -> tuple:
    """
    Загрузка и подготовка данных для обучения.
    
    Returns:
        Кортеж (panel_df, macro_df) с панельными данными и макро-факторами
    """
    logger.info("=== Этап 1: Загрузка данных ===")
    
    # Инициализация загрузчика
    data_loader = DataLoader()
    
    # Загружаем данные по тикерам из конфига
    tickers = settings.default_tickers
    logger.info(f"Загрузка данных для {len(tickers)} тикеров: {tickers[:5]}...")
    
    # Загружаем данные за период обучения + немного раньше для расчета индикаторов
    start_date = "2017-01-01"  # Начинаем раньше для расчета SMA200
    end_date = settings.test_end_date
    
    price_data = data_loader.download_multiple_tickers(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        use_cache=True
    )
    
    if not price_data:
        raise ValueError("Не удалось загрузить данные ни для одного тикера")
    
    logger.info(f"Успешно загружены данные для {len(price_data)} тикеров")
    
    # Загружаем макро-данные
    macro_df = data_loader.get_macro_data(start_date, end_date)
    
    return price_data, macro_df


def create_features(price_data: dict, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Создание признаков для всех тикеров.
    
    Args:
        price_data: Словарь {ticker: price_dataframe}
        macro_df: DataFrame с макро-факторами
        
    Returns:
        Панельный DataFrame со всеми признаками
    """
    logger.info("=== Этап 2: Feature Engineering ===")
    
    feature_engine = FeatureEngine()
    
    # Создаем панельные данные
    panel_df = feature_engine.create_panel_data(
        ticker_data_dict=price_data,
        macro_df=macro_df,
        horizon=settings.prediction_horizon
    )
    
    if panel_df.empty:
        raise ValueError("Панельный DataFrame пуст после обработки")
    
    logger.info(f"Создан панельный DataFrame: {len(panel_df)} записей")
    logger.info(f"Колонки: {list(panel_df.columns)}")
    
    # Сохраняем промежуточный результат
    features_path = settings.features_dir / "panel_data.csv"
    panel_df.to_csv(features_path)
    logger.info(f"Панельные данные сохранены в {features_path}")
    
    return panel_df


def split_data_time_series(panel_df: pd.DataFrame) -> tuple:
    """
    Разбиение данных с учетом временной структуры (без shuffle!).
    
    Train: 2018-2021
    Val: 2022
    Test: 2023-2024
    
    Args:
        panel_df: Панельный DataFrame
        
    Returns:
        Кортеж (train_df, val_df, test_df)
    """
    logger.info("=== Этап 3: Разбиение на train/val/test ===")
    
    # Убеждаемся что индекс датированный
    if 'Date' in panel_df.columns:
        panel_df['Date'] = pd.to_datetime(panel_df['Date'])
        panel_df.set_index('Date', inplace=True)
    
    # Разбиение по датам
    train_df = panel_df[
        (panel_df.index >= settings.train_start_date) & 
        (panel_df.index <= settings.train_end_date)
    ]
    
    val_df = panel_df[
        (panel_df.index >= settings.val_start_date) & 
        (panel_df.index <= settings.val_end_date)
    ]
    
    test_df = panel_df[
        (panel_df.index >= settings.test_start_date) & 
        (panel_df.index <= settings.test_end_date)
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


def prepare_features_and_target(df: pd.DataFrame) -> tuple:
    """
    Подготовка признаков и таргета для обучения.
    
    Args:
        df: DataFrame с данными
        
    Returns:
        Кортеж (X, y, feature_columns)
    """
    feature_engine = FeatureEngine()
    feature_columns = feature_engine.get_feature_columns()
    
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
) -> lgb.LGBMClassifier:
    """
    Обучение модели LightGBM с early stopping.
    
    Args:
        X_train, y_train: Обучающие данные
        X_val, y_val: Валидационные данные
        feature_columns: Список признаков
        
    Returns:
        Обученная модель
    """
    logger.info("=== Этап 4: Обучение модели ===")
    
    # Параметры модели из конфига
    model = lgb.LGBMClassifier(
        n_estimators=settings.lgb_num_estimators,
        learning_rate=settings.lgb_learning_rate,
        max_depth=settings.lgb_max_depth,
        min_child_samples=settings.lgb_min_child_samples,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    
    # Обучение с early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='logloss',
        early_stopping_rounds=settings.lgb_early_stopping_rounds,
        verbose=True
    )
    
    logger.info(f"Обучение завершено. Использовано {model.best_iteration_} итераций из {settings.lgb_num_estimators}")
    
    return model


def evaluate_model(
    model: lgb.LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_columns: list
) -> dict:
    """
    Оценка качества модели на тестовой выборке.
    
    Args:
        model: Обученная модель
        X_test, y_test: Тестовые данные
        feature_columns: Список признаков
        
    Returns:
        Словарь с метриками
    """
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
        'auc_roc': None  # Можно добавить если нужен ROC-AUC
    }
    
    logger.info("\n" + "="*50)
    logger.info("МЕТРИКИ НА ТЕСТОВОЙ ВЫБОРКЕ:")
    logger.info("="*50)
    for metric_name, value in metrics.items():
        if value is not None:
            logger.info(f"{metric_name.upper()}: {value:.4f}")
    
    # Classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=['DOWN', 'UP']))
    
    # Feature importance
    feature_importance = dict(zip(feature_columns, model.feature_importances_.tolist()))
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    logger.info("\nТоп-10 важных признаков:")
    for feat, imp in sorted_importance[:10]:
        logger.info(f"  {feat}: {imp:.2f}")
    
    return metrics, feature_importance


def save_model(
    model: lgb.LGBMClassifier,
    feature_columns: list,
    feature_importance: dict,
    metrics: dict
) -> None:
    """
    Сохранение модели и метаданных.
    
    Args:
        model: Обученная модель
        feature_columns: Список признаков
        feature_importance: Важность признаков
        metrics: Метрики качества
    """
    logger.info("=== Этап 6: Сохранение артефактов ===")
    
    # Создаем директорию моделей если не существует
    settings.model_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем модель вместе с метаданными
    model_data = {
        'model': model,
        'feature_columns': feature_columns,
        'feature_importance': feature_importance,
        'trained_date': datetime.now().isoformat(),
        'metrics': metrics,
        'config': {
            'tickers': settings.default_tickers,
            'train_period': f"{settings.train_start_date} - {settings.train_end_date}",
            'prediction_horizon': settings.prediction_horizon
        }
    }
    
    # Сохраняем в joblib формат
    import joblib
    joblib.dump(model_data, settings.model_file_path)
    logger.info(f"Модель сохранена в {settings.model_file_path}")
    
    # Сохраняем feature importance отдельно в JSON
    importance_path = settings.feature_importance_path
    with open(importance_path, 'w', encoding='utf-8') as f:
        json.dump(feature_importance, f, indent=2, ensure_ascii=False)
    logger.info(f"Feature importance сохранен в {importance_path}")
    
    # Сохраняем метрики в JSON
    metrics_path = settings.model_file_path.parent / "metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        # Преобразуем numpy типы к Python типам
        serializable_metrics = {}
        for k, v in metrics.items():
            if v is not None:
                serializable_metrics[k] = float(v) if isinstance(v, (np.floating, float)) else v
        json.dump(serializable_metrics, f, indent=2)
    logger.info(f"Метрики сохранены в {metrics_path}")


def main():
    """
    Главная функция пайплайна обучения.
    """
    logger.info("="*60)
    logger.info("ЗАПУСК ML PIPELINE ДЛЯ ОБУЧЕНИЯ МОДЕЛИ")
    logger.info("="*60)
    logger.info(f"Время запуска: {datetime.now().isoformat()}")
    logger.info(f"Конфигурация: {len(settings.default_tickers)} тикеров, горизонт {settings.prediction_horizon} день")
    
    try:
        # Этап 1: Загрузка данных
        price_data, macro_df = load_and_prepare_data()
        
        # Этап 2: Создание признаков
        panel_df = create_features(price_data, macro_df)
        
        # Этап 3: Разбиение на train/val/test
        train_df, val_df, test_df = split_data_time_series(panel_df)
        
        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            raise ValueError("Один из наборов данных пуст после разбиения")
        
        # Подготовка признаков
        X_train, y_train, feature_columns = prepare_features_and_target(train_df)
        X_val, y_val, _ = prepare_features_and_target(val_df)
        X_test, y_test, _ = prepare_features_and_target(test_df)
        
        logger.info(f"Признаков используется: {len(feature_columns)}")
        
        # Этап 4: Обучение модели
        model = train_model(X_train, y_train, X_val, y_val, feature_columns)
        
        # Этап 5: Оценка модели
        metrics, feature_importance = evaluate_model(model, X_test, y_test, feature_columns)
        
        # Этап 6: Сохранение
        save_model(model, feature_columns, feature_importance, metrics)
        
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
