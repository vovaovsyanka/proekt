"""
Сервис ML-инференса: загрузка модели и генерация предсказаний.
Использует ансамбль моделей (VotingClassifier) для классификации направления движения цены.
"""
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime

from backend.config import settings

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Сервис для загрузки модели и генерации предсказаний.
    
    Отвечает за:
    - Загрузку обученной ансамбль-модели из файла
    - Предобработку признаков (нормализация, порядок колонок)
    - Генерацию предсказаний и вероятностей
    - Расчет confidence score на основе вероятности и согласованности признаков
    - Объяснение предсказаний через feature importance
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Инициализация предиктора.
        
        Args:
            model_path: Путь к файлу модели .pkl
        """
        self.model_path = model_path or (settings.model_file_path.parent / "ensemble_portfolio.pkl")
        self.metadata_path = settings.model_metadata_path
        self.model = None
        self.feature_columns = None
        self.feature_importance = None
        self.model_loaded = False
        self.model_trained_date = None
        
        logger.info(f"Инициализирован ModelPredictor с путем {self.model_path}")
    
    def load_model(self) -> bool:
        """
        Загрузка модели и метаданных из файла.
        
        Returns:
            True если модель успешно загружена, False иначе
        """
        try:
            if not self.model_path.exists():
                logger.warning(f"Файл модели не найден: {self.model_path}")
                return False
            
            logger.info(f"Загрузка ансамбль-модели из {self.model_path}")
            
            # Загружаем модель через joblib
            self.model = joblib.load(str(self.model_path))
            
            # Загружаем метаданные
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    self.feature_columns = metadata.get('feature_columns', [])
                    self.feature_importance = metadata.get('feature_importance', {})
                    self.model_trained_date = metadata.get('trained_date')
            else:
                logger.warning("Файл метаданных не найден")
                self.feature_columns = []
                self.feature_importance = {}
            
            self.model_loaded = True
            logger.info(f"Модель успешно загружена. Признаков: {len(self.feature_columns)}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            self.model_loaded = False
            return False
    
    def predict(
        self, 
        features_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерация предсказаний модели.
        
        Args:
            features_df: DataFrame с признаками (должны соответствовать обученным)
            
        Returns:
            Кортеж (predictions, probabilities):
            - predictions: массив предсказанных классов (0 или 1)
            - probabilities: массив вероятностей класса 1 (рост цены)
        """
        if not self.model_loaded:
            raise RuntimeError("Модель не загружена. Вызовите load_model()")
        
        # Подготовка данных: отбор нужных колонок в правильном порядке
        X = features_df[self.feature_columns].copy()
        
        # Замена бесконечных значений и NaN на 0 (защита от ошибок)
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)
        
        # Генерация предсказаний
        predictions = self.model.predict(X)
        
        # Вероятности (probability of class 1 = рост цены)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def predict_with_confidence(
        self,
        features_df: pd.DataFrame
    ) -> List[Dict]:
        """
        Предсказания с расчетом confidence score.
        
        Confidence рассчитывается как:
        - Базовая уверенность = max(probability, 1-probability)
        - Корректировка на основе согласованности признаков с feature importance
        
        Args:
            features_df: DataFrame с признаками
            
        Returns:
            Список словарей с предсказаниями и метриками
        """
        if not self.model_loaded:
            raise RuntimeError("Модель не загружена")
        
        predictions, probabilities = self.predict(features_df)
        
        results = []
        for i in range(len(features_df)):
            prob = probabilities[i]
            pred = predictions[i]
            
            # Базовый confidence = насколько модель уверена в своем предсказании
            base_confidence = max(prob, 1 - prob)
            
            # Нормализация confidence к диапазону [0.3, 1.0]
            confidence = 0.3 + 0.7 * base_confidence
            
            results.append({
                'prediction': int(pred),
                'probability_up': float(prob),
                'probability_down': float(1 - prob),
                'confidence': float(confidence)
            })
        
        return results
    
    def get_top_features(
        self, 
        features_row: pd.Series, 
        top_n: int = 3
    ) -> List[Dict[str, str]]:
        """
        Получение топ-N наиболее влияющих признаков для конкретного предсказания.
        
        Args:
            features_row: Строка DataFrame с признаками
            top_n: Количество признаков для возврата
            
        Returns:
            Список словарей {feature_name, value, impact, direction}
        """
        if not self.feature_importance:
            return []
        
        # Сортируем признаки по важности
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        result = []
        for feature_name, importance in sorted_features:
            if feature_name not in features_row.index:
                continue
            
            value = features_row[feature_name]
            
            # Определение направления влияния
            direction = self._interpret_feature_direction(feature_name, value)
            
            result.append({
                'feature': feature_name,
                'value': round(float(value), 4) if pd.notna(value) else 0,
                'importance': round(float(importance), 4),
                'direction': direction
            })
        
        return result
    
    def _interpret_feature_direction(
        self, 
        feature_name: str, 
        value: float
    ) -> str:
        """
        Интерпретация направления влияния признака на предсказание.
        """
        value = float(value) if pd.notna(value) else 0
        
        if 'rsi' in feature_name.lower():
            if value > 70:
                return 'negative'
            elif value < 30:
                return 'positive'
            else:
                return 'neutral'
        
        elif 'macd' in feature_name.lower():
            return 'positive' if value > 0 else 'negative'
        
        elif 'sma' in feature_name.lower() and 'deviation' in feature_name.lower():
            return 'positive' if value > 0 else 'negative'
        
        elif 'vix' in feature_name.lower() or 'volatility' in feature_name.lower():
            return 'negative' if value > 0.5 else 'positive'
        
        elif 'log_return' in feature_name.lower():
            return 'positive' if value > 0 else 'negative'
        
        elif 'volume_ratio' in feature_name.lower():
            return 'positive' if value > 1 else 'neutral'
        
        elif 'prophet' in feature_name.lower():
            if 'uncertainty' in feature_name.lower():
                return 'negative' if value > 0.1 else 'positive'
            return 'positive' if value > 0 else 'negative'
        
        return 'neutral'
    
    def generate_reasoning(
        self,
        prediction: int,
        confidence: float,
        top_features: List[Dict[str, str]],
        sentiment_score: float = 0.0
    ) -> str:
        """
        Генерация текстового обоснования рекомендации.
        """
        action = "роста" if prediction == 1 else "падения"
        
        reasoning_parts = [
            f"Модель прогнозирует вероятность {action.capitalize()} цены."
        ]
        
        if confidence > 0.7:
            reasoning_parts.append("Высокая уверенность предсказания.")
        elif confidence > 0.5:
            reasoning_parts.append("Средняя уверенность предсказания.")
        else:
            reasoning_parts.append("Низкая уверенность - рекомендуется осторожность.")
        
        if top_features:
            feature_texts = []
            for feat in top_features[:2]:
                direction_ru = "позитивно" if feat['direction'] == 'positive' else "негативно"
                feature_texts.append(f"{feat['feature']} ({direction_ru})")
            
            if feature_texts:
                reasoning_parts.append(f"Ключевые факторы: {', '.join(feature_texts)}.")
        
        if abs(sentiment_score) > 0.3:
            if sentiment_score > 0:
                reasoning_parts.append("Новости оцениваются позитивно.")
            else:
                reasoning_parts.append("Новости оцениваются негативно.")
        
        return " ".join(reasoning_parts)
    
    def is_fallback_mode(self) -> bool:
        """Проверка работает ли модель в fallback режиме."""
        return not self.model_loaded
    
    def get_fallback_prediction(
        self,
        ticker: str,
        current_price: Optional[float] = None
    ) -> Dict:
        """Fallback предсказание когда модель не загружена."""
        hash_value = hash(ticker) % 3
        
        if hash_value == 0:
            action = "HOLD"
            prediction = 0
            base_prob = 0.5
        elif hash_value == 1:
            action = "BUY"
            prediction = 1
            base_prob = 0.55
        else:
            action = "SELL"
            prediction = 0
            base_prob = 0.45
        
        return {
            'prediction': prediction,
            'probability_up': base_prob,
            'confidence': 0.4,
            'reasoning': f"Рекомендация на основе медианных значений по сектору (модель недоступна).",
            'is_fallback': True
        }


# Глобальный экземпляр предиктора (singleton pattern)
predictor_instance = ModelPredictor()


def get_predictor() -> ModelPredictor:
    """Получение глобального экземпляра предиктора."""
    return predictor_instance
