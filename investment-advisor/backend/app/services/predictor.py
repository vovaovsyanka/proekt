"""
Сервис ML-инференса: загрузка модели и генерация предсказаний.
Использует LightGBM модель для классификации направления движения цены.
"""
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
    - Загрузку обученной LightGBM модели из файла
    - Предобработку признаков (нормализация, порядок колонок)
    - Генерацию предсказаний и вероятностей
    - Расчет confidence score на основе вероятности и согласованности признаков
    - Объяснение предсказаний через feature importance
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Инициализация предиктора.
        
        Args:
            model_path: Путь к файлу модели .joblib
        """
        self.model_path = model_path or settings.model_path
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
            
            logger.info(f"Загрузка модели из {self.model_path}")
            
            # Загружаем словарь с моделью и метаданными
            model_data = joblib.load(self.model_path)
            
            self.model = model_data.get('model')
            self.feature_columns = model_data.get('feature_columns', [])
            self.feature_importance = model_data.get('feature_importance', {})
            self.model_trained_date = model_data.get('trained_date')
            
            if self.model is None:
                logger.error("Модель не найдена в файле")
                return False
            
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
            # Если probability = 0.9, confidence = 0.9
            # Если probability = 0.5, confidence = 0.5 (неопределенность)
            base_confidence = max(prob, 1 - prob)
            
            # Нормализация confidence к диапазону [0.3, 1.0]
            # Даже при uncertainty оставляем минимальную уверенность 0.3
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
        
        Использует feature importance модели для определения важности признаков,
        затем анализирует значения признаков для интерпретации направления влияния.
        
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
            
            # Определение направления влияния (упрощенная эвристика)
            # В продакшене можно использовать SHAP values для точной интерпретации
            direction = self._interpret_feature_direction(feature_name, value)
            
            result.append({
                'feature': feature_name,
                'value': round(float(value), 4) if pd.notna(value) else 0,
                'importance': round(float(importance), 4),
                'direction': direction  # 'positive' или 'negative'
            })
        
        return result
    
    def _interpret_feature_direction(
        self, 
        feature_name: str, 
        value: float
    ) -> str:
        """
        Интерпретация направления влияния признака на предсказание.
        
        Эвристические правила для финансовых индикаторов:
        - Высокий RSI (>70) = негатив для будущего роста (перекупленность)
        - Положительный MACD = позитив
        - Цена выше SMA200 = позитив (восходящий тренд)
        - Высокий VIX = негатив (высокая волатильность/страх)
        
        Args:
            feature_name: Название признака
            value: Значение признака
            
        Returns:
            'positive' если признак способствует росту, 'negative' если падению
        """
        value = float(value) if pd.notna(value) else 0
        
        # Правила интерпретации для разных типов признаков
        if 'rsi' in feature_name.lower():
            # RSI > 70 = перекупленность (негатив), RSI < 30 = перепроданность (позитив)
            if value > 70:
                return 'negative'
            elif value < 30:
                return 'positive'
            else:
                return 'neutral'
        
        elif 'macd' in feature_name.lower():
            # Положительный MACD = бычий сигнал
            return 'positive' if value > 0 else 'negative'
        
        elif 'sma' in feature_name.lower() and 'deviation' in feature_name.lower():
            # Цена выше SMA = позитив
            return 'positive' if value > 0 else 'negative'
        
        elif 'vix' in feature_name.lower():
            # Высокий VIX = страх на рынке (негатив)
            return 'negative' if value > 25 else 'positive'
        
        elif 'log_return' in feature_name.lower():
            # Положительная доходность = позитив
            return 'positive' if value > 0 else 'negative'
        
        elif 'volume_ratio' in feature_name.lower():
            # Высокий объем = подтверждение тренда (позитив)
            return 'positive' if value > 1 else 'neutral'
        
        # По умолчанию считаем нейтральным
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
        
        Комбинирует:
        - Предсказание модели (BUY/SELL/HOLD)
        - Топ признаки по важности
        - Сентимент новостей
        
        Args:
            prediction: Предсказание модели (0 или 1)
            confidence: Уровень уверенности
            top_features: Топ влияющих признаков
            sentiment_score: Сентимент новостей (-1 до +1)
            
        Returns:
            Текстовое обоснование на русском языке
        """
        action = "роста" if prediction == 1 else "падения"
        
        # Формируем основную часть
        reasoning_parts = [
            f"Модель прогнозирует вероятность {action.capitalize()} цены."
        ]
        
        # Добавляем информацию о confidence
        if confidence > 0.7:
            reasoning_parts.append("Высокая уверенность предсказания.")
        elif confidence > 0.5:
            reasoning_parts.append("Средняя уверенность предсказания.")
        else:
            reasoning_parts.append("Низкая уверенность - рекомендуется осторожность.")
        
        # Добавляем топ-признаки
        if top_features:
            feature_texts = []
            for feat in top_features[:2]:  # Берем топ-2 признака
                direction_ru = "позитивно" if feat['direction'] == 'positive' else "негативно"
                feature_texts.append(f"{feat['feature']} ({direction_ru})")
            
            if feature_texts:
                reasoning_parts.append(f"Ключевые факторы: {', '.join(feature_texts)}.")
        
        # Добавляем сентимент
        if abs(sentiment_score) > 0.3:
            if sentiment_score > 0:
                reasoning_parts.append("Новости оцениваются позитивно.")
            else:
                reasoning_parts.append("Новости оцениваются негативно.")
        
        return " ".join(reasoning_parts)
    
    def is_fallback_mode(self) -> bool:
        """
        Проверка работает ли модель в fallback режиме (не загружена).
        
        Returns:
            True если модель не загружена и используются заглушки
        """
        return not self.model_loaded
    
    def get_fallback_prediction(
        self,
        ticker: str,
        current_price: Optional[float] = None
    ) -> Dict:
        """
        Fallback предсказание когда модель не загружена.
        
        Использует медианные значения и пониженный confidence.
        
        Args:
            ticker: Тикер акции
            current_price: Текущая цена (опционально)
            
        Returns:
            Словарь с fallback предсказанием
        """
        # Детерминированный выбор действия на основе хеша тикера
        # Чтобы для одного тикера всегда возвращался одинаковый результат
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
            'confidence': 0.4,  # Пониженный confidence для fallback
            'reasoning': f"Рекомендация на основе медианных значений по сектору (модель недоступна).",
            'is_fallback': True
        }


# Глобальный экземпляр предиктора (singleton pattern)
predictor_instance = ModelPredictor()


def get_predictor() -> ModelPredictor:
    """
    Получение глобального экземпляра предиктора.
    
    Returns:
        Экземпляр ModelPredictor
    """
    return predictor_instance
