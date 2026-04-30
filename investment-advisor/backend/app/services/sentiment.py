"""
Сервис анализа тональности новостей (NLP).
Использует предобученную модель FinBERT для классификации сентимента финансовых новостей.
"""
import logging
from typing import List, Dict, Optional
from functools import lru_cache
import hashlib

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Анализатор тональности финансовых новостей.
    
    Использует модель FinBERT (ProsusAI/finbert) - специализированная BERT модель,
    дообученная на финансовых текстах. Показывает лучшие результаты чем общие модели
    sentiment analysis для финансовой тематики.
    
    Почему FinBERT:
    - Обучена на финансовых новостях и отчетах
    - Понимает финансовую терминологию (earnings, dividend, IPO и т.д.)
    - Различает контекст (например "bull market" vs "bear market")
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Инициализация анализатора сентимента.
        
        Args:
            model_name: Название модели transformers для загрузки
        """
        self.model_name = model_name
        self._pipeline = None
        logger.info(f"Инициализирован SentimentAnalyzer с моделью {model_name}")
    
    @property
    def pipeline(self):
        """
        Ленивая загрузка модели pipeline.
        Загружается только при первом вызове для экономии памяти.
        """
        if self._pipeline is None:
            logger.info(f"Загрузка модели {self.model_name}...")
            try:
                from transformers import pipeline
                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    device=-1,  # CPU (для GPU использовать device=0)
                    truncation=True,
                    max_length=512
                )
                logger.info("Модель успешно загружена")
            except Exception as e:
                logger.error(f"Ошибка загрузки модели: {e}")
                raise
        return self._pipeline
    
    def analyze_single_text(self, text: str) -> Dict[str, float]:
        """
        Анализ тональности одного текста.
        
        Args:
            text: Текст новости для анализа
            
        Returns:
            Словарь с метриками:
            - positive: вероятность позитивного сентимента (0-1)
            - negative: вероятность негативного сентимента (0-1)
            - neutral: вероятность нейтрального сентимента (0-1)
            - compound: итоговый скор от -1 (негатив) до +1 (позитив)
        """
        if not text or not text.strip():
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0
            }
        
        try:
            # Получаем предсказание от модели
            result = self.pipeline(text)[0]
            
            label = result['label']
            score = result['score']
            
            # FinBERT возвращает один из трех лейблов: positive/negative/neutral
            # Преобразуем в вероятности по всем классам
            sentiment_scores = {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 0.0
            }
            
            # Присваиваем скор полученному лейблу
            sentiment_scores[label.lower()] = score
            
            # Compound score: нормализованный показатель от -1 до +1
            # positive дает +score, negative дает -score, neutral дает 0
            compound = sentiment_scores['positive'] - sentiment_scores['negative']
            
            return {
                'positive': sentiment_scores['positive'],
                'negative': sentiment_scores['negative'],
                'neutral': sentiment_scores['neutral'],
                'compound': compound
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа текста: {e}")
            # Возвращаем нейтральный сентимент при ошибке
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0
            }
    
    def analyze_news_list(
        self, 
        news_list: List[Dict[str, str]],
        title_field: str = 'title'
    ) -> Dict[str, float]:
        """
        Агрегированный анализ списка новостей.
        
        Args:
            news_list: Список новостей (словарей с текстом)
            title_field: Название поля с текстом новости
            
        Returns:
            Агрегированные метрики сентимента:
            - avg_positive: средняя вероятность позитива
            - avg_negative: средняя вероятность негатива
            - avg_neutral: средняя вероятность нейтральности
            - avg_compound: средний compound score
            - news_count: количество проанализированных новостей
            - dominant_sentiment: преобладающий сентимент
        """
        if not news_list:
            return {
                'avg_positive': 0.0,
                'avg_negative': 0.0,
                'avg_neutral': 1.0,
                'avg_compound': 0.0,
                'news_count': 0,
                'dominant_sentiment': 'neutral'
            }
        
        all_scores = []
        
        for news_item in news_list:
            text = news_item.get(title_field, '')
            if text:
                scores = self.analyze_single_text(text)
                all_scores.append(scores)
        
        if not all_scores:
            return {
                'avg_positive': 0.0,
                'avg_negative': 0.0,
                'avg_neutral': 1.0,
                'avg_compound': 0.0,
                'news_count': 0,
                'dominant_sentiment': 'neutral'
            }
        
        # Усреднение метрик по всем новостям
        avg_positive = sum(s['positive'] for s in all_scores) / len(all_scores)
        avg_negative = sum(s['negative'] for s in all_scores) / len(all_scores)
        avg_neutral = sum(s['neutral'] for s in all_scores) / len(all_scores)
        avg_compound = sum(s['compound'] for s in all_scores) / len(all_scores)
        
        # Определение доминирующего сентимента
        if avg_positive > avg_negative and avg_positive > avg_neutral:
            dominant = 'positive'
        elif avg_negative > avg_positive and avg_negative > avg_neutral:
            dominant = 'negative'
        else:
            dominant = 'neutral'
        
        return {
            'avg_positive': round(avg_positive, 4),
            'avg_negative': round(avg_negative, 4),
            'avg_neutral': round(avg_neutral, 4),
            'avg_compound': round(avg_compound, 4),
            'news_count': len(all_scores),
            'dominant_sentiment': dominant
        }
    
    def get_sentiment_feature(self, news_list: List[Dict[str, str]]) -> float:
        """
        Получение единого числового признака сентимента для ML модели.
        
        Args:
            news_list: Список новостей
            
        Returns:
            Числовой признак (compound score) для использования в ML модели
        """
        aggregated = self.analyze_news_list(news_list)
        return aggregated['avg_compound']


# Глобальный кэш для результатов анализа
# Кэшируем чтобы не запускать инференс многократно на одних и тех же данных
_sentiment_cache = {}


def get_cached_sentiment(
    analyzer: SentimentAnalyzer,
    news_list: List[Dict[str, str]]
) -> Dict[str, float]:
    """
    Анализ сентимента с кэшированием результатов.
    
    Args:
        analyzer: Экземпляр SentimentAnalyzer
        news_list: Список новостей
        
    Returns:
        Агрегированные метрики сентимента
    """
    # Создаем хеш от новостей для ключа кэша
    news_string = str(sorted([n.get('title', '') for n in news_list]))
    cache_key = hashlib.md5(news_string.encode()).hexdigest()
    
    if cache_key in _sentiment_cache:
        logger.debug("Сентимент загружен из кэша")
        return _sentiment_cache[cache_key]
    
    # Анализ и сохранение в кэш
    result = analyzer.analyze_news_list(news_list)
    _sentiment_cache[cache_key] = result
    
    logger.debug(f"Сентимент рассчитан и закэширован. Результат: {result['dominant_sentiment']}")
    return result
