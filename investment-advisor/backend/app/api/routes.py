"""
API Routes: endpoints для работы с рекомендациями.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
import logging
from datetime import datetime
import concurrent.futures
from functools import partial

from backend.app.models.schemas import (
    PortfolioRequest,
    PortfolioResponse,
    Recommendation,
    HealthResponse,
    ErrorResponse
)
from backend.app.services.data_loader import DataLoader
from backend.app.services.feature_engine import FeatureEngine
from backend.app.services.sentiment import SentimentAnalyzer, get_cached_sentiment
from backend.app.services.predictor import get_predictor, ModelPredictor
from backend.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["recommendations"])


def process_ticker_prediction(
    ticker: str,
    data_loader: DataLoader,
    feature_engine: FeatureEngine,
    sentiment_analyzer: SentimentAnalyzer,
    predictor: ModelPredictor,
    lookback_days: int = 90
) -> Recommendation:
    """
    Обработка предсказания для одного тикера (синхронная функция для ThreadPool).
    
    Args:
        ticker: Тикер акции
        data_loader: Загрузчик данных
        feature_engine: Движок признаков
        sentiment_analyzer: Анализатор сентимента
        predictor: Предиктор модели
        lookback_days: Количество дней данных для анализа
        
    Returns:
        Recommendation с предсказанием
    """
    logger.info(f"Обработка тикера {ticker}")
    
    # Получаем текущую дату и рассчитываем диапазон
    end_date = datetime.now()
    start_date = datetime(end_date.year - (1 if lookback_days < 365 else 2), end_date.month, end_date.day)
    
    # Загружаем данные
    price_df = data_loader.download_stock_data(
        ticker=ticker,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        use_cache=True
    )
    
    if price_df.empty:
        logger.warning(f"Нет данных для {ticker}, используем fallback")
        return Recommendation(
            ticker=ticker,
            action="HOLD",
            confidence=0.3,
            expected_return=0.0,
            reasoning=f"Недостаточно данных для анализа {ticker}"
        )
    
    # Получаем последнюю цену
    current_price = price_df['close'].iloc[-1]
    
    # Рассчитываем признаки для последней даты
    macro_df = data_loader.get_macro_data(
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )
    
    # Обрабатываем данные через feature engine
    processed_df = feature_engine.process_single_ticker(
        ticker=ticker,
        price_df=price_df,
        macro_df=macro_df,
        horizon=settings.prediction_horizon
    )
    
    if processed_df.empty:
        logger.warning(f"Не удалось рассчитать признаки для {ticker}")
        return Recommendation(
            ticker=ticker,
            action="HOLD",
            confidence=0.3,
            expected_return=0.0,
            reasoning=f"Ошибка расчета признаков для {ticker}"
        )
    
    # Берем последнюю запись для инференса
    last_row = processed_df.iloc[[-1]]
    
    # Получаем предсказание от модели
    if predictor.model_loaded:
        try:
            predictions = predictor.predict_with_confidence(last_row)
            pred_data = predictions[0]
            
            prediction = pred_data['prediction']
            confidence = pred_data['confidence']
            prob_up = pred_data['probability_up']
            
            # Определяем действие
            if confidence < settings.confidence_threshold:
                action = "HOLD"
            elif prediction == 1:
                action = "BUY"
            else:
                action = "SELL"
            
            # Расчет ожидаемой доходности (упрощенно на основе вероятности)
            expected_return = (prob_up - 0.5) * 10  # Масштабируем к процентам
            
            # Получаем топ признаки
            top_features = predictor.get_top_features(last_row.iloc[0], top_n=3)
            
            # Получаем сентимент новостей
            news_list = data_loader.get_news_sentiment_data(
                ticker=ticker,
                limit=settings.news_count_for_sentiment
            )
            sentiment_result = get_cached_sentiment(sentiment_analyzer, news_list)
            sentiment_score = sentiment_result['avg_compound']
            
            # Генерируем обоснование
            reasoning = predictor.generate_reasoning(
                prediction=prediction,
                confidence=confidence,
                top_features=top_features,
                sentiment_score=sentiment_score
            )
            
            return Recommendation(
                ticker=ticker,
                action=action,
                confidence=round(confidence, 3),
                expected_return=round(expected_return, 2),
                reasoning=reasoning,
                current_price=round(current_price, 2)
            )
            
        except Exception as e:
            logger.error(f"Ошибка инференса для {ticker}: {e}")
            return Recommendation(
                ticker=ticker,
                action="HOLD",
                confidence=0.4,
                expected_return=0.0,
                reasoning=f"Ошибка модели: {str(e)[:100]}"
            )
    else:
        # Fallback режим
        fallback_pred = predictor.get_fallback_prediction(ticker, current_price)
        return Recommendation(
            ticker=ticker,
            action="HOLD" if fallback_pred['prediction'] == 0 else "BUY",
            confidence=fallback_pred['confidence'],
            expected_return=0.0,
            reasoning=fallback_pred['reasoning'],
            current_price=round(current_price, 2)
        )


@router.post("/recommendations", response_model=PortfolioResponse)
async def get_recommendations(
    request: PortfolioRequest,
    background_tasks: BackgroundTasks
):
    """
    Получить рекомендации по портфелю пользователя.
    
    Анализирует каждый тикер из портфеля используя:
    - Технические индикаторы (RSI, MACD, SMA и др.)
    - ML модель (LightGBM)
    - Сентимент новостей (FinBERT)
    
    Возвращает рекомендации BUY/SELL/HOLD с обоснованием.
    """
    logger.info(f"Запрос рекомендаций для портфеля с {len(request.positions)} позициями")
    
    # Инициализация сервисов
    data_loader = DataLoader()
    feature_engine = FeatureEngine()
    sentiment_analyzer = SentimentAnalyzer()
    predictor = get_predictor()
    
    # Пытаемся загрузить модель если еще не загружена
    if not predictor.model_loaded:
        logger.info("Попытка загрузки модели...")
        predictor.load_model()
        
        if not predictor.model_loaded:
            logger.warning("Модель не загружена, работаем в fallback режиме")
    
    # Извлекаем уникальные тикеры из портфеля
    tickers = list(set(pos.ticker for pos in request.positions))
    logger.info(f"Уникальные тикеры для анализа: {tickers}")
    
    # Обрабатываем тикеры параллельно в ThreadPoolExecutor
    # чтобы не блокировать event loop при CPU-bound операциях
    recommendations = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Создаем futures для каждого тикера
        future_to_ticker = {
            executor.submit(
                process_ticker_prediction,
                ticker,
                data_loader,
                feature_engine,
                sentiment_analyzer,
                predictor
            ): ticker
            for ticker in tickers
        }
        
        # Собираем результаты по мере завершения
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                recommendation = future.result(timeout=30)  # Таймаут 30 секунд
                recommendations.append(recommendation)
            except concurrent.futures.TimeoutError:
                logger.error(f"Таймаут обработки для {ticker}")
                recommendations.append(Recommendation(
                    ticker=ticker,
                    action="HOLD",
                    confidence=0.3,
                    expected_return=0.0,
                    reasoning="Таймаут обработки данных"
                ))
            except Exception as e:
                logger.error(f"Ошибка обработки {ticker}: {e}")
                recommendations.append(Recommendation(
                    ticker=ticker,
                    action="HOLD",
                    confidence=0.3,
                    expected_return=0.0,
                    reasoning=f"Ошибка: {str(e)[:100]}"
                ))
    
    # Сортируем рекомендации: сначала BUY, потом HOLD, потом SELL
    action_order = {"BUY": 0, "HOLD": 1, "SELL": 2}
    recommendations.sort(key=lambda r: (action_order.get(r.action, 1), -r.confidence))
    
    # Расчет общей стоимости портфеля
    total_value = request.cash
    for pos in request.positions:
        rec = next((r for r in recommendations if r.ticker == pos.ticker), None)
        if rec and rec.current_price:
            total_value += pos.shares * rec.current_price
    
    # Формируем ответ
    response = PortfolioResponse(
        recommendations=recommendations,
        total_value=round(total_value, 2),
        model_version="1.0.0" if predictor.model_loaded else "fallback"
    )
    
    logger.info(f"Сгенерировано {len(recommendations)} рекомендаций")
    return response


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Проверка статуса сервиса.
    
    Возвращает информацию о загруженности модели и дате обучения.
    """
    predictor = get_predictor()
    
    # Пытаемся загрузить модель если еще не загружена
    if not predictor.model_loaded:
        predictor.load_model()
    
    return HealthResponse(
        status="ok" if predictor.model_loaded else "degraded",
        model_loaded=predictor.model_loaded,
        model_trained_date=predictor.model_trained_date,
        version="1.0.0"
    )


@router.get("/tickers")
async def get_available_tickers():
    """
    Получить список доступных для анализа тикеров.
    
    Возвращает тикеры из конфигурации (обученные в модели).
    """
    return {
        "tickers": settings.default_tickers,
        "count": len(settings.default_tickers)
    }
