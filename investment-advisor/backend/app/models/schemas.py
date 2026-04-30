"""
Pydantic схемы для API запросов и ответов.
Определяет структуру данных для валидации входных и выходных данных.
"""
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from datetime import datetime


class Position(BaseModel):
    """
    Позиция в портфеле пользователя.
    
    Attributes:
        ticker: Тикер акции (например, "AAPL")
        shares: Количество акций в портфеле
    """
    ticker: str = Field(..., description="Тикер акции", min_length=1, max_length=10)
    shares: int = Field(..., description="Количество акций", gt=0)


class PortfolioRequest(BaseModel):
    """
    Запрос на анализ портфеля.
    
    Attributes:
        cash: Доступная наличность для инвестиций
        positions: Список текущих позиций в портфеле
    """
    cash: float = Field(..., description="Доступная наличность", ge=0)
    positions: List[Position] = Field(..., description="Список позиций в портфеле")
    
    class Config:
        json_schema_extra = {
            "example": {
                "cash": 10000.0,
                "positions": [
                    {"ticker": "AAPL", "shares": 50},
                    {"ticker": "MSFT", "shares": 30}
                ]
            }
        }


class Recommendation(BaseModel):
    """
    Рекомендация по конкретной акции.
    
    Attributes:
        ticker: Тикер акции
        action: Рекомендуемое действие (BUY/SELL/HOLD)
        confidence: Уровень уверенности модели (0.0-1.0)
        expected_return: Ожидаемая доходность (в процентах)
        reasoning: Текстовое обоснование рекомендации
        current_price: Текущая цена акции (опционально)
        target_price: Целевая цена (опционально)
    """
    ticker: str = Field(..., description="Тикер акции")
    action: Literal["BUY", "SELL", "HOLD"] = Field(..., description="Рекомендуемое действие")
    confidence: float = Field(..., description="Уровень уверенности", ge=0.0, le=1.0)
    expected_return: float = Field(..., description="Ожидаемая доходность (%)")
    reasoning: str = Field(..., description="Обоснование рекомендации")
    current_price: Optional[float] = Field(None, description="Текущая цена акции")
    target_price: Optional[float] = Field(None, description="Целевая цена")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "action": "BUY",
                "confidence": 0.78,
                "expected_return": 5.2,
                "reasoning": "RSI указывает на перепроданность + позитивный сентимент новостей"
            }
        }


class PortfolioResponse(BaseModel):
    """
    Ответ API с рекомендациями по портфелю.
    
    Attributes:
        recommendations: Список рекомендаций по акциям
        total_value: Общая стоимость портфеля
        analysis_timestamp: Время анализа
        model_version: Версия используемой модели
    """
    recommendations: List[Recommendation] = Field(..., description="Список рекомендаций")
    total_value: Optional[float] = Field(None, description="Общая стоимость портфеля")
    analysis_timestamp: datetime = Field(default_factory=datetime.now, description="Время анализа")
    model_version: Optional[str] = Field(None, description="Версия модели")
    
    class Config:
        json_schema_extra = {
            "example": {
                "recommendations": [
                    {
                        "ticker": "AAPL",
                        "action": "BUY",
                        "confidence": 0.78,
                        "expected_return": 5.2,
                        "reasoning": "RSI указывает на перепроданность + позитивный сентимент"
                    }
                ],
                "total_value": 125000.0,
                "analysis_timestamp": "2024-01-15T10:30:00",
                "model_version": "1.0.0"
            }
        }


class HealthResponse(BaseModel):
    """
    Ответ endpoint здоровья сервиса.
    
    Attributes:
        status: Статус сервиса (ok/error)
        model_loaded: Загружена ли модель в память
        model_trained_date: Дата последнего обучения модели
        version: Версия API
    """
    status: Literal["ok", "error"] = Field(..., description="Статус сервиса")
    model_loaded: bool = Field(..., description="Загружена ли модель")
    model_trained_date: Optional[str] = Field(None, description="Дата обучения модели")
    version: str = Field(default="1.0.0", description="Версия API")


class ErrorResponse(BaseModel):
    """
    Стандартный формат ошибок API.
    
    Attributes:
        detail: Описание ошибки
        error_code: Код ошибки для программной обработки
    """
    detail: str = Field(..., description="Описание ошибки")
    error_code: Optional[str] = Field(None, description="Код ошибки")
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Ticker not found in training data",
                "error_code": "TICKER_NOT_FOUND"
            }
        }
