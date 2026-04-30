"""
FastAPI приложение: система инвестиционных рекомендаций.
Main entrypoint для запуска сервера.

Запуск:
    uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
from pathlib import Path

# Добавляем корень проекта в path для импортов
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.config import settings
from backend.app.api.routes import router
from backend.app.services.predictor import get_predictor

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager для приложения.
    Выполняет инициализацию при старте и очистку при остановке.
    """
    # Startup: загрузка модели
    logger.info("="*50)
    logger.info("ЗАПУСК ПРИЛОЖЕНИЯ INVESTMENT ADVISOR")
    logger.info("="*50)
    
    predictor = get_predictor()
    logger.info("Попытка загрузки ML модели...")
    
    if predictor.load_model():
        logger.info(f"Модель успешно загружена. Дата обучения: {predictor.model_trained_date}")
        logger.info(f"Количество признаков: {len(predictor.feature_columns)}")
    else:
        logger.warning("Модель не найдена. Приложение будет работать в fallback режиме.")
        logger.warning("Для полноценной работы запустите: python backend/ml_pipeline/train.py")
    
    yield
    
    # Shutdown: очистка ресурсов
    logger.info("Остановка приложения...")
    logger.info("Приложение остановлено")


# Создание FastAPI приложения
app = FastAPI(
    title="Investment Advisor API",
    description="""
## Система инвестиционных рекомендаций на основе ML

Этот API предоставляет рекомендации по акциям используя:
- **Технический анализ**: RSI, MACD, SMA, EMA, ATR
- **ML модель**: LightGBM classifier обученный на исторических данных
- **NLP сентимент**: FinBERT для анализа тональности новостей

### Основные endpoints:
- **POST /api/v1/recommendations** - Получить рекомендации по портфелю
- **GET /api/v1/health** - Проверка статуса сервиса
- **GET /api/v1/tickers** - Список доступных тикеров
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Настройка CORS для фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://localhost:8000",  # Backend
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Глобальный обработчик ошибок
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Обработчик необработанных исключений."""
    logger.error(f"Необработанная ошибка: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Внутренняя ошибка сервера",
            "error_code": "INTERNAL_ERROR"
        }
    )


# Подключение роутов
app.include_router(router)


# Root endpoint
@app.get("/")
async def root():
    """
    Корневой endpoint с информацией о сервисе.
    """
    return {
        "service": "Investment Advisor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


# Health check для Kubernetes/load balancer
@app.get("/healthz")
async def healthz():
    """Простой health check для оркестраторов."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Запуск сервера на {settings.host}:{settings.port}")
    
    uvicorn.run(
        "backend.app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower()
    )
