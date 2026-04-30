# 📈 Investment Advisor - Система инвестиционных рекомендаций

ML-система для анализа портфеля акций и генерации рекомендаций на основе технических индикаторов, машинного обучения и NLP-анализа новостей.

## 🎯 Возможности

- **Технический анализ**: Расчет 20+ технических индикаторов (RSI, MACD, SMA, EMA, ATR и др.) через библиотеку `finta`
- **ML модель**: LightGBM классификатор обученный на исторических данных S&P500
- **NLP сентимент**: Анализ тональности новостей через FinBERT
- **REST API**: FastAPI backend с автоматической документацией
- **Веб-интерфейс**: Современный UI на Next.js + React + TailwindCSS

## 🏗️ Архитектура

```
investment-advisor/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entrypoint
│   │   ├── models/schemas.py    # Pydantic модели
│   │   ├── services/
│   │   │   ├── data_loader.py   # Загрузка данных (yfinance)
│   │   │   ├── feature_engine.py # Технические индикаторы (finta)
│   │   │   ├── sentiment.py     # NLP анализ (FinBERT)
│   │   │   └── predictor.py     # ML инференс
│   │   └── api/routes.py        # API endpoints
│   ├── ml_pipeline/
│   │   └── train.py             # Скрипт обучения модели
│   ├── config.py                # Конфигурация
│   └── requirements.txt
├── frontend/
│   ├── app/
│   │   ├── page.tsx             # Главная страница
│   │   ├── layout.tsx           # Layout компонент
│   │   └── globals.css          # Глобальные стили
│   └── package.json
├── data/                        # Данные и кэш
├── models/                      # Сохраненные модели
└── README.md
```

## 🚀 Полная инструкция по запуску

### Предварительные требования

- Python 3.10 или выше
- Node.js 18+ и npm
- pip (Python package manager)

### Шаг 1: Установка зависимостей Backend

```bash
cd investment-advisor/backend
pip install -r requirements.txt
```

**Важно:** Используется библиотека `finta` вместо `pandas-ta` (оригинальный репозиторий pandas-ta был удалён автором).

Основные пакеты:
- `fastapi`, `uvicorn` - веб-сервер
- `finta` - технические индикаторы
- `lightgbm`, `scikit-learn` - ML модели
- `transformers`, `torch` - NLP (FinBERT)
- `yfinance` - загрузка финансовых данных

### Шаг 2: Создание .env файла (опционально)

```bash
cd investment-advisor/backend
cp .env.example .env
```

Файл `.env` не обязателен - все параметры имеют значения по умолчанию в `config.py`.

### Шаг 3: Генерация данных (если yfinance не работает)

**Важно:** Если вы видите ошибки `yfinance - ERROR - Failed to get ticker` при обучении, используйте генератор синтетических данных:

```bash
cd investment-advisor
python backend/ml_pipeline/generate_synthetic_data.py
```

Это создаст реалистичные данные для 30 тикеров S&P500 за период 2017-2024 и сохранит их в кэш.

**Почему это нужно?** Yahoo Finance может блокировать запросы из-за:
- Частых запросов (rate limiting)
- Отсутствия заголовков User-Agent
- Блокировки по региону/провайдеру
- Нестабильности API

Синтетические данные используют геометрическое броуновское движение с реалистичными параметрами (доходность, волатильность, тренды по секторам).

### Шаг 4: Обучение модели

Обучение ML модели на исторических данных (занимает 5-15 минут):

```bash
cd investment-advisor
python backend/ml_pipeline/train.py
```

Что происходит при обучении:
1. Загружаются данные по 30 тикерам за 2018-2024 через yfinance
2. Рассчитываются технические индикаторы (SMA, EMA, RSI, MACD, ATR, etc.)
3. Добавляются макро-факторы (VIX, инфляция, ставки) или заглушки
4. Создается панельный DataFrame [date, ticker, features..., target]
5. Разбиение: Train (2018-2021), Val (2022), Test (2023-2024)
6. Обучается LightGBM с early stopping
7. Модель сохраняется в `models/lgb_portfolio.joblib`
8. Метрики и feature importance сохраняются в JSON

После успешного обучения вы увидите:
```
ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!
Модель сохранена в .../models/lgb_portfolio.joblib
```

### Шаг 5: Запуск Backend (FastAPI)

```bash
cd investment-advisor
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend запустится на: **http://localhost:8000**

Проверка работы:
- Swagger документация: http://localhost:8000/docs
- Health check: http://localhost:8000/api/v1/health

### Шаг 6: Запуск Frontend (Next.js)

Откройте новый терминал:

```bash
cd investment-advisor/frontend
npm install
npm run dev
```

Frontend запустится на: **http://localhost:3000**

## 📡 API Endpoints

### POST /api/v1/recommendations

Получить рекомендации по портфелю.

**Request:**
```json
{
  "cash": 10000.0,
  "positions": [
    {"ticker": "AAPL", "shares": 50},
    {"ticker": "MSFT", "shares": 30}
  ]
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "ticker": "AAPL",
      "action": "BUY",
      "confidence": 0.78,
      "expected_return": 5.2,
      "reasoning": "Модель прогнозирует вероятность роста цены. RSI (позитивно), MACD (позитивно)",
      "current_price": 185.50
    }
  ],
  "total_value": 125000.0,
  "model_version": "1.0.0"
}
```

### GET /api/v1/health

Проверка статуса сервиса:

```bash
curl http://localhost:8000/api/v1/health
```

Ответ:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_trained_date": "2024-01-15T10:30:00",
  "version": "1.0.0"
}
```

### GET /api/v1/tickers

Список доступных тикеров:

```bash
curl http://localhost:8000/api/v1/tickers
```

## 🔧 Конфигурация

Все настройки находятся в `backend/config.py`. Основные параметры:

| Параметр | Значение по умолчанию | Описание |
|----------|----------------------|----------|
| `default_tickers` | 30 тикеров S&P500 | Список для обучения |
| `prediction_horizon` | 1 день | Горизонт прогноза |
| `confidence_threshold` | 0.5 | Мин. уверенность |
| `train_start_date` | 2018-01-01 | Начало train периода |
| `train_end_date` | 2021-12-31 | Конец train периода |
| `lgb_num_estimators` | 1000 | Количество деревьев |
| `lgb_learning_rate` | 0.05 | Скорость обучения |

## 📊 Как это работает

### 1. Сбор данных
- Исторические цены через yfinance (OHLCV данные)
- Макроэкономические показатели (инфляция, ставки, VIX) - при недоступности используются forward-fill медианные значения
- Новости для анализа сентимента (через FinBERT)

### 2. Feature Engineering (finta)
Библиотека `finta` рассчитывает:
- **SMA** (Simple Moving Average) - трендовые индикаторы за 20, 50, 200 дней
- **EMA** (Exponential Moving Average) - экспоненциальные скользящие средние
- **RSI** (Relative Strength Index) - осциллятор перекупленности/перепроданности
- **MACD** (Moving Average Convergence Divergence) - индикатор импульса
- **ATR** (Average True Range) - мера волатильности
- **Log Returns** - логарифмическая доходность
- **Volume Ratio** - отношение объема к среднему
- **Price Deviations** - отклонение цены от SMA

### 3. ML Модель
- **Алгоритм**: LightGBM Classifier (градиентный бустинг)
- **Целевая переменная**: бинарная (1 = цена завтра вырастет, 0 = упадет)
- **Разбиение**: временное (без shuffle!) - Train (2018-2021), Val (2022), Test (2023-2024)
- **Early stopping**: 50 раундов для предотвращения переобучения
- **Метрики**: Accuracy, Precision, Recall, F1-Score

### 4. Генерация рекомендаций
Для каждого тикера в портфеле:
1. Загружаются последние 90 дней цен
2. Рассчитываются признаки (как при обучении)
3. Анализируются последние 10 новостей (FinBERT)
4. Модель делает предсказание + probability
5. Confidence = max(prob, 1-prob), нормализованный к [0.3, 1.0]
6. Топ-3 признака по feature importance определяют обоснование
7. Финальная рекомендация:
   - **BUY** если prediction=1 и confidence > 0.6
   - **SELL** если prediction=0 и confidence > 0.6
   - **HOLD** иначе

## 📈 Метрики модели

После обучения в консоли и файле `models/metrics.json`:

```json
{
  "accuracy": 0.52,
  "precision": 0.54,
  "recall": 0.48,
  "f1": 0.51
}
```

Предсказание направления цены - сложная задача, поэтому метрики ~50-55% являются нормальными для эффективного рынка.

## 🧪 Примеры использования

### cURL запрос

```bash
curl -X POST http://localhost:8000/api/v1/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "cash": 50000,
    "positions": [
      {"ticker": "AAPL", "shares": 100},
      {"ticker": "GOOGL", "shares": 50},
      {"ticker": "TSLA", "shares": 75}
    ]
  }'
```

### Python клиент

```python
import requests

response = requests.post(
    'http://localhost:8000/api/v1/recommendations',
    json={
        'cash': 50000,
        'positions': [
            {'ticker': 'AAPL', 'shares': 100},
            {'ticker': 'MSFT', 'shares': 50}
        ]
    }
)

print(response.json())
```

### Проверка здоровья API

```bash
curl http://localhost:8000/api/v1/health
```

## ⚠️ Возможные ошибки и решения

### Ошибка: "ModuleNotFoundError: No module named 'finta'"
**Решение**: `pip install finta==1.3`

### Ошибка: "Model file not found"
**Решение**: Запустите генерацию данных и обучение:
```bash
python backend/ml_pipeline/generate_synthetic_data.py
python backend/ml_pipeline/train.py
```

### Ошибка: yfinance не загружает данные
**Решение**: Используйте генератор синтетических данных:
```bash
python backend/ml_pipeline/generate_synthetic_data.py
```

### Ошибка: "Port 8000 already in use"
**Решение**: Используйте другой порт: `uvicorn backend.app.main:app --port 8001`

### Frontend не запускается
**Решение**: 
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

## ⚠️ Отказ от ответственности

Этот проект создан **в учебных целях**. Не используйте эти рекомендации для реальной торговли без дополнительного анализа и консультации с финансовыми советниками. Прошлые результаты не гарантируют будущую доходность.

## 📝 Лицензия

MIT License - свободное использование в учебных и исследовательских целях.

## 🤝 Вклад

Проект открыт для улучшений:
- Добавление новых источников данных (Quandl, Alpha Vantage)
- Улучшение фичей (дополнительные индикаторы, альтернативные данные)
- Другие модели (XGBoost, нейросети, ансамбли)
- Расширение функционала фронтенда (графики, история, backtesting)
