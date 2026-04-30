# 📈 Investment Advisor - Система инвестиционных рекомендаций

ML-система для анализа портфеля акций и генерации рекомендаций на основе технических индикаторов, машинного обучения и NLP-анализа новостей.

## 🎯 Возможности

- **Технический анализ**: Расчет 20+ технических индикаторов (RSI, MACD, SMA, EMA, ATR и др.)
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
│   │   │   ├── feature_engine.py # Технические индикаторы
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

## 🚀 Быстрый старт

### 1. Установка зависимостей

#### Backend
```bash
cd investment-advisor/backend
pip install -r requirements.txt
```

#### Frontend
```bash
cd investment-advisor/frontend
npm install
```

### 2. Обучение модели

Обучение ML модели на исторических данных (занимает 5-15 минут):

```bash
cd investment-advisor
python backend/ml_pipeline/train.py
```

После обучения модель сохраняется в `models/lgb_portfolio.joblib`

### 3. Запуск backend

```bash
cd investment-advisor
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

API будет доступно по адресу: http://localhost:8000
Документация Swagger: http://localhost:8000/docs

### 4. Запуск frontend

```bash
cd investment-advisor/frontend
npm run dev
```

Веб-интерфейс откроется по адресу: http://localhost:3000

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

### GET /api/v1/tickers

Список доступных тикеров:

```bash
curl http://localhost:8000/api/v1/tickers
```

## 🔧 Конфигурация

Скопируйте `.env.example` в `.env` и настройте параметры:

```bash
cp backend/.env.example backend/.env
```

Основные параметры:
- `MODEL_PATH` - путь к файлу модели
- `DEFAULT_TICKERS` - список тикеров для обучения
- `CONFIDENCE_THRESHOLD` - порог уверенности для рекомендаций
- `PREDICTION_HORIZON` - горизонт прогноза (дни)

## 📊 Как это работает

### 1. Сбор данных
- Исторические цены через yfinance
- Макроэкономические показатели (инфляция, ставки, VIX)
- Новости для анализа сентимента

### 2. Feature Engineering
- Технические индикаторы: SMA, EMA, RSI, MACD, ATR
- Логарифмическая доходность
- Объемные соотношения
- Отклонения от скользящих средних

### 3. ML Модель
- LightGBM Classifier
- Обучается предсказывать направление движения цены (вверх/вниз)
- Разбиение: Train (2018-2021), Val (2022), Test (2023-2024)
- Early stopping для предотвращения переобучения

### 4. Генерация рекомендаций
- Предсказание модели + confidence score
- Анализ сентимента новостей (FinBERT)
- Объяснение через feature importance
- Финальная рекомендация: BUY / HOLD / SELL

## 📈 Метрики модели

Модель оценивается по следующим метрикам:
- **Accuracy** - доля правильных предсказаний
- **Precision** - точность положительных предсказаний
- **Recall** - полнота обнаружения положительных случаев
- **F1-Score** - гармоническое среднее precision и recall

Метрики сохраняются в `models/metrics.json` после обучения.

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

## ⚠️ Отказ от ответственности

Этот проект создан в учебных целях. Не используйте эти рекомендации для реальной торговли без дополнительного анализа и консультации с финансовыми советниками. Прошлые результаты не гарантируют будущую доходность.

## 📝 Лицензия

MIT License - свободное использование в учебных и исследовательских целях.

## 🤝 Вклад

Проект открыт для улучшений:
- Добавление новых источников данных
- Улучшение фичей и моделей
- Оптимизация производительности
- Расширение функционала фронтенда
