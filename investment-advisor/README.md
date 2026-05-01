# 📈 Investment Advisor - Система инвестиционных рекомендаций

ML-система для анализа портфеля акций и генерации рекомендаций на основе технических индикаторов, машинного обучения (CatBoost) и анализа новостей. Использует реальные данные с Московской Биржи (MOEX).

## 🎯 Возможности

- **Технический анализ**: Расчет 20+ технических индикаторов (RSI, MACD, SMA, EMA, ATR и др.)
- **ML модель**: CatBoost классификатор обученный на исторических данных российских акций
- **Прогнозирование временных рядов**: Prophet для создания дополнительных фичей
- **Макроэкономические данные РФ**: Ключевая ставка ЦБ, инфляция, курс USD/RUB, цена нефти Brent
- **REST API**: FastAPI backend с автоматической документацией
- **Веб-интерфейс**: Современный UI на Next.js + React + TailwindCSS с автокомплитом тикеров

## 🏗️ Архитектура

```
investment-advisor/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entrypoint
│   │   ├── models/schemas.py    # Pydantic модели
│   │   ├── services/
│   │   │   ├── data_loader.py   # Загрузка данных (MOEX, макро)
│   │   │   ├── feature_engine.py # Технические индикаторы
│   │   │   ├── sentiment.py     # Анализ тональности новостей
│   │   │   └── predictor.py     # ML инференс
│   │   └── api/routes.py        # API endpoints
│   ├── config.py                # Конфигурация
│   └── requirements.txt
├── scripts/
│   ├── collect_data.py          # Сбор данных с MOEX
│   └── train_model.py           # Обучение модели (CatBoost)
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

**Важно:** Используются следующие ключевые пакеты:
- `fastapi`, `uvicorn` - веб-сервер
- `catboost` - ML модель (вместо LightGBM)
- `prophet` - прогнозирование временных рядов
- `pandas`, `numpy` - работа с данными
- `requests` - HTTP запросы к MOEX API
- `feedparser` - парсинг RSS новостей

### Шаг 2: Сбор данных с Московской Биржи

Сбор исторических данных OHLCV для российских акций через MOEX ISS API:

```bash
cd investment-advisor
python scripts/collect_data.py --tickers SBER,GAZP,LKOH,NVTK,YNDX --start-date 2020-01-01 --end-date 2024-12-31
```

Параметры:
- `--tickers`: Список тикеров через запятую (по умолчанию все российские из конфига)
- `--start-date`: Дата начала в формате YYYY-MM-DD
- `--end-date`: Дата окончания в формате YYYY-MM-DD
- `--no-cache`: Не использовать кэш
- `--output-dir`: Директория для сохранения данных

Данные сохраняются в кэш (`data/cache/`) для повторного использования.

**Источники данных:**
- MOEX ISS API - котировки акций (OHLCV)
- ЦБ РФ API - макроэкономические показатели (ключевая ставка)
- Росстат - инфляция, ВВП
- РБК RSS - новости для анализа сентимента

### Шаг 3: Обучение модели

Обучение CatBoost модели на собранных данных (занимает 5-15 минут):

```bash
cd investment-advisor
python scripts/train_model.py --tickers SBER,GAZP,LKOH,NVTK,YNDX
```

Что происходит при обучении:
1. Загружаются данные из кэша или собираются заново
2. Рассчитываются технические индикаторы (SMA, EMA, RSI, MACD, ATR, etc.)
3. Добавляются макро-факторы РФ (ключевая ставка, инфляция, USD/RUB, Brent)
4. Создаются прогнозные фичи через Prophet
5. Создается панельный DataFrame [date, ticker, features..., target]
6. Разбиение: Train (2019-2021), Val (2022), Test (2023-2024)
7. Обучается CatBoost с early stopping
8. Модель сохраняется в `models/catboost_portfolio.joblib`
9. Метрики и feature importance сохраняются в JSON

После успешного обучения вы увидите:
```
ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!
Модель сохранена в .../models/catboost_portfolio.joblib
```

### Шаг 4: Запуск Backend (FastAPI)

```bash
cd investment-advisor
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend запустится на: **http://localhost:8000**

Проверка работы:
- Swagger документация: http://localhost:8000/docs
- Health check: http://localhost:8000/api/v1/health

### Шаг 5: Запуск Frontend (Next.js)

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
  "cash": 100000.0,
  "positions": [
    {"ticker": "SBER", "shares": 100},
    {"ticker": "GAZP", "shares": 200}
  ]
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "ticker": "SBER",
      "action": "BUY",
      "confidence": 0.78,
      "expected_return": 5.2,
      "reasoning": "Модель прогнозирует вероятность роста цены. RSI (позитивно), MACD (позитивно)",
      "current_price": 285.50,
      "predicted_price": 298.00
    }
  ],
  "portfolio_analysis": {
    "total_value": 125000.0,
    "predicted_value": 132000.0,
    "recommended_allocation": [...]
  },
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
  "version": "2.0.0"
}
```

### GET /api/v1/tickers

Список доступных российских тикеров:

```bash
curl http://localhost:8000/api/v1/tickers
```

## 🔧 Конфигурация

Все настройки находятся в `backend/config.py`. Основные параметры:

| Параметр | Значение по умолчанию | Описание |
|----------|----------------------|----------|
| `default_tickers` | 24 российских тикера | Список для обучения (SBER, GAZP, LKOH...) |
| `prediction_horizon` | 1 день | Горизонт прогноза |
| `confidence_threshold` | 0.5 | Мин. уверенность |
| `train_start_date` | 2019-01-01 | Начало train периода |
| `train_end_date` | 2021-12-31 | Конец train периода |
| `catboost_iterations` | 1000 | Количество деревьев |
| `catboost_learning_rate` | 0.05 | Скорость обучения |

## 📊 Как это работает

### 1. Сбор данных
- Исторические цены через MOEX ISS API (OHLCV данные)
- Макроэкономические показатели РФ (ключевая ставка ЦБ, инфляция, USD/RUB, Brent)
- Новости через RSS ленты (РБК, Коммерсант) для анализа сентимента

### 2. Feature Engineering
Рассчитываются следующие признаки:
- **SMA** (Simple Moving Average) - трендовые индикаторы за 20, 50, 200 дней
- **EMA** (Exponential Moving Average) - экспоненциальные скользящие средние
- **RSI** (Relative Strength Index) - осциллятор перекупленности/перепроданности
- **MACD** (Moving Average Convergence Divergence) - индикатор импульса
- **ATR** (Average True Range) - мера волатильности
- **Log Returns** - логарифмическая доходность
- **Volume Ratio** - отношение объема к среднему
- **Price Deviations** - отклонение цены от SMA
- **Prophet Forecast** - прогноз тренда, неопределенность

### 3. ML Модель
- **Алгоритм**: CatBoost Classifier (градиентный бустинг на деревьях решений)
- **Целевая переменная**: бинарная (1 = цена завтра вырастет, 0 = упадет)
- **Разбиение**: временное (без shuffle!) - Train (2019-2021), Val (2022), Test (2023-2024)
- **Early stopping**: 50 раундов для предотвращения переобучения
- **Метрики**: Accuracy, Precision, Recall, F1-Score, AUC

### 4. Генерация рекомендаций
Для каждого тикера в портфеле:
1. Загружаются последние 90 дней цен
2. Рассчитываются признаки (как при обучении)
3. Анализируются последние новости
4. Модель делает предсказание + probability
5. Confidence = max(prob, 1-prob), нормализованный к [0.3, 1.0]
6. Топ-3 признака по feature importance определяют обоснование
7. Финальная рекомендация:
   - **BUY** если prediction=1 и confidence > 0.6
   - **SELL** если prediction=0 и confidence > 0.6
   - **HOLD** иначе
8. Прогнозируется цена акции и стоимость портфеля

## 📈 Метрики модели

После обучения в консоли и файле `models/metrics.json`:

```json
{
  "accuracy": 0.54,
  "precision": 0.56,
  "recall": 0.52,
  "f1": 0.54
}
```

Предсказание направления цены - сложная задача, поэтому метрики ~50-58% являются нормальными для эффективного рынка.

## 🧪 Примеры использования

### cURL запрос

```bash
curl -X POST http://localhost:8000/api/v1/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "cash": 500000,
    "positions": [
      {"ticker": "SBER", "shares": 100},
      {"ticker": "GAZP", "shares": 200},
      {"ticker": "LKOH", "shares": 50}
    ]
  }'
```

### Python клиент

```python
import requests

response = requests.post(
    'http://localhost:8000/api/v1/recommendations',
    json={
        'cash': 500000,
        'positions': [
            {'ticker': 'SBER', 'shares': 100},
            {'ticker': 'GAZP', 'shares': 200}
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

### Ошибка: "ModuleNotFoundError: No module named 'catboost'"
**Решение**: `pip install catboost==1.2.3`

### Ошибка: "Model file not found"
**Решение**: Запустите сбор данных и обучение:
```bash
python scripts/collect_data.py
python scripts/train_model.py
```

### Ошибка: "No data for ticker" при сборе данных
**Решение**: Проверьте правильность тикера (российские тикеры пишутся заглавными буквами, например SBER, GAZP). Убедитесь что есть интернет соединение для доступа к MOEX API.

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

## 🇷🇺 Доступные российские тикеры

По умолчанию используются следующие тикеры:
- **SBER** - Сбербанк
- **GAZP** - Газпром
- **LKOH** - Лукойл
- **NVTK** - Новатэк
- **YNDX** - Яндекс
- **TCSG** - Тинькофф
- **VTBR** - ВТБ
- **ROSN** - Роснефть
- **GMKN** - Норникель
- **NLMK** - НЛМК
- **SNGS** - Сургутнефтегаз
- **HYDR** - РусГидро
- **FEES** - ФСК ЕЭС
- **TRNFP** - Транснефть
- **MTSS** - МТС
- **AFKS** - АФК Система
- **PIKK** - ПИК
- **CHMF** - Северсталь
- **MAGN** - Магнитка
- **RTKM** - Ростелеком
- **BSPB** - Банк Санкт-Петербург
- **VKCO** - VK
- **OZON** - Ozon
- **SGZH** - Сахалинская энергия

## ⚠️ Отказ от ответственности

Этот проект создан **в учебных целях**. Не используйте эти рекомендации для реальной торговли без дополнительного анализа и консультации с финансовыми советниками. Прошлые результаты не гарантируют будущую доходность.

## 📝 Лицензия

MIT License - свободное использование в учебных и исследовательских целях.

## 🤝 Вклад

Проект открыт для улучшений:
- Подключение реального API ЦБ РФ и Росстата для макро-данных
- Парсинг Telegram-каналов для анализа сентимента
- Добавление фундаментальных показателей (выручка, прибыль, P/E)
- Улучшение фичей (дополнительные индикаторы, альтернативные данные)
- Другие модели (XGBoost, нейросети, ансамбли)
- Расширение функционала фронтенда (графики, история, backtesting)
