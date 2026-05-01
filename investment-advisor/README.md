# Investment Advisor AI 💰📈

**Интеллектуальная система для анализа и оптимизации инвестиционного портфеля на российском рынке акций**

## 🚀 Особенности

### ML-модель нового поколения
- **Ансамбль моделей**: LightGBM + GradientBoosting + RandomForest + LogisticRegression с soft voting
- **Расширенные признаки**: 35+ технических индикаторов (RSI, MACD, Bollinger Bands, Stochastic, ADX, ATR)
- **NLP анализ новостей**: FinBERT для анализа тональности финансовых новостей
- **Макроэкономические факторы**: ключевая ставка ЦБ, инфляция, курс USD/RUB, цена нефти Brent
- **Приоритет локальных данных**: сначала используются датасеты с Kaggle, затем API для актуализации

### Технологии
- **Backend**: FastAPI, Python 3.10+
- **Frontend**: Next.js 14, React, TypeScript, TailwindCSS
- **ML**: scikit-learn, LightGBM, transformers (FinBERT)
- **Данные**: MOEX ISS API, макроэкономические показатели РФ

## 📊 Источники данных

### Приоритетная загрузка:
1. **Локальные датасеты** (`/workspace/data/raw/`):
   - [Russia Stocks Prices OHLCV](https://www.kaggle.com/datasets/olegshpagin/russia-stocks-prices-ohlcv)
   - Другие CSV файлы с историческими данными

2. **API сервисы** (для дополнения):
   - MOEX ISS API - исторические данные по российским акциям
   - Макроэкономические показатели (ЦБ РФ, Росстат)
   - Финансовые новости (РБК, Smart-Lab)

## 🛠️ Установка

### Требования
- Python 3.10+
- Node.js 18+
- pip, npm

### Backend
```bash
cd investment-advisor/backend
pip install -r requirements.txt
```

### Frontend
```bash
cd investment-advisor/frontend
npm install
```

## 📁 Структура проекта

```
investment-advisor/
├── backend/
│   ├── app/
│   │   ├── api/          # API endpoints
│   │   ├── services/     # Бизнес-логика
│   │   │   ├── data_loader.py      # Загрузка данных (MOEX + кэш)
│   │   │   ├── feature_engine.py   # Расчет признаков
│   │   │   ├── sentiment.py        # NLP анализ (FinBERT)
│   │   │   └── predictor.py        # Инференс модели
│   │   ├── models/       # Pydantic схемы
│   │   └── main.py       # FastAPI приложение
│   ├── models/           # Обученные модели
│   └── config.py
├── scripts/
│   ├── collect_data.py   # Сбор данных (MOEX, новости, макро)
│   └── train_model.py    # ML pipeline обучения
├── frontend/
│   └── app/              # Next.js компоненты
└── data/
    ├── raw/              # Локальные датасеты (Kaggle)
    ├── cache/            # Кэш API запросов
    └── features/         # Подготовленные признаки
```

## 🎯 Использование

### 1. Сбор данных
```bash
# Скачать исторические данные, макро-показатели и новости
python scripts/collect_data.py --start-date 2020-01-01 --end-date 2024-12-31
```

### 2. Обучение модели
```bash
# Обучить ансамбль моделей с использованием всех признаков
python scripts/train_model.py --tickers SBER,GAZP,LKOH,NVTK,YNDX,TCSG,VTBR,ROSN,GMKN,NLMK
```

**Параметры:**
- `--tickers`: Список тикеров через запятую
- `--start-date`: Дата начала обучения
- `--end-date`: Дата окончания обучения
- `--horizon`: Горизонт прогнозирования (дни)

### 3. Запуск backend
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Запуск frontend
```bash
cd frontend
npm run dev
```

Откройте http://localhost:3000

## 📈 API Endpoints

### `GET /api/v1/tickers`
Список доступных тикеров для анализа.

### `POST /api/v1/recommendations`
Получить рекомендации по портфелю.

**Request:**
```json
{
  "positions": [
    {"ticker": "SBER", "shares": 100, "avg_price": 250.5},
    {"ticker": "GAZP", "shares": 50, "avg_price": 180.0}
  ],
  "cash": 50000
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
      "current_price": 265.3,
      "reasoning": "По акции SBER рекомендуется покупать. Модель прогнозирует рост...\n\nУверенность модели: высокая (78.0%).\nОжидаемая потенциальная доходность: +5.20%.\n\nКлючевые факторы:\n  1. ключевая ставка ЦБ: 16.00\n  2. 20-дневная волатильность: 0.02\n  3. сентимент новостей: +0.45\n\nСентимент финансовых новостей: позитивный (+0.45).\n\nМакроэкономическая ситуация:\n  - Ключевая ставка: 16.00%\n  - Нефть Brent: $85.2\n  - USD/RUB: 92.5"
    }
  ],
  "total_value": 125765.0,
  "model_version": "2.0.0"
}
```

### `GET /api/v1/health`
Проверка статуса сервиса и модели.

## 🧠 ML Pipeline

### Этапы обучения:
1. **Загрузка данных**: приоритет локальным датасетам → API
2. **Feature Engineering**:
   - Технические индикаторы (SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic, ADX, ATR)
   - Макроэкономические признаки с изменениями
   - NLP сентимент новостей через FinBERT
3. **Разбиение**: train (2019-2021), val (2022), test (2023-2024)
4. **Обучение ансамбля**:
   - LightGBM (вес 3)
   - GradientBoosting (вес 2)
   - RandomForest (вес 2)
   - Logistic Regression calibrated (вес 1)
5. **Оценка**: accuracy, precision, recall, F1, ROC-AUC
6. **Сохранение**: модель, scaler, метрики, feature importance

### Признаки модели (35+):
- **Технические**: sma_20/50/200, ema_12/26, rsi, macd*, atr, bb_*, stochastic_*, adx
- **Волатильность**: volatility_20d/60d, log_return
- **Моментум**: momentum_10d/20d, price_sma*_deviation
- **Объем**: volume_ratio, volume_ma_ratio
- **Макро**: key_rate*, inflation*, usd_rub*, brent* (+ изменения)
- **Новости**: news_sentiment, news_count_7d, news_sentiment_7d

## 📝 Лицензия

MIT License
