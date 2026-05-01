# Инструкция по сбору и подготовке данных для инвестиционного советника

## 📋 Обзор датасетов

Вот какие датасеты ты нашел и какие из них НУЖНЫ:

| Датасет | Ссылка | Статус | Что содержит |
|---------|--------|--------|--------------|
| **Kaggle: Russia Stocks Prices OHLCV** | [olegshpagin/russia-stocks-prices-ohlcv](https://www.kaggle.com/datasets/olegshpagin/russia-stocks-prices-ohlcv/data) | ✅ **ОСНОВНОЙ** | Дневные свечи (OHLCV) по крупным российским тикерам |
| **GitHub: moex-dataset** | [foykes/moex-dataset](https://github.com/foykes/moex-dataset) | ⚠️ ДОПОЛНИТЕЛЬНЫЙ | Альтернативные данные MOEX (можно использовать для проверки) |
| **HuggingFace: RussianFinancialNews** | [Kasymkhan/RussianFinancialNews](https://huggingface.co/datasets/Kasymkhan/RussianFinancialNews) | ✅ **ОСНОВНОЙ** | Финансовые новости с разметкой |
| **GitHub: financial-news-sentiment** | [WebOfRussia/financial-news-sentiment](https://github.com/WebOfRussia/financial-news-sentiment) | ⚠️ ДОПОЛНИТЕЛЬНЫЙ | Датасет для анализа тональности (можно использовать для обучения NLP) |
| **GitHub: RFSD** | [irlcode/RFSD](https://github.com/irlcode/RFSD) | ✅ **ОСНОВНОЙ** | Финансовая отчетность компаний (фундаментальные данные) |

---

## 🚀 Пошаговая инструкция

### Шаг 1: Подготовка окружения

```bash
# Перейди в директорию проекта
cd /workspace/investment-advisor/backend

# Установи зависимости
pip install -r requirements.txt

# Дополнительно для работы с датасетами
pip install kagglehub huggingface_hub datasets pandas pyarrow
```

### Шаг 2: Настройка Kaggle API (если еще не настроено)

1. Зарегистрируйся на https://www.kaggle.com/
2. Перейди в Account → Create New API Token
3. Скачается файл `kaggle.json`
4. Помести его в `~/.kaggle/kaggle.json`
5. Установи права: `chmod 600 ~/.kaggle/kaggle.json`

### Шаг 3: Автоматическая загрузка всех датасетов

**Запусти единый скрипт загрузки:**

```bash
cd /workspace/investment-advisor/backend
python download_datasets.py
```

Этот скрипт автоматически:
- ✅ Скачает OHLCV данные с Kaggle (папка D1 с дневными свечами)
- ✅ Загрузит новости RussianFinancialNews с HuggingFace
- ✅ Скачает макроэкономические данные
- ✅ Загрузит RFSD (фундаментальные данные)
- ✅ Склонирует GitHub репозитории (moex-dataset, financial-news-sentiment)
- ✅ Создаст файл `ticker_list.csv` со списком всех доступных тикеров

### Шаг 4: Проверка загруженных данных

После выполнения скрипта проверь структуру:

```
/workspace/investment-advisor/backend/data/
├── raw/
│   ├── SBER_D1.csv              # OHLCV данные по тикеру SBER
│   ├── GAZP_D1.csv              # OHLCV данные по тикеру GAZP
│   ├── ...                      # Другие тикеры
│   ├── Kasymkhan_RussianFinancialNews.parquet  # Новости
│   ├── rfsd_2023.parquet        # Фундаментальные данные
│   ├── russian_investment.csv   # Макроэкономика
│   ├── ticker_list.csv          # Список всех тикеров
│   ├── moex-dataset/            # Клонированный репозиторий
│   └── financial-news-sentiment/ # Клонированный репозиторий
└── features/                    # Сюда будут сохраняться обработанные данные
```

---

## 📊 Что скачивать (детали)

### 1. OHLCV Данные (ОБЯЗАТЕЛЬНО)

**Источник:** Kaggle - olegshpagin/russia-stocks-prices-ohlcv

**Что скачивать:**
- Папка `D1` - дневные свечи (основной формат для обучения)
- Файлы вида `TICKER_D1.csv` (например, `SBER_D1.csv`, `GAZP_D1.csv`)

**Структура файла:**
```csv
<date>,<open>,<high>,<low>,<close>,<vol>
2020-01-01,100.5,102.0,99.8,101.2,1000000
```

**Автоматическая загрузка:**
```python
import kagglehub
path = kagglehub.dataset_download("olegshpagin/russia-stocks-prices-ohlcv")
```

### 2. Новости (ОБЯЗАТЕЛЬНО)

**Источник:** HuggingFace - Kasymkhan/RussianFinancialNews

**Что скачивать:**
- Файлы `data/train-00000-of-00001.parquet`
- Файлы `data/test-00000-of-00001.parquet`

**Структура:**
```python
{
    'title': str,      # Заголовок новости
    'text': str,       # Текст новости
    'published': str,  # Дата публикации
    'ticker': str,     # Связанный тикер (если есть)
}
```

**Автоматическая загрузка:**
```python
from huggingface_hub import hf_hub_download
train_path = hf_hub_download(
    repo_id="Kasymkhan/RussianFinancialNews",
    filename="data/train-00000-of-00001.parquet",
    repo_type="dataset"
)
```

### 3. Фундаментальные данные (ОБЯЗАТЕЛЬНО)

**Источник:** HuggingFace - irlspbru/RFSD

**Что скачивать:**
- Датасет "2023" (последний доступный год)

**Структура:**
```python
{
    'ticker': str,           # Тикер компании
    'financial_year_end': str, # Дата окончания финансового года
    'revenue': float,        # Выручка
    'net_income': float,     # Чистая прибыль
    'total_assets': float,   # Активы
    'equity': float,         # Собственный капитал
    # ... другие финансовые показатели
}
```

**Автоматическая загрузка:**
```python
from datasets import load_dataset
ds = load_dataset("irlspbru/RFSD", "2023", split="train")
```

### 4. Макроэкономические данные (ОБЯЗАТЕЛЬНО)

**Источник:** Kaggle - demirtry/russian-investment-activity

**Что скачивать:**
- Файл `russian_investment.csv`

**Структура:**
```csv
date,key_rate,inflation,usd_rub,gdp_growth,...
2020-01-01,7.75,2.5,63.0,2.2,...
```

### 5. Дополнительные данные (ОПЦИОНАЛЬНО)

**moex-dataset (GitHub):**
```bash
git clone https://github.com/foykes/moex-dataset.git
```
Используй для верификации данных или заполнения пробелов.

**financial-news-sentiment (GitHub):**
```bash
git clone https://github.com/WebOfRussia/financial-news-sentiment.git
```
Используй для дообучения модели анализа тональности.

---

## 🔧 Обработка данных

После загрузки всех данных запусти пайплайн обработки:

```bash
cd /workspace/investment-advisor/backend
python data_pipeline.py --start-date 2020-01-01 --end-date 2024-12-31
```

**Что делает пайплайн:**
1. Читает все OHLCV файлы из `data/raw/*_D1.csv`
2. Объединяет их в единый DataFrame
3. Добавляет фундаментальные данные (RFSD)
4. Обрабатывает новости и рассчитывает сентимент
5. Добавляет макроэкономические показатели
6. Сохраняет готовые данные в `data/features/`

---

## 📁 Итоговая структура данных

```
data/
├── raw/                          # Сырые данные (не менять)
│   ├── *_D1.csv                  # OHLCV по тикерам
│   ├── Kasymkhan_RussianFinancialNews.parquet
│   ├── rfsd_2023.parquet
│   ├── russian_investment.csv
│   ├── ticker_list.csv
│   └── [github repos]/
│
└── features/                     # Обработанные данные (для ML)
    ├── ohlcv_fundamentals_YYYYMMDD_YYYYMMDD.parquet
    ├── news_sentiment_YYYYMMDD_YYYYMMDD.parquet
    └── macro_indicators_YYYYMMDD_YYYYMMDD.parquet
```

---

## 🎯 Рекомендации

### Что использовать для обучения:

1. **Основной датасет:** OHLCV + RFSD + Новости
   - OHLCV: ценовые паттерны, объемы
   - RFSD: фундаментальные показатели (P/E, P/B, ROE и т.д.)
   - Новости: сентимент-анализ

2. **Дополнительно:** Макроэкономика
   - Ключевая ставка ЦБ
   - Курс USD/RUB
   - Инфляция
   - Цены на нефть

### Что можно пропустить:

- **moex-dataset**: если хватает данных из Kaggle OHLCV
- **financial-news-sentiment**: если используешь готовую модель (FinBERT/rubert-sentiment)

### Формат данных для обучения:

Создай единый DataFrame с колонками:

```python
{
    'date': datetime,           # Дата
    'ticker': str,              # Тикер
    'open': float,              # Цена открытия
    'high': float,              # Максимум
    'low': float,               # Минимум
    'close': float,             # Цена закрытия
    'volume': float,            # Объем
    
    # Технические индикаторы (рассчитываются)
    'sma_20': float,            # Скользящее среднее 20
    'rsi_14': float,            # RSI 14
    'macd': float,              # MACD
    
    # Фундаментальные данные
    'pe_ratio': float,          # P/E
    'pb_ratio': float,          # P/B
    'roe': float,               # ROE
    'revenue': float,           # Выручка
    
    # Сентимент
    'daily_sentiment_mean': float,  # Средний сентимент за день
    'daily_news_count': int,        # Количество новостей
    
    # Макро
    'key_rate': float,        # Ключевая ставка
    'usd_rub': float,         # Курс доллара
    'inflation': float,       # Инфляция
    
    # Target (что предсказываем)
    'target_return_1d': float,    # Доходность завтра
    'target_direction': int       # Направление (0/1)
}
```

---

## 🛠 Пример кода для подготовки данных

```python
import pandas as pd
from pathlib import Path

# Загружаем OHLCV
raw_dir = Path("data/raw")
ohlcv_files = list(raw_dir.glob("*_D1.csv"))

all_dfs = []
for file in ohlcv_files:
    ticker = file.stem.replace("_D1", "")
    df = pd.read_csv(file)
    df['ticker'] = ticker
    all_dfs.append(df)

ohlcv_df = pd.concat(all_dfs, ignore_index=True)

# Загружаем новости
news_df = pd.read_parquet(raw_dir / "Kasymkhan_RussianFinancialNews.parquet")

# Загружаем фундаментальные данные
fund_df = pd.read_parquet(raw_dir / "rfsd_2023.parquet")

# Загружаем макро
macro_df = pd.read_csv(raw_dir / "russian_investment.csv")

# Теперь объединяй и обрабатывай...
```

---

## ❓ Частые проблемы

**Проблема:** Kaggle API не работает  
**Решение:** Проверь что `~/.kaggle/kaggle.json` существует и имеет права 600

**Проблема:** HuggingFace не загружает датасет  
**Решение:** Попробуй прямую загрузку через `hf_hub_download()` вместо `load_dataset()`

**Проблема:** Несоответствие форматов дат  
**Решение:** Используй `pd.to_datetime(date_col, format='mixed')`

**Проблема:** Мало данных по некоторым тикерам  
**Решение:** Дозагрузи через MOEX API (реализовано в `data_pipeline.py`)

---

## 📈 Следующие шаги

1. ✅ Загрузи все датасеты через `download_datasets.py`
2. ✅ Запусти обработку через `data_pipeline.py`
3. ✅ Рассчитай технические индикаторы
4. ✅ Подготовь финальный датасет для обучения
5. ✅ Обучи модель (`scripts/train_model.py`)

Удачи! 🚀
