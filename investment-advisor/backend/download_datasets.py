#!/usr/bin/env python3
"""
Скачивание и подготовка всех датасетов (Kaggle, Hugging Face, GitHub).

Источники:
- Kaggle: olegshpagin/russia-stocks-prices-ohlcv (OHLCV данные)
- Kaggle: demirtry/russian-investment-activity (макроэкономика)
- HuggingFace: Kasymkhan/RussianFinancialNews (новости)
- HuggingFace: irlspbru/RFSD (фундаментальные данные)
- GitHub: moex-dataset, financial-news-sentiment (дополнительно)

Использование:
    python download_datasets.py
"""
import logging
import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd
import kagglehub
from huggingface_hub import hf_hub_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_kaggle_dataset(identifier: str, extract_subfolder: str = None):
    """Скачивает Kaggle‑датасет. Если указан extract_subfolder, копирует его содержимое."""
    logger.info(f"Скачивание Kaggle: {identifier}")
    try:
        path = Path(kagglehub.dataset_download(identifier))
        logger.info(f"  -> {path}")
        
        if extract_subfolder:
            sub_path = path / extract_subfolder
            if sub_path.exists():
                for item in sub_path.iterdir():
                    dst = RAW_DIR / item.name
                    if item.is_file():
                        shutil.copy2(item, dst)
                    elif item.is_dir() and not dst.exists():
                        shutil.copytree(item, dst)
        else:
            for f in path.rglob("*.csv"):
                shutil.copy2(f, RAW_DIR / f.name)
                logger.info(f"  -> скопирован {f.name}")
        return path
    except Exception as e:
        logger.error(f"  -> ошибка: {e}")
        return None


def download_news_parquet():
    """Загружает новости из HuggingFace в формате parquet."""
    logger.info("Скачивание новостного датасета...")
    repo_id = "Kasymkhan/RussianFinancialNews"
    frames = []
    
    for split in ("train", "test"):
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"data/{split}-00000-of-00001.parquet",
                repo_type="dataset"
            )
            df = pd.read_parquet(local_path)
            frames.append(df)
            logger.info(f"  -> загружен {split}: {len(df)} записей")
        except Exception as e:
            logger.error(f"  -> ошибка загрузки {split}: {e}")
    
    if frames:
        full = pd.concat(frames, ignore_index=True)
        dst = RAW_DIR / "Kasymkhan_RussianFinancialNews.parquet"
        full.to_parquet(dst)
        logger.info(f"  -> сохранён в {dst}")
        return dst
    return None


def clone_github_repo(repo_url: str, target_dir: Path = None) -> Path:
    """Клонирует GitHub-репозиторий, если его ещё нет."""
    if target_dir is None:
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        target_dir = RAW_DIR / repo_name
    
    if target_dir.exists():
        logger.info(f"Репозиторий {repo_url} уже существует, пропускаем")
        return target_dir
    
    logger.info(f"Клонирование GitHub: {repo_url}")
    try:
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(target_dir)], check=True)
        logger.info(f"  -> {target_dir}")
        return target_dir
    except Exception as e:
        logger.error(f"  -> ошибка: {e}")
        return None


def download_rfsd():
    """Загружает фундаментальные данные RFSD."""
    logger.info("Скачивание RFSD (фундаментальные данные)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("irlspbru/RFSD", "2023", split="train", streaming=False)
        df = pd.DataFrame(ds)
        dst = RAW_DIR / "rfsd_2023.parquet"
        df.to_parquet(dst)
        logger.info(f"  -> сохранён в {dst}")
        return dst
    except Exception as e:
        logger.warning(f"  -> ошибка: {e}")
        return None


def create_ticker_list():
    """Создает ticker_list.csv из имён файлов *_D1.csv."""
    ticker_file = RAW_DIR / "ticker_list.csv"
    if ticker_file.exists():
        logger.info("ticker_list.csv уже существует")
        return ticker_file
    
    logger.info("Создание ticker_list.csv...")
    tickers = set()
    
    for f in RAW_DIR.glob("*_D1.csv"):
        ticker = f.stem.replace("_D1", "").upper()
        if ticker:
            tickers.add(ticker)
    
    if tickers:
        pd.DataFrame({"ticker": sorted(tickers)}).to_csv(ticker_file, index=False)
        logger.info(f"  -> создан с {len(tickers)} тикерами")
    else:
        fallback = ["SBER","GAZP","LKOH","NVTK","YNDX","TCSG","VTBR",
                    "ROSN","GMKN","NLMK","SNGS","HYDR","MTSS","CHMF","MAGN"]
        pd.DataFrame({"ticker": fallback}).to_csv(ticker_file, index=False)
        logger.info(f"  -> создан fallback список с {len(fallback)} тикерами")
    
    return ticker_file


def main():
    logger.info("=" * 60)
    logger.info("Загрузка всех необходимых датасетов")
    logger.info("=" * 60)

    # 1. OHLCV данные (дневные свечи) - ОСНОВНОЙ
    download_kaggle_dataset("olegshpagin/russia-stocks-prices-ohlcv", extract_subfolder="D1")

    # 2. Макроэкономика - ОСНОВНОЙ
    download_kaggle_dataset("demirtry/russian-investment-activity")

    # 3. Новости - ОСНОВНОЙ
    download_news_parquet()

    # 4. Фундаментальные данные - ОСНОВНОЙ
    download_rfsd()

    # 5. GitHub репозитории - ДОПОЛНИТЕЛЬНО
    clone_github_repo("https://github.com/foykes/moex-dataset.git")
    clone_github_repo("https://github.com/WebOfRussia/financial-news-sentiment.git")

    # 6. Создание списка тикеров
    create_ticker_list()

    logger.info("=" * 60)
    logger.info("✅ Загрузка датасетов завершена!")
    logger.info("=" * 60)
    logger.info("\nСледующий шаг: python data_pipeline.py --start-date 2020-01-01 --end-date 2024-12-31")


if __name__ == "__main__":
    main()