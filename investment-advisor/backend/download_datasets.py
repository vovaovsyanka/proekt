#!/usr/bin/env python3
"""
Скачивание и подготовка всех датасетов (Kaggle, Hugging Face, GitHub).
Новости загружаются напрямую из parquet-файлов репозитория.
"""
import logging
import shutil
import subprocess
import sys
import time
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
            csv_files = list(path.rglob("*.csv"))
            for f in csv_files:
                dst = RAW_DIR / f.name
                shutil.copy2(f, dst)
                logger.info(f"  -> скопирован {f.name}")
        return path
    except Exception as e:
        logger.error(f"  -> ошибка: {e}")
        return None


def download_news_parquet():
    """Загружает train и test датасеты новостей напрямую через huggingface_hub."""
    logger.info("Скачивание новостного датасета (прямые ссылки)...")
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


def clone_github_repo(repo_url: str, target_dir: str | None = None) -> Path | None:
    """Клонирует GitHub-репозиторий, если его ещё нет."""
    if target_dir is None:
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        target_dir = RAW_DIR / repo_name
    else:
        target_dir = Path(target_dir)
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


def main():
    logger.info("=" * 60)
    logger.info("Загрузка всех необходимых датасетов")
    logger.info("=" * 60)

    # 1. OHLCV (дневные свечи)
    download_kaggle_dataset("olegshpagin/russia-stocks-prices-ohlcv", extract_subfolder="D1")

    # 2. ALGOPACK (расширенные данные)
    download_kaggle_dataset("olegshpagin/algopack-extra-market-data")

    # 3. Макроэкономика
    download_kaggle_dataset("demirtry/russian-investment-activity")

    # 4. Новости – прямая загрузка parquet
    download_news_parquet()

    # 5. RFSD (фундаментальные) – если Hugging Face доступен
    logger.info("Скачивание RFSD (2023)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("irlspbru/RFSD", "2023", split="train", streaming=False)
        df = pd.DataFrame(ds)
        dst = RAW_DIR / "rfsd_2023.parquet"
        df.to_parquet(dst)
        logger.info(f"  -> сохранён в {dst}")
    except Exception as e:
        logger.warning(f"  -> ошибка: {e}")

    # 6. GitHub
    clone_github_repo("https://github.com/foykes/moex-dataset.git")
    clone_github_repo("https://github.com/WebOfRussia/financial-news-sentiment.git")

    # 7. Создание ticker_list.csv из имён файлов *_D1.csv
    ticker_file = RAW_DIR / "ticker_list.csv"
    if not ticker_file.exists():
        logger.info("Создание ticker_list.csv...")
        tickers = set()
        for f in RAW_DIR.glob("*_D1.csv"):
            ticker = f.stem.replace("_D1", "")
            if ticker:
                tickers.add(ticker.upper())
        if tickers:
            pd.DataFrame({"ticker": sorted(tickers)}).to_csv(ticker_file, index=False)
            logger.info(f"  -> создан с {len(tickers)} тикерами")
        else:
            fallback = ["SBER","GAZP","LKOH","NVTK","YNDX","TCSG","VTBR",
                        "ROSN","GMKN","NLMK","SNGS","HYDR","MTSS","CHMF","MAGN"]
            pd.DataFrame({"ticker": fallback}).to_csv(ticker_file, index=False)

    logger.info("=" * 60)
    logger.info("Загрузка датасетов завершена!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()