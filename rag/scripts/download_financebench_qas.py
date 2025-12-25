"""Download FinanceBench QA data (CSV) from the PatronusAI repo."""

import os
import requests
from pathlib import Path

FINANCEBENCH_CSV_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/data/financebench.csv"


def download_csv(dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading FinanceBench QA to {dest} ...")
    resp = requests.get(FINANCEBENCH_CSV_URL, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print("Done.")


if __name__ == "__main__":
    target = Path(__file__).parent.parent / "data" / "question_sets" / "financebench_full.csv"
    if target.exists():
        print(f"{target} already exists. Skipping.")
    else:
        download_csv(target)
