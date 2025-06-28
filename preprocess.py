import pandas as pd
import json
from pathlib import Path
import numpy as np


def main():
    csv_path = Path('GoodReads_100k_books.csv.xz')
    if not csv_path.exists():
        csv_path = Path('GoodReads_50k_books.csv.xz')
    df = pd.read_csv(csv_path, encoding='utf-8-sig', low_memory=False)

    df = df[['pages', 'desc', 'reviews', 'rating']].copy()
    df['blurb'] = df['desc'].fillna('').astype(str).str.len()
    df['pages'] = pd.to_numeric(df['pages'], errors='coerce')
    df['reviews'] = pd.to_numeric(df['reviews'], errors='coerce')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df.dropna(subset=['pages', 'reviews', 'rating'], inplace=True)

    for col in ['pages', 'blurb', 'reviews', 'rating']:
        low, high = df[col].quantile([0.005, 0.995])
        df = df[(df[col] >= low) & (df[col] <= high)]

    df = df[['pages', 'blurb', 'reviews', 'rating']]
    if len(df) > 5000:
        df = df.sample(5000, random_state=42)

    data = df.to_dict(orient='records')
    out_path = Path('scatter_data.json')
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == '__main__':
    main()
