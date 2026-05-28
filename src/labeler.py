from __future__ import annotations

import pandas as pd

from . import config


def label_gender(gender_raw: str) -> str | None:
    #Map the site's gender field to our class, or None if unmappable."""
    return config.GENDER_MAP.get(str(gender_raw).strip().lower())


def label_sleeve_from_length(sleeve_length) -> str | None:
    #Map Myntra's structured 'Sleeve Length' attribute to our class."""
    if not isinstance(sleeve_length, str) or not sleeve_length.strip():
        return None
    val = sleeve_length.strip().lower()
    if any(bad in val for bad in config.SLEEVE_LENGTH_EXCLUDE):
        return None  
    return config.SLEEVE_LENGTH_MAP.get(val)


def label_sleeve(text: str) -> str | None:
    #Fallback: infer sleeve type from product text, or None if undeterminable."""
    blob = str(text).lower()
    if any(bad in blob for bad in config.SLEEVE_EXCLUDE):
        return None 
    for cls, keywords in config.SLEEVE_KEYWORDS.items():
        if any(kw in blob for kw in keywords):
            return cls
    return None  # no sleeve keyword present


def _distribution(series: pd.Series) -> str:
    counts = series.value_counts(dropna=False).to_dict()
    return ", ".join(f"{k}={v}" for k, v in counts.items())


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gender_label"] = df["gender_raw"].apply(label_gender)

    # Prefer the structured "Sleeve Length" attribute; fall back to text keywords.
    if "sleeve_length" in df.columns:
        structured = df["sleeve_length"].apply(label_sleeve_from_length)
    else:
        structured = pd.Series([None] * len(df), index=df.index)
    text_based = df["text_blob"].apply(label_sleeve)
    df["sleeve_label"] = structured.where(structured.notna(), text_based)
    return df


def run() -> pd.DataFrame:
    if not config.METADATA_CSV.exists():
        raise FileNotFoundError(
            f"{config.METADATA_CSV} not found. Run `python -m src.scraper` first."
        )
    df = pd.read_csv(config.METADATA_CSV)
    df = build_labels(df)

    # Keep rows useful for at least one task; drop fully unlabeled ones.
    usable = df[df["gender_label"].notna() | df["sleeve_label"].notna()].copy()
    usable.to_csv(config.LABELED_CSV, index=False)

    print(f"Labeled {len(df)} products -> kept {len(usable)} -> {config.LABELED_CSV}\n")
    print("Gender label distribution:")
    print(f"  {_distribution(df['gender_label'])}")
    print(f"  usable for gender training: {df['gender_label'].notna().sum()}\n")
    print("Sleeve label distribution:")
    print(f"  {_distribution(df['sleeve_label'])}")
    print(f"  usable for sleeve training: {df['sleeve_label'].notna().sum()}")

    if df["sleeve_label"].notna().sum() < 40:
        print(
            "\nWARNING: very few sleeve labels were found. Sleeve keywords may be "
            "rare in the scraped text. Consider scraping more products, widening "
            "SLEEVE_KEYWORDS in config.py, or hand-labeling a subset."
        )
    return usable


if __name__ == "__main__":
    run()
