from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from . import config

GATEWAY_URL = "https://www.myntra.com/gateway/v2/search/{query}"
PRODUCT_URL = "https://www.myntra.com/gateway/v2/product/{product_id}"
ROWS_PER_PAGE = 50  # Myntra caps search rows per request around this

DETAIL_ATTRS = ["Sleeve Length", "Sleeve Styling", "Neck", "Fit", "Patterns"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.myntra.com/",
    # The gateway returns 401 without these app-identifying headers that Myntra's own web client sends:
    "x-meta-app": "channel=web",
    "x-myntraweb": "Yes",
    "x-requested-with": "browser",
}


class ScrapeBlockedError(RuntimeError):
    """Raised when Myntra refuses our requests (403 / bot wall / empty body)."""


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        session.get("https://www.myntra.com/", timeout=config.REQUEST_TIMEOUT)
    except requests.RequestException:
        pass
    return session


def fetch_page(session: requests.Session, query: str, offset: int) -> list[dict]:
    #Fetch one page of products for a query. Returns raw product dicts."""
    url = GATEWAY_URL.format(query=query)
    params = {"rows": ROWS_PER_PAGE, "o": offset, "plaEnabled": "false"}

    last_exc: Exception | None = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
            if resp.status_code in (401, 403):
                raise ScrapeBlockedError(
                    f"Myntra returned {resp.status_code} for '{query}'. The request "
                    "was blocked/unauthorized despite app headers + cookie warm-up. "
                    "See README → Troubleshooting scraping."
                )
            resp.raise_for_status()
            data = resp.json()
            return data.get("products", []) or []
        except ScrapeBlockedError:
            raise  # don't retry a hard block
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            wait = config.SLEEP_BETWEEN_REQUESTS * attempt
            print(f"  retry {attempt}/{config.MAX_RETRIES} for '{query}' "
                  f"(offset {offset}) after error: {exc} — sleeping {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch '{query}' offset {offset}: {last_exc}")


def parse_product(raw: dict, query: str, gender_hint: str) -> dict | None:
    """Normalise a raw Myntra product into our flat metadata schema."""
    image_url = raw.get("searchImage")
    product_id = raw.get("productId")
    if not image_url or product_id is None:
        return None  # unusable without an image / id

    # Build a rich text field used downstream for weak-supervision labelling.
    name = raw.get("product", "")
    brand = raw.get("brand", "")
    extra = raw.get("additionalInfo", "") or ""
    text_blob = " ".join(str(x) for x in (brand, name, extra)).strip()

    return {
        "product_id": product_id,
        "brand": brand,
        "name": name,
        "text_blob": text_blob,
        # Prefer the site's own gender field; fall back to the query's hint.
        "gender_raw": (raw.get("gender") or gender_hint or "").lower(),
        "category": raw.get("category", ""),
        "price": raw.get("price"),
        "search_query": query,
        "image_url": image_url,
        "product_url": f"https://www.myntra.com/{raw.get('landingPageUrl', '')}",
    }


def fetch_attributes(session: requests.Session, product_id: int) -> dict:
    """Return a product's structured articleAttributes from its detail page.

    Returns {} on any failure — one missing detail page must not abort the scrape.
    """
    url = PRODUCT_URL.format(product_id=product_id)
    try:
        resp = session.get(url, timeout=config.REQUEST_TIMEOUT)
        resp.raise_for_status()
        attrs = resp.json().get("style", {}).get("articleAttributes", {})
        return attrs if isinstance(attrs, dict) else {}
    except (requests.RequestException, ValueError):
        return {}


def enrich_with_details(session: requests.Session, df: pd.DataFrame) -> pd.DataFrame:
    """Add a `sleeve_length` column (+ richer text_blob) from detail pages.

    Detail pages are fetched concurrently for speed; per-product failures
    degrade gracefully to an empty attribute set.
    """
    ids = df["product_id"].tolist()
    print(f"\nFetching detail-page attributes for {len(ids)} products "
          f"({config.PDP_WORKERS} workers)...")
    attrs_by_id: dict = {}
    with ThreadPoolExecutor(max_workers=config.PDP_WORKERS) as pool:
        futures = {pool.submit(fetch_attributes, session, pid): pid for pid in ids}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Details", unit="item"):
            attrs_by_id[futures[fut]] = fut.result()

    sleeve_lengths, enriched_text = [], []
    for _, row in df.iterrows():
        attrs = attrs_by_id.get(row["product_id"], {})
        sleeve_lengths.append(attrs.get("Sleeve Length"))
        extra = " ".join(str(attrs[a]) for a in DETAIL_ATTRS if attrs.get(a))
        enriched_text.append(f"{row['text_blob']} {extra}".strip())

    df = df.copy()
    df["sleeve_length"] = sleeve_lengths
    df["text_blob"] = enriched_text
    return df


def download_image(session: requests.Session, url: str, dest: Path) -> bool:
    """Download an image to dest. Returns True on success."""
    if dest.exists():
        return True
    try:
        resp = session.get(url, timeout=config.REQUEST_TIMEOUT, stream=True)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except requests.RequestException as exc:
        print(f"  ! failed to download {url}: {exc}")
        return False


def scrape(target_count: int = config.TARGET_IMAGE_COUNT) -> pd.DataFrame:
    """Scrape products across all configured queries and persist metadata."""
    session = build_session()
    collected: dict[int, dict] = {}  # product_id -> record (dedupes across queries)

    print(f"Scraping Myntra for up to {target_count} products "
          f"across {len(config.SCRAPE_QUERIES)} queries...\n")

    pbar = tqdm(total=target_count, desc="Products", unit="item")
    try:
        for spec in config.SCRAPE_QUERIES:
            if len(collected) >= target_count:
                break
            query, hint = spec["query"], spec["gender_hint"]
            # Even share per query (ceil division) so every category — and thus
            # both genders and both sleeve types — is represented.
            per_query_cap = -(-target_count // len(config.SCRAPE_QUERIES))
            offset = 0
            taken_for_query = 0

            while taken_for_query < per_query_cap and len(collected) < target_count:
                products = fetch_page(session, query, offset)
                if not products:
                    break  # exhausted this query
                for raw in products:
                    rec = parse_product(raw, query, hint)
                    if rec and rec["product_id"] not in collected:
                        collected[rec["product_id"]] = rec
                        taken_for_query += 1
                        pbar.update(1)
                        if len(collected) >= target_count:
                            break
                offset += ROWS_PER_PAGE
                time.sleep(config.SLEEP_BETWEEN_REQUESTS)
    finally:
        pbar.close()

    if not collected:
        raise ScrapeBlockedError(
            "No products were scraped. Myntra likely blocked the requests. "
            "See README → Troubleshooting scraping for fallback options."
        )

    df = pd.DataFrame(collected.values())

    # Enrich with structured detail-page attributes (esp. Sleeve Length).
    if config.FETCH_PRODUCT_DETAILS:
        df = enrich_with_details(session, df)

    # Download images and record the local path (skip rows whose image fails).
    print(f"\nDownloading {len(df)} images to {config.IMAGES_DIR} ...")
    image_paths: list[str | None] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Images", unit="img"):
        ext = Path(row["image_url"]).suffix or ".jpg"
        dest = config.IMAGES_DIR / f"{row['product_id']}{ext}"
        ok = download_image(session, row["image_url"], dest)
        image_paths.append(str(dest.relative_to(config.BASE_DIR)) if ok else None)
    df["image_path"] = image_paths
    df = df[df["image_path"].notna()].reset_index(drop=True)

    config.METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.METADATA_CSV, index=False)
    print(f"\nSaved {len(df)} products -> {config.METADATA_CSV}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Myntra fashion products.")
    parser.add_argument(
        "--count", type=int, default=config.TARGET_IMAGE_COUNT,
        help=f"number of products to scrape (default {config.TARGET_IMAGE_COUNT})",
    )
    args = parser.parse_args()
    try:
        scrape(args.count)
    except ScrapeBlockedError as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
