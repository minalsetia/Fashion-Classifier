"""Streamlit UI for the Fashion Attribute Classification app.

Four pages (sidebar):
  • Products          — gallery of scraped products + their labels
  • Single Prediction — classify one uploaded image / URL / catalogue item
  • Batch Prediction  — classify many items at once
  • History           — prediction-tracking records read back from the DB

Run:
    streamlit run app.py
"""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st
from PIL import Image

from src import config, database
from src.predict import COMBINED_VERSION, Predictor, load_image

st.set_page_config(page_title="Fashion Attribute Classifier", page_icon="👕",
                   layout="wide")


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource
def get_predictor() -> Predictor:
    return Predictor()


@st.cache_data
def load_catalogue() -> pd.DataFrame:
    """Scraped products (prefer labeled.csv so gallery can show labels)."""
    path = config.LABELED_CSV if config.LABELED_CSV.exists() else config.METADATA_CSV
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _image_source(row: pd.Series) -> str:
    """Local file if present (faster), else the remote URL."""
    local = config.BASE_DIR / str(row.get("image_path", ""))
    return str(local) if local.exists() else str(row.get("image_url", ""))


def _result_table(results: list[dict]) -> pd.DataFrame:
    cols = ["image_ref", "predicted_gender", "gender_confidence",
            "predicted_sleeve", "sleeve_confidence", "status", "error_message"]
    return pd.DataFrame(results)[cols]


def _log_upload(predictor: Predictor, img: Image.Image, ref: str,
                run_id: str, run_type: str) -> dict:
    """Predict an uploaded PIL image and write a tracking row."""
    preds = predictor.predict_image(img)
    database.log_prediction(
        run_id=run_id, run_type=run_type, image_ref=ref,
        predicted_gender=preds["predicted_gender"],
        gender_confidence=preds["gender_confidence"],
        predicted_sleeve=preds["predicted_sleeve"],
        sleeve_confidence=preds["sleeve_confidence"],
        model_version=preds["model_version"], status="success",
    )
    return {"image_ref": ref, "status": "success", "error_message": None, **preds}


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_products(catalogue: pd.DataFrame) -> None:
    st.header("🛍️ Product Catalogue")
    if catalogue.empty:
        st.warning("No products yet. Run `python -m src.scraper` then "
                   "`python -m src.labeler`.")
        return

    st.caption(f"{len(catalogue)} scraped products.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Products", len(catalogue))
    if "gender_label" in catalogue:
        c2.metric("Gender-labeled", int(catalogue["gender_label"].notna().sum()))
    if "sleeve_label" in catalogue:
        c3.metric("Sleeve-labeled", int(catalogue["sleeve_label"].notna().sum()))

    n_show = st.slider("How many to display", 4, min(120, len(catalogue)),
                       min(24, len(catalogue)), step=4)
    cols = st.columns(4)
    for i, (_, row) in enumerate(catalogue.head(n_show).iterrows()):
        with cols[i % 4]:
            st.image(_image_source(row), use_container_width=True)
            label_bits = []
            if pd.notna(row.get("gender_label")):
                label_bits.append(f"👤 {row['gender_label']}")
            if pd.notna(row.get("sleeve_label")):
                label_bits.append(f"👕 {row['sleeve_label']}")
            st.caption(f"{str(row.get('name', ''))[:40]}\n\n" + " · ".join(label_bits))


def page_single(predictor: Predictor, catalogue: pd.DataFrame) -> None:
    st.header("🔍 Single Prediction")
    mode = st.radio("Input source", ["Upload image", "Image URL", "Pick from catalogue"],
                    horizontal=True)

    img: Image.Image | None = None
    ref: str | None = None
    source_kind = "upload"

    if mode == "Upload image":
        up = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
        if up:
            img = Image.open(io.BytesIO(up.read())).convert("RGB")
            ref = f"upload:{up.name}"
    elif mode == "Image URL":
        url = st.text_input("Image URL")
        if url:
            ref, source_kind = url, "url"
    else:
        if catalogue.empty:
            st.info("No catalogue available — scrape products first.")
        else:
            names = catalogue["name"].fillna("(unnamed)").astype(str).tolist()
            pick = st.selectbox("Product", range(len(names)),
                                format_func=lambda i: names[i][:60])
            row = catalogue.iloc[pick]
            ref, source_kind = _image_source(row), "url"
            st.image(ref, width=220)

    if st.button("Predict", type="primary") and ref is not None:
        with st.spinner("Classifying..."):
            if source_kind == "upload":
                run_id = database.new_run_id()
                result = _log_upload(predictor, img, ref, run_id, "single")
            else:
                result = predictor.predict_single(ref)
        if result["status"] == "error":
            st.error(f"Failed: {result['error_message']}")
        else:
            if source_kind == "url":
                st.image(ref, width=220)
            cols = st.columns(2)
            cols[0].metric("Gender", str(result["predicted_gender"]),
                           f"conf {result['gender_confidence']}")
            cols[1].metric("Sleeve", str(result["predicted_sleeve"]),
                           f"conf {result['sleeve_confidence']}")
            st.caption(f"Model: {result['model_version']} · logged to history ✅")


def page_batch(predictor: Predictor, catalogue: pd.DataFrame) -> None:
    st.header("📦 Batch Prediction")
    mode = st.radio("Input source", ["Upload images", "Select from catalogue"],
                    horizontal=True)
    results: list[dict] = []

    if mode == "Upload images":
        ups = st.file_uploader("Choose images", type=["jpg", "jpeg", "png", "webp"],
                               accept_multiple_files=True)
        if ups and st.button("Run batch", type="primary"):
            run_id = database.new_run_id()
            with st.spinner(f"Classifying {len(ups)} images..."):
                for up in ups:
                    try:
                        img = Image.open(io.BytesIO(up.read())).convert("RGB")
                        results.append(_log_upload(predictor, img,
                                                   f"upload:{up.name}", run_id, "batch"))
                    except Exception as exc:
                        results.append({"image_ref": f"upload:{up.name}",
                                        "status": "error", "error_message": str(exc),
                                        "predicted_gender": None, "gender_confidence": None,
                                        "predicted_sleeve": None, "sleeve_confidence": None})
    else:
        if catalogue.empty:
            st.info("No catalogue available — scrape products first.")
        else:
            n = st.slider("How many catalogue items", 2, min(50, len(catalogue)),
                          min(10, len(catalogue)))
            sample = catalogue.head(n)
            if st.button("Run batch", type="primary"):
                sources = [_image_source(r) for _, r in sample.iterrows()]
                with st.spinner(f"Classifying {len(sources)} items..."):
                    results = predictor.predict_batch(sources)

    if results:
        ok = sum(r["status"] == "success" for r in results)
        st.success(f"Done — {ok}/{len(results)} succeeded. Logged to history ✅")
        st.dataframe(_result_table(results), use_container_width=True)


def page_history() -> None:
    st.header("🕓 Prediction History")
    if st.button("🔄 Refresh"):
        st.rerun()
    df = database.fetch_history(limit=500)
    if df.empty:
        st.info("No predictions logged yet.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Total records", len(df))
    c2.metric("Single runs", int((df["run_type"] == "single").sum()))
    c3.metric("Batch records", int((df["run_type"] == "batch").sum()))

    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download history CSV", df.to_csv(index=False),
                       "prediction_history.csv", "text/csv")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("👕 Fashion Attribute Classifier")
    database.init_db()
    catalogue = load_catalogue()
    predictor = get_predictor()

    if not predictor.ready:
        st.sidebar.error("⚠️ No trained models found. Run "
                         "`python -m src.train --task all` first.")
    else:
        st.sidebar.success(f"Models loaded:\n\n`{COMBINED_VERSION}`")
    if st.sidebar.button("Reload models / data"):
        get_predictor.clear()
        load_catalogue.clear()
        st.rerun()

    page = st.sidebar.radio("Navigate", ["Products", "Single Prediction",
                                         "Batch Prediction", "History"])

    if page == "Products":
        page_products(catalogue)
    elif page == "Single Prediction":
        if predictor.ready:
            page_single(predictor, catalogue)
        else:
            st.warning("Train the models before predicting.")
    elif page == "Batch Prediction":
        if predictor.ready:
            page_batch(predictor, catalogue)
        else:
            st.warning("Train the models before predicting.")
    else:
        page_history()


if __name__ == "__main__":
    main()
