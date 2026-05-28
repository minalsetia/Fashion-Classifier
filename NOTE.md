# Project Note — Fashion Attribute Classification

## Approach
End-to-end pipeline: **scrape Myntra** (image + metadata) → **derive labels via
weak supervision** → **fine-tune two image classifiers** → **serve via a
Streamlit UI** → **log every prediction to SQLite**.

500 products were scraped from Myntra's internal JSON gateway across six
category queries (`men/women × t-shirts/shirts/tops/kurtas`) so both genders
and both sleeve types are represented. An initial `401 Unauthorized` was
resolved by adding Myntra's own app-identifying headers
(`x-meta-app`, `x-myntraweb`, `x-requested-with`) and warming up cookies via a
homepage request. Each product is then enriched concurrently (8-thread pool)
with its **product-detail `articleAttributes`**, giving Myntra's structured
**"Sleeve Length"** field — far more reliable than parsing the title text.

Labels are produced with **no manual annotation**:
- **Gender** from the site's category/gender field (`men → male`, `women → female`; unisex dropped).
- **Sleeve** from the structured Sleeve Length (`Long Sleeves → full`, `Short Sleeves → half`; sleeveless / three-quarter dropped), with product-text keyword matching kept as a fallback.

This yielded **488 gender-labeled** and **418 sleeve-labeled** images with a
near 50/50 class balance on both tasks.

## Model choice
**Transfer learning on an ImageNet-pretrained ResNet18.** With only a few
hundred images, training from scratch would overfit immediately; a pretrained
backbone already encodes generic visual features, so we train a fresh 2-class
head while gently fine-tuning the backbone.

Implementation details:
- **Two separate models** (one per task) — simpler to tune, debug, and explain than a multi-head model, with negligible extra compute at this scale.
- **Two parameter groups in the optimizer** — head at `LR=1e-3`, backbone at `LR=1e-4` — prevents catastrophic forgetting of pretrained features.
- **Standard augmentation** (RandomResizedCrop, HorizontalFlip, Rotation, ColorJitter) to fight overfitting on a small dataset.
- 8 epochs, batch 32, Adam, **stratified 80/20** train/val split, **best-by-val-accuracy checkpoint**.

Results on the held-out validation set:

| Task | Val accuracy | Confusion matrix | Notes |
|------|-------------|------------------|-------|
| Gender (female/male) | **1.00** (98 imgs) | `[[58, 0], [0, 40]]` | clothing photos are highly gender-separable; small val set means this is likely a slight over-estimate |
| Sleeve (full/half) | **0.92** (84 imgs) | `[[41, 3], [4, 36]]` | the genuinely harder task; 7 errors / 84, balanced precision and recall (~0.91) |

## Limitations
- **Scraping fragility** — Myntra's gateway requires specific app headers and a warmed-up session; their HTML/API can change and bot defences can tighten without notice.
- **Weak labels are not gold** — sleeve labels come from Myntra's own attribute (usually correct, occasionally mis-tagged); unisex items are dropped from gender training, shrinking that set slightly.
- **Small validation set** — 98 (gender) / 84 (sleeve) images; the 100% gender accuracy is likely a slight over-estimate of true generalisation.
- **Narrow scope** — only upper-body garments where sleeve length is meaningful; only two binary attributes are modelled.
- **Domain shift risk** — trained on Myntra studio shots (clean background, fixed lighting); real user photos with cluttered scenes and varied lighting would degrade accuracy.

## Improvements with more time
- **Multi-head architecture** (shared backbone, two heads) for efficiency and to let related signals reinforce each other; extend to more classes (sleeveless, three-quarter) and more attributes (pattern, fit, neck).
- **Hand-labeled gold test set** (~200 images) as a true held-out benchmark, kept separate from the noisy weak labels.
- **Stronger training loop:** k-fold CV, early stopping, LR scheduling (cosine annealing), class-weighting for imbalance, mixed precision when a GPU is available.
- **More and more-diverse data:** scrape across price/brand ranges, add user-uploaded photos to bridge the studio-vs-wild domain gap.
- **Production engineering:** a FastAPI inference service, a proper model registry with versioning, latency/confidence metrics, and A/B-tested rollouts.
- **Scraper robustness:** Playwright fallback when the JSON gateway is walled, proxy rotation, and a small retry queue for transient failures.
