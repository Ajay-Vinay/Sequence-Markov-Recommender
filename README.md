
# Sequence-Aware Movie Recommender via Genre Markov Chain (with Context & MMR)

This repo contains a **Colab-ready** notebook that builds a **sequence-aware recommender** using a **first-order Markov chain** over movie **genres**, with **weekday vs weekend** context-specific transitions and an optional **MMR** diversity re-rank. It evaluates against a **global popularity** baseline on MovieLens **ml-latest-small** (~100k ratings).

## How to Run (Colab)
1. Open `Sequence_Markov_Recommender.ipynb` in Google Colab.
2. Run all cells in order. The notebook downloads the dataset automatically and finishes in a few minutes.
3. Outputs:
   - Overall **Hit@10** and **Precision@10**
   - Per-context Hit@10 (weekday vs weekend)
   - Transition heatmap
   - Example recommendations with reasons

## Methods
- **Markov (Genre-Level):** For each user, use their **last TRAIN event’s** primary genre `g_t` and context (weekday/weekend) to get `P(next_genre | g_t, context)`. Recommend top movies from top predicted next genres not seen by the user.
- **MMR Re-Rank (Optional):** Increase intra-list diversity using **Maximal Marginal Relevance** on TF-IDF vectors of `title + genres`.
- **Popularity Baseline:** Global Top-N by frequency, excluding already seen items.

## Evaluation
- **Split:** Leave-One-Out per user (last interaction in TEST; rest in TRAIN).
- **Metrics:** Hit@10 and Precision@10 (macro-averaged).
- **Context:** Mark the context by the last TRAIN event (weekday/weekend).

## Files
- `Sequence_Markov_Recommender.ipynb` — main notebook
- `requirements.txt` — minimal Python deps (Colab already has most)

## Requirements
- Python 3.9+
- pandas, numpy, scikit-learn, matplotlib, scipy

## Notes & Extensions
- Swap genre with richer item features (e.g., tags, synopsis).
- Add more contexts (hour-of-day, month).
- Try higher-order transitions or a small session-based model.
