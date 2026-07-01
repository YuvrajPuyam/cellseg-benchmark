# Vendored third-party sources

## hover_net_stats_utils.py
- **Upstream:** https://github.com/vqdang/hover_net - `metrics/stats_utils.py`
- **Pinned commit:** `a0f80c7acb9a14964d597d15d7ebd9688a0230cb`
- **License:** Apache-2.0 (see upstream `LICENSE`)
- **What/why:** `get_fast_aji`, `get_fast_aji_plus`, `remap_label` - AJI / AJI+ are the one
  instance metric `stardist.matching` does not provide. Vendored verbatim except the unused
  `cv2` / `matplotlib` imports were dropped (AJI needs neither). Prefer **AJI+** (Hungarian
  1-to-1) over classic AJI (greedy 1-to-many, over-penalises).
