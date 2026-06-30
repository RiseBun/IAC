# IAC Index Audit

IAC training is only meaningful when positive rows use real future frames.
Before training or reporting results, audit each JSONL index:

```bash
python tools/audit_consistency_index.py \
  indices_navsim/consistency_train.jsonl \
  indices_navsim/consistency_val.jsonl \
  --image-root /path/to/sensor_or_data_root \
  --fail-positive-exact-overlap 0.01 \
  --fail-positive-any-overlap 0.05 \
  --fail-missing-images
```

Key fields:

- `positive_exact_overlap_ratio`: fraction of positive rows where `history_images`
  and `future_images` are identical. This should be near zero for a real IAC
  index.
- `positive_any_overlap_ratio`: fraction of positive rows sharing any history
  and future frame path. A high value means the index may be replaying history.
- `future_image_policy`: copied from the index summary when present. Policies
  such as `history_tail` or `repeat_current` are compatibility fallbacks, not a
  strict future-frame setup.

For NAVSIM/OpenScene logs, rebuild with `--future-image-policy future` when the
sensor blobs contain future frames. Use fallback policies only for smoke tests or
trajectory-only experiments, and do not report those results as image-action
future consistency.
