# Dataset tools

`dataset/` contains the deterministic selection and redaction utilities used
to rebuild `datasets/benchmark_public.jsonl` from a licensed NAVSIM copy.
These tools are not needed to submit or score a model. Private future images,
ground truth, and absolute source paths are removed by
`dataset/build_public_manifest.py` before publication.
