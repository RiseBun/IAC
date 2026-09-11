from pathlib import Path

import pytest

from scripts.prepare_epona_pure_speed_action_roots import prepare


def test_prepare_epona_roots(tmp_path: Path) -> None:
    source = tmp_path / "sample.pkl"
    source.write_bytes(b"placeholder")
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {
            "source_key": f"s{i}",
            "source_sample": str(source),
            "speed_role": "fast",
            "intervention_type": "pure_speed",
            "action_trajectory": [[float(j), 0.0, 0.0] for j in range(8)],
        }
        for i in range(3)
    ]
    import json

    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = prepare(manifest, tmp_path / "out", shard_size=2, speed_role="fast")
    assert report["source_count"] == 3
    assert len(report["manifest_paths"]) == 2
    assert json.loads((tmp_path / "out/shard_0/manifest.json").read_text())[0]["predicted_action_trajectory"]


def test_missing_source_sample_fails(tmp_path: Path) -> None:
    import json

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"source_key": "s", "action_trajectory": [[0, 0, 0]] * 8}) + "\n")
    with pytest.raises(ValueError, match="source_sample"):
        prepare(manifest, tmp_path / "out")
