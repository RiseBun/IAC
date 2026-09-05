"""Detect scene, sample, and image overlap between evaluation splits."""

from __future__ import annotations

from typing import Any


def split_keys(rows: list[dict[str, Any]]) -> tuple[set[str], set[str], set[str]]:
    scenes = {str(row.get("scene_id") or row.get("scene_name") or "") for row in rows}
    samples = {str(row.get("sample_id") or "") for row in rows}
    frames = {
        str(frame)
        for row in rows
        for frame in (
            list(row.get("frame_paths") or [])
            + list(row.get("history_frame_paths") or [])
            + list(row.get("future_frame_paths") or [])
        )
    }
    scenes.discard("")
    samples.discard("")
    return scenes, samples, frames
