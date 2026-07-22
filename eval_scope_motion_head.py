#!/usr/bin/env python3
"""Evaluate a checkpoint produced by ``train_scope_motion_head.py``.

All standard IAC sample and group-ranking metrics are delegated to the existing
``eval_dinov2_critic.py`` implementation.
"""

from __future__ import annotations

import eval_dinov2_critic as evaluator
from train_scope_motion_head import ScopeDinoMotionCritic


def main() -> None:
    evaluator.DINOv2ConsistencyCritic = ScopeDinoMotionCritic
    evaluator.main()


if __name__ == "__main__":
    main()
