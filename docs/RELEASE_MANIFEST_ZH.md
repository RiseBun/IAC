# IAC Benchmark 发布清单

本清单定义 GitHub `main` 中哪些文件属于可复现协议，哪些内容
必须留在私有评测环境。

## 发布内容

| 路径 | 作用 | 是否必需 |
|---|---|---:|
| `src/iac_new/` | 光流、标定地面几何、连续解码、可观测性、CFAC/FAU 评分 | 是 |
| `src/iac_new/foresight_metrics.py` | CFAC/FAU 主度量实现 | 是 |
| `scripts/validate_wam_submission.py` | 提交格式与泄漏审计 | 是 |
| `scripts/audit_wam_level1_outputs.py` | 接受 4/8 未来帧的输出审计 | 是 |
| `scripts/build_wam_level1_continuous_manifest.py` | 公开身份与私有帧 join | 是 |
| `scripts/evaluate_continuous_decoder.py` | Step 1 图像侧解码入口 | 是 |
| `scripts/evaluate_continuous_motion_alignment.py` | 图像运动与 native action 对齐 | 是 |
| `scripts/score_iac_submission.py` | 能力分层记分板 | 是 |
| `configs/plane.json` | 冻结配置：448×256 解码、512×288 RAFT 推理 | 是 |
| `datasets/benchmark_public.jsonl` | 1000 条脱敏 NAVSIM 主榜身份与协议元数据 | 是 |
| `datasets/benchmark.audit.json` | 选集、分层和泄漏审计摘要 | 是 |
| `weights/` | RAFT-Large 权重、来源和 SHA-256 | 是 |
| `LICENSE` / `CITATION.cff` | 开源许可与引用元数据 | 是 |
| `tests/` | 确定性协议与几何测试 | 推荐 |
| `docs/` | 协议、结果、审计与复现边界 | 是 |
| `reproduction/` | DriveWAM 与 NAVSIM/PDM 参考实验复现工具 | 推荐 |
| `tools/dataset/` | 从许可 NAVSIM 数据重建冻结选集 | 推荐 |

## 明确排除

- 原始 NAVSIM/Waymo 图像、未来 GT、私有绝对路径；
- DriveWAM、LingBot-VA 或其他 WAM checkpoint；
- 生成视频、逐样本服务器日志和中间缓存；
- 未冻结的分辨率消融、区域航向 shadow 配置与实验日志；
- 早期小规模试点结果和历史运行记录。

## 复现边界

公开 manifest 只提供 `sample_id`、split、时间轴、标定和历史状态。评测端必须将
私有图像与 GT 按 `sample_id` join；提交方不得把 GT、realized state 或候选轨迹
注入图像侧解码器。所有不可观测区间必须 abstain，不能用 0 或插值掩盖。

## 发布前检查

```bash
python -m pip install -e .
PYTHONPATH=src:. python -m pytest -q
python scripts/audit_benchmark_manifest.py --public --manifest datasets/benchmark_public.jsonl --output <audit.json>
(cd weights && sha256sum -c SHA256SUMS.txt)
```

发布包版本由 `VERSION`、`pyproject.toml` 和 `src/iac_new/__init__.py` 三处共同
声明，当前均为 `1.0.0`。
