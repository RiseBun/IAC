# IAC 冻结视觉层资产登记

冻结日期：2026-09-15

长期目录：

`/mnt/slurmfs-4090node3/user_data/zchen897/model_registry/iac_visual_layer_v1_20260915`

该目录必须长期保留，不属于实验临时输出，不得被实验清理脚本删除。

## 冻结内容

- Reloc3r-512 模型权重、配置和源码快照；
- Metric3Dv2-v2-S 模型权重和源码快照；
- IAC 的 Metric3D + Reloc3r + known-rotation PnP 推理代码；
- 视觉输出层及 AS 的冻结配置；
- 全目录 `SHA256SUMS` 完整性清单；
- `configs/visual_layer_assets_v1.json` 机器可读资产登记。

SegFormer 仅是可选的道路/动态物体掩码辅助，不是冻结正式运动后端，因此不影响正式 yaw 或全图纵向输出的复算。

## 完整性规则

使用长期目录中的权重前，必须先在目录根部执行：

```bash
sha256sum -c SHA256SUMS
```

Reloc3r-512 权重必须匹配官方 revision
`55014aae1798ec830df51d7d2cd47c41f4388187` 和 SHA256
`d02f5f4d749c926195981fbd8de7f78009a72f9a1d3b9b8708dd9794bf5455c1`。

Metric3Dv2-v2-S 权重 SHA256 必须为
`b34b2a2be9148054991cef7e417930e1320602ba7bc503b0ee4e7888543728f6`。

推理必须通过 `--reloc3r-checkpoint` 指向长期目录中的
`weights/reloc3r-512`，不得依赖运行时联网下载或临时 Hugging Face 缓存。
