# MAS 方向—停止视觉读数审计（2026-09-14）

## 结论

将 MAS 的视觉状态收窄为 `stop / left / right / straight` 后，RAFT-Large 能在
NAVSIM logged-real 上提供高覆盖、可审计的粗粒度运动读数。它不恢复米、速度、
横向距离或曲率。

| NAVSIM source-disjoint（n=1000） | coverage | accuracy | 95% CI | macro accuracy |
|---|---:|---:|---:|---:|
| 四类联合读数 | 97.4% | 80.5% | [77.9%, 82.9%] | 84.1% |

逐类 recall：stop `100%`（n=50）、left `82.9%`（n=414）、right `81.0%`
（n=248）、straight `72.5%`（n=262）。当前误差主要来自直行与轻微转向的边界。

## Stop 通道

`stop` 被定义为候选轨迹在四秒内始终没有离开原点 `0.25 m`。视觉证据不使用
动作或候选轨迹：在固定道路 ROI 内，逐区间计算包含小光流的 q25 幅度，再取四个
区间的中位数。

- 真实视频 stop/moving 为 `50/950`；
- source-disjoint coverage `99.4%`；
- balanced accuracy `100%`；
- stop recall 95% CI `[92.9%, 100%]`；
- 固定阈值为 `0.324 px`；
- stop/moving 的该证据中位数分别约为 `0.016 px / 39.7 px`；
- always-moving、always-stop 和 identity-shuffle 控制均约为 `0.5` balanced accuracy。

这只验证“整段基本静止”，不验证制动后在时域末端才停车，也不提供绝对速度。

## Direction 通道

停止样本不重复进入方向分数。在 950 条 moving 样本中，RAFT 方向读数覆盖
`924/950 = 97.3%`，准确率 `79.4%`，95% CI `[76.7%, 81.9%]`。折算到全部
1000 条 source，方向 coverage 为 `92.4%`。

## DriveWAM 的当前读数边界

1000 条原生 DriveWAM action 中，粗 token 计数为 left `437`、right `230`、
straight `332`、stop `1`。联合视觉—动作 coverage 为 `93.9%`，在可计分的
moving 子集上方向一致率为 `75.6%`，95% CI `[72.8%, 78.3%]`。

由于只有一个 native stop action，不能把 moving 一致率伪装成 balanced
`MAS.stop`。该通道必须标记为 `diagnostic_insufficient_native_stop_actions`；正式
stop 确认需要同源 stop/move 干预或包含足够 native stop 的另一个 WAM。

## 跨模型方向确认

以下结果全部使用 NAVSIM logged-real 冻结的同一 RAFT 读数器和阈值。MAS 在各模型
的 left/right 分支上逐支比较；RCS 在同一 source 内比较 left/right 差分。RCS 的
`measurement coverage` 是左右视觉都可读的比例，计分 n 还要求 action yaw 差达到
`0.01 rad`。

| 模型 | MAS.direction | MAS coverage | RCS.direction | RCS 95% source CI | RCS measurement coverage | RCS 计分 n/总 pair |
|---|---:|---:|---:|---:|---:|---:|
| DriveWAM | 77.9% | 96.9% | 86.2% | [80.4%, 91.3%] | 93.7% | 138/255 |
| Epona | 77.7% | 95.4% | 91.9% | [85.9%, 97.0%] | 94.3% | 99/174 |
| DriveVA（共享 seed 重跑） | 60.2% | 88.0% | 83.9% | [71.0%, 96.8%] | 82.0% | 31/50 |
| WorldDrive（强转向 smoke） | 100% | 100% | 100% | Wilson [72.2%, 100%] | 100% | 10/10 |

MAS 的区间大幅重叠，不能单独支撑模型排名。DriveWAM 与 Epona 的 RCS 正常/
反转分数分别为 `86.2%/13.8%` 和 `91.9%/8.1%`，但两者 source-bootstrap CI
重叠，不能据此宣称两模型有显著排序。

DriveVA 的旧运行发现了协议污染：manifest 虽记录同一 pair 的
`nuisance_seed` 相同，生成器实际传给模型的是逐行递增的 `seed`。因此左右分支
同时改变了动作和扩散噪声，旧 `61.3%` 不能解释为模型能力。实际共享 seed 后，
RCS 上升到 `83.9%`，反转控制降为 `16.1%`；说明同源差分读数确实能在第三种架构
上捕捉动作响应。它仍未通过正式 promotion：source-bootstrap CI 下界为 `71.0%`
（低于 `75%`），measurement pair coverage 为 `82.0%`（低于 `90%`），且仅有
`31` 个可计分 pair。单支 MAS 同时降至 `60.2%`，说明 DriveVA 上场景/生成偏置
不能靠单支绝对方向匹配消除，而 RCS 的同源差分可以部分抵消。

WorldDrive 的 `10/10` 来自预先筛选的强 left/right 命令对，只证明冻结接口能在
第四种架构上跑通。样本量太小、没有 straight/stop，且 Wilson 下界仅 `72.2%`；
因此必须标记为 smoke，不能将 `100%` 当作正式分数或模型排名证据。

这些模型并非全部使用同一 source 池。DriveWAM/Epona 有 `174` 个共同 source；
DriveVA 与二者的交集仅 `22/18`，WorldDrive 又是独立的 10-pair 筛选集。因此当前
结果可以验证协议跨模型可执行和响应信号，但不能组成严格 leaderboard。

原始三个分支集均无 stop/move twin；随后新增了 DriveWAM 的受控 stop/move
实验。它从 255 条 source 中候选无关地筛出“历史速度不超过 `1 m/s`、logged
future 前进至少 `2 m`”的 56 条，并冻结其中 32 对。两支保持相同 source、history
和实际扩散 seed，通过 context action chunk 注入 stationary/move pose。

**该次 DriveWAM 运行已判为无效，不得解释成模型性能。** 人工查看本地拉取的
生成帧后，32 对 stop/move future 均呈块状解码噪声，而不是可辨认的道路视频；
同一服务器上旧的 DriveWAM 正常运行仍能生成清晰道路场景。因此先前记录的
`51.6%` balanced accuracy、`RCS.stop=6.25%`、`0.361/0.359 px` 以及 CoTracker
复核只保留作失败运行审计，不能支持“DriveWAM 不响应停止”。必须在生成质量门
通过后重新生成和评分。机器可读报告已标记 `invalid_generation_artifact`。

作为独立模型对照，Epona 在 21 个同源 stop/move twin 上生成了可辨认的道路视频。
冻结 NAVSIM 读数器覆盖 `100%`，但 `MAS.stop` balanced accuracy 仅 `57.1%`
（stop/moving recall `14.3%/100%`）；绝对终点判定只有 `3/21=14.3%`。
按修正后的 RCS 定义，`RCS.stop=85.7%`（18/21），source-bootstrap 95% CI
`[71.4%, 100%]`：RCS 只检验同源干预的相对响应（stop 视觉运动是否小于
move），不要求 stop 分支达到绝对停止；绝对 MAS.stop 结果另行报告。这是一项
有效的响应结果：Epona 的 stop 干预通常降低了视觉运动，但多数样本没有完全停住。
它证明问题不是读数器在所有生成视频上都不可用，但尚不能外推到其他 WAM。

## 与 SE(2) 的关系

在先前同一 255-source 对照中，两者方向准确率接近，但 RAFT 的基础 coverage 更高：

| reader | 基础 coverage / accuracy | 可靠性门 coverage / accuracy |
|---|---:|---:|
| corrected SE(2)-yaw | 85.1% / 74.7% | 71.8% / 80.9% |
| RAFT whole-clip direction | 96.5% / 74.4% | 71.0% / 83.4% |

SE(2) 会额外输出连续 yaw、纵向、横向、速度和曲率，但现有验证只支持 yaw 方向；
其余字段仍受幅度坍缩、yaw—lateral 歧义和投影/优化失败影响，只能作为
diagnostic。对当前粗粒度 MAS，纯流场路径更短、覆盖更高，也避免把未经验证的
连续量混入 primary 分数。

机器可读结果见：

- `reports/mas_direction_navsim1000_drivewam_20260914.json`
- `reports/mas_stop_navsim1000_drivewam_20260914.json`
- `reports/mas_direction_stop_navsim1000_drivewam_20260914.json`
- `reports/flow_token_drivewam_pair_20260914.json`
- `reports/flow_token_epona_pair_20260914.json`
- `reports/flow_token_driveva_pair_seedfixed_20260914.json`
- `reports/flow_token_worlddrive_pair_smoke10_20260914.json`
- `reports/flow_token_scorecard_20260914.json`
- `reports/drivewam_stop_move_control_20260914.json`
- `reports/whole_clip_yaw_backend_ab_20260914.json`
