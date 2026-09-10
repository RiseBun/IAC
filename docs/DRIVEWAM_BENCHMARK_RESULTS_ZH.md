# DriveWAM Benchmark 实验结果

## 范围

- 数据集：NAVSIM，1000 个非重叠样本；直行上限 30%，包含转弯、加速、制动和停车分层。
- 模型：`drivewam_navsim_checkpoint_20260824`，沿用服务器已有 checkpoint 与 LingBot-VA base，不重复下载。
- 输出位置由运行者通过命令行参数指定，不写入仓库。
- 图像探针：冻结 RAFT-Large、前后向一致性、地面平面自车几何、candidate-blind continuous decoder；纵向米制速度仅作诊断。

## 当前冻结结论（2026-09-10）

> **时间契约勘误（2026-09-10）：旧 DriveWAM 生成端结果撤回。**
> 输入构建器曾把 `4 history + 8 future` 共 12 帧直接交给官方
> `NavSimEpisodeDataset`；官方读取器却把下标 0 当作当前帧，并选择
> `[0,2,4,6,8]`。因此旧运行实际条件帧约为
> `[-1.5,-0.5,+0.5,+1.5,+2.5]s`，而动作和输出标签以 `t=0` 为锚点。
> 依赖旧生成帧的 coverage、方向准确率、Spearman、能量和几何保真度数字均无效。
> 只有下文明确标为 `temporal-fixed` 的重跑结果可以继续使用。真实帧的 logged-GT
> 审计和 Epona common-random 结果不经过该输入链，仍然有效。

Step 1 已恢复为修正后的连续 SE(2) 后端，primary 只保留 yaw 的方向与排序；
Step 1-S 降为独立诊断。它不是 GitHub 旧版的原样回滚：内参尺寸、固定证据分母、
输出投影支持、诚实拟合状态和 pair 双方 `explained` 门都已加入。完整差异、最新
覆盖率、准确率及 reference lineage 边界见
[`STEP1_SE2_YAW_V1_2_ZH.md`](STEP1_SE2_YAW_V1_2_ZH.md)。

`temporal-fixed` 重跑中，SE(2) Step 1.2 的正式 pair coverage 为
`171/255 = 67.1%`；106 个 material yaw pair 上方向准确率为
`87/106 = 82.1% [73.7%, 88.2%]`，Spearman 为
`0.832 [0.731, 0.902]`。排序仍强，但 coverage 未达 90%，方向 CI 下界也略低于
预注册的 0.75。S1.3 流场结构 pilot 为 `254/255 = 99.6%` coverage、
`126/149 = 84.6% [77.9%, 89.5%]` 方向准确率、Spearman
`0.779 [0.706, 0.841]`；它通过了单模型的 coverage 与方向门，但尚未通过同分布
两模型分离门，因此仍不能升级 primary。旧的 `175/255`、`90/103`、`0.786`
以及 S1.3 的 `255/255`、`116/135`、`0.713` 只保留为已撤回历史值。

固定粗网格初始化 G 是当前 SE(2) 的 Step 1.3 候选。它不改输入证据或质量门，
把生成端 pair coverage 提到 `205/255 = 80.4%`，yaw 方向准确率保持为
`95/116 = 81.9% [73.9%, 87.8%]`，Spearman 提到
`0.925 [0.861, 0.960]`。真实帧 `explained` 同时由 217/255 提到
231/255，yaw 读出比例中位数由 0.916 提到 0.939、Spearman 由 0.918 提到
0.992。它证明坏起点是 SE(2) 覆盖损失的一部分，但仍未使严格成对覆盖达到 90%，
所以保持 `experimental_ablation`，不静默改写 Step 1.2。

Step 1 的输出现在明确分成三层：`measurement_available` 表示四个未来区间
都有足够的视内投影支持；`motion_explanation_status` 表示自由刚体地面轨迹是否
相对零流基线有实际改善；二者都不等价于“轨迹幅度准确”。只有存在独立 logged
reference 时，才报告准确度/一致性。采样器采用道路中远场空间分层，避免可靠性
权重把证据集中到预测时域内容易出界的近场像素。

### Step 1.2 logged-GT 准确率审计

旧 real-counterpart manifest 把 `gt_candidate_id` 错指向 `wam_action_head`。因此
任何直接读取该字段的历史校准、相对误差或 history-only 对照都使用了错误参考，
这些结果全部撤回。动作头可以用于 action-alignment，但不能命名为 logged GT。

修正审计不再信任该字段，而是沿 255 条记录各自的 `lineage.source_sample` 打开
NAVSIM pickle，并验证 `future_trajectory[*].pose` 与
`realized_future_ego_state[:3]` 逐项相同。真实帧运行得到
`measurement_available = 243/255 (95.3%)`，
`explained/weak/abstain = 217/26/12`。在 explained 子集上采用固定 material
阈值 x >= 1.0 m、|y| >= 0.3 m、|yaw| >= 0.05 rad：

| 字段 | n | 末点读出比例中位数 | IQR | 方向准确率 | Spearman |
|---|---:|---:|---:|---:|---:|
| longitudinal | 217 | 0.565 | [0.314, 0.734] | 100% | 0.509 |
| lateral | 123 | 0.480 | [0.212, 0.646] | 95.1% | 0.926 |
| yaw | 117 | 0.916 | [0.695, 0.974] | 99.1% | 0.918 |

lateral-turn 分层含 92 条 explained，其中 91 条达到 lateral/yaw material 阈值；
yaw 读出比例中位数为 `0.930`、IQR `[0.809, 0.976]`。因此真实图像域的 yaw
方向、排序和幅度读出均通过内部效度检查。横向和纵向仍有明显幅度欠读，继续只作
diagnostic；纵向五折标定后仍只有约 27% 落在真值 +/-20% 内，不能靠单一增益升级。

这也撤回了“严格 logged-GT 只有 85 条且不含转弯”的旧说法。那是错误 manifest
连接造成的子集假象。生成 manifest 现已不设置 GT 指针；协议层拒绝
`gt_candidate_id=wam_action_head`，且冻结配置要求真正的 GT candidate 带
`navsim_logged_realized` 或 `private_realized_gt` 来源。配置顶层的
`intrinsics_source_size` 也不能替代逐样本字段，评估入口对此有回归测试。

边界不变：真实帧准确率说明 Step 1 在标定域能读 yaw；生成帧上的 action
一致率说明 WAM 视觉响应与自身条件动作是否同向。后者不是现实正确性，也不能拿
logged GT 去要求一个被不同动作干预的反事实分支复现实际发生的未来。

## Step 1-S：无米制重建的流场结构诊断线（2026-09-10）

该诊断线跳过 `flow -> SE(2) -> metre/radian`，直接从冻结 RAFT 时序流提取归一化
FoE、散度、旋度、横向流中心和左右空间对比。取点只依赖图像侧前后向一致性与
固定空间分层，不读取候选轨迹、GT 或左右动作；缺少支持的 interval 不填零。

首轮 255 条真实视频验证显示，这把序尺在转弯域有很强的内部效度：横向流中心
对末点 yaw 的 Spearman 为 `0.946 [0.897, 0.976]`，FoE-x 为
`-0.876 [-0.936, -0.778]`，curl 为 `-0.619 [-0.757, -0.449]`。真实视频
可用 interval 为 `74.0%`；要求四个 interval 全可用时为 `128/255`，协议必须
保留 `partial` 状态。

修正生成视频内参后，510 条生成分支的可用 interval 为 `71.8%`；255 个左右 pair
中，`231 (90.6%)` 至少有一个共同可用 interval，`68 (26.7%)` 四个 interval
全部共同可用。lateral-turn 的 95 个可比 pair 上：

| 结构量 | 对 action-head 对应差的 Spearman | 方向命中 |
|---|---:|---:|
| horizontal-flow center -> lateral | 0.631 [0.447, 0.767] | 60/87 = 69.0% [58.6%, 77.7%] |
| FoE-x -> yaw | 0.387 [0.167, 0.584] | 42/64 = 65.6% |
| curl -> yaw | 0.392 [0.196, 0.560] | 45/64 = 70.3% |
| left-right horizontal contrast -> yaw | 0.121（区间含零） | 34/64 = 53.1% |

因此首轮结论是“绕开投影后可以覆盖大多数 pair，并恢复显著的分支内方向/排序
信号”，不是“已经得到可靠满覆盖指标”。横向流中心是该诊断线的候选量，
FoE/curl 是诊断量；左右 contrast 未通过，应暂时停用。

冻结 pilot scorer 要求左右至少两个共同 interval。该口径覆盖
`189/255 = 74.1%`，action-response Spearman 为
`0.586 [0.455, 0.699]`；action 横向差超过 0.05 m 的 147 对中，方向命中为
`102/147 = 69.4% [61.5%, 76.3%]`。这是 pilot 通过，不是最终准确度通过。

原 pilot 的升级判据已预注册在 `configs/flow_structure.json`：pair 覆盖率至少
90%，方向准确率 95% CI 下界至少 0.75，并且冻结指标必须在同一协议上分离至少
两个 WAM。DriveWAM 结果没有通过，故当前状态固定为 `diagnostic_pilot`。

仿射流场的中位可解释度在真实帧为 `0.885`、生成帧为 `0.895`。生成帧并不低，
说明平滑但错误的流也能得到高分，所以该量不能单独命名为 WAM 几何保真度。
当前几何保真度必须拆开报告：输入支持、共同 interval 覆盖、命令方向一致率和
action-response 秩相关，不合并成一个事后加权总分。

本轮还发现一个独立的上游正确性问题：CCFC 生成清单中的图像已预缩放到
`448x256`，内参却仍对应原始 `1920x1080`。旧提取器把文件尺寸误当成标定尺寸，
使主点落到图像外并污染生成端投影/米制统计。现已引入显式
`intrinsics_source_size` 契约，并在去畸变和输出坐标两处使用；因此此前生成端的
`18% explained` 等数字撤回，不能作为 Step 1-S 的有效基线。真实帧文件本身为
原始尺寸，不受该错误影响。

### 修正内参后的刚体运动复核

在同一 255 个 source、左右 510 条生成分支上重跑指定轨迹能量。旧报告中生成减
真实的 GT 配对能量差 `+2.613` 降为 `+1.115 [1.032, 1.213]`，约 57% 的旧差距
来自内参错误。更重要的是，生成帧中 action 轨迹优于 logged GT 的 interval 为
`1410/2007 = 70.3%`，而真实帧中仅为 `234/1016 = 23.0%`。生成帧 action 能量
相对真实帧 action 基线的配对差为 `+0.073 [-0.048, 0.174]`，没有显著域间差异。
因此“生成帧既不能被 GT、也不能被自身 action 解释”的旧结论撤回。logged GT
偏离只能说明反事实分支不同于实际发生的未来，不能单独判定 WAM 错误。

随后用固定中远场空间分层、固定证据分母和宽网格加局部搜索做自由刚体拟合：

| 来源 | 自由拟合能量中位数 | 相对零流改善中位数 | 改善超过 0.05 | 零轨迹 |
|---|---:|---:|---:|---:|
| 真实帧 255 | 0.292 | -0.693 | 242/255 = 94.9% | 8/255 = 3.1% |
| 生成帧 510 | 0.425 | -0.548 | 485/510 = 95.1% | 19/510 = 3.7% |

这推翻了“生成视频通常不能被刚体运动解释”：修正内参和采样后，生成帧与真实帧
几乎同样普遍地存在显著优于零流的刚体解释。生成分支内的自由拟合对 action 的
yaw 方向命中为 `112/135 = 83.0% [75.5%, 88.9%]`、Spearman `0.852`；lateral
为 `148/205 = 72.2% [65.5%, 78.2%]`、Spearman `0.519`；longitudinal 为
`109/216 = 50.5%`、Spearman `0.095`。所以 SE(2) 路径重新成为有效诊断候选，
尤其是 yaw，但这些数字仍不证明米制幅度准确，也尚未满足跨模型评测定义。

### SEA-RAFT A/B

SEA-RAFT 使用同一 255 个 source、同一 `1920x1080 -> 448x256` 内参变换、同一
Step 1-S 描述子和门槛。相对 RAFT-Large，pair 覆盖从 `74.1%` 降至 `42.4%`，
action-response Spearman 从 `0.586` 变为 `0.594`，方向准确率从 `69.4%` 变为
`69.3%`，其 CI 下界由 `61.5%` 降至 `58.2%`。真实帧全 interval 可用率也从
`50.2%` 降至 `35.3%`。因此现成 SEA-RAFT 不改善本任务，冻结后端仍为
RAFT-Large；后续若训练生成域光流，必须直接优化覆盖和方向/排序，而不是只替换
预训练权重。

### 跨 WAM Adapter 与 Epona common-random smoke

不同 WAM 的尺寸、裁剪、时间步和动作坐标由显式 Adapter 归一化，冻结的 Step 1
与评分门不随模型变化。Adapter 只能执行有 lineage 的候选无关变换；逐帧实际尺寸、
标定源尺寸或 canonical 射线不一致时直接失败，不能用模型专属阈值换取覆盖。

Epona 旧 5 组 probe 有两项生产问题，不能作为跨模型证据：原始 NAVSIM 时间戳间隔
约 `0.500 s`，旧 manifest 却写成 `0.2 s`；此外 `generate_gt_pose_gt_yaw`
内部逐帧采样 `torch.randn`，旧脚本没有在 left/right 间重置随机种子。因此旧 pair
同时改变了动作和扩散噪声。

修正后重新生成 5 组 common-random pair：每组使用完整 8 步源控制，left/right
共享全部自回归扩散随机数；Epona 的 `1920x1080 -> 1024x512` 直接 resize 与
右正 lateral 坐标均由 Adapter 显式转换。冻结 RAFT-Large + FB 门的结果为：

| 时间口径 | 可用 interval | 可评分 pair |
|---|---:|---:|
| 统一 1/2/3/4 s | 6/40 = 15.0% | 0/5 = 0% |
| 原生 0.5 s、前 4 s | 25/80 = 31.25% | 1/5 = 20% |

关闭 FB 仅作诊断时，统一 1 Hz 口径覆盖 `5/5`，且横向方向为 `5/5`，Wilson 95% CI
`[0.566, 1.000]`；action-response Spearman 为 `0.800 [0.111, 1.000]`。该对照说明
Epona 视频中存在命令方向信号，但现有生成域光流可靠性判据无法把它认证为稳定测量。
所以“至少两个 WAM”升级门仍未通过：当前结果定位的是评测器跨生成域覆盖缺陷，
不是 Epona 几何保真度的正式分数。

产物位于 `benchmark_v3_runs/epona_common_random_20260910/`。生成入口为
`scripts/run_epona_common_random_probe.py`，清单 Adapter 为
`tools/build_epona_flow_structure_manifest.py`。

### Step 1-S1.3：生成域覆盖与 yaw 语义对齐

旧 Step 1-S 的主要覆盖损失来自把真实视频上常用的 hard FB 一致性门直接搬到
生成视频。新 pilot 关闭 hard FB，但不把所有流无条件视为可靠：RAFT 最后 8 次
迭代的向量 RMS 以 `1 + flow magnitude` 归一化，只在真实 NAVSIM 视频上冻结
interval 阈值。按 source hash 切分后，真实帧留出集在 493 个可用 interval 上
yaw 方向准确率为 `274/296 = 92.6% [89.0%, 95.0%]`，Spearman 为 `0.973`。

第二个协议错配也同时修正：`horizontal_flow_center` 是 yaw-response 描述子，
旧配置却用它对齐 action trajectory 的 lateral-y 列，且沿用了米制
`minimum_action_delta=0.05`。S1.3 改为 action yaw 列，并沿用 Step 1.2 已冻结
的 material yaw 门 `0.01 rad`。这不是提高分数的字段切换，而是把测量量、参考量
和阈值单位恢复为同一物理语义。

同一 255 个 DriveWAM pair 的 `temporal-fixed` 重跑结果：

| 量 | 时间错位旧运行（撤回） | S1.3 temporal-fixed |
|---|---:|---:|
| 可评分 pair | 255/255 = 100% | 254/255 = 99.6% |
| 可用 interval | 2022/2040 = 99.1% | 2025/2040 = 99.3% |
| action-response Spearman | 0.713 [0.626, 0.791] | 0.779 [0.706, 0.841] |
| material 方向准确率 | 116/135 = 85.9% | 126/149 = 84.6% [77.9%, 89.5%] |

相对同一次 temporal-fixed 视频上的 SE(2) Step 1.2，S1.3 把 pair coverage 从
`171/255 = 67.1%` 提到 `254/255 = 99.6%`，material yaw pair 从 106 提到
149；方向准确率相近（`82.1%` -> `84.6%`），Spearman 从 `0.832` 降到
`0.779`。因此它解决的是覆盖，不是让每个已覆盖样本更准。当前执行策略已冻结为：
S1.3 是唯一 Step 1 主执行路径；Step 1.2 与 1.3-G 只作为米制/拟合诊断，不能与
S1.3 级联、融合或择优。这个工程选择不等于通过通用 primary 升级门；同源、
同干预分布下的两模型可分离性仍待验证。

覆盖率不再是当前绑定约束，但小 yaw 干预仍有清晰分辨率下限：

| `|Delta yaw_action|` 门 | material pair | 方向准确率 |
|---:|---:|---:|
| >= 0.005 rad | 223 | 170/223 = 76.2% [70.2%, 81.3%] |
| >= 0.010 rad | 149 | 126/149 = 84.6% [77.9%, 89.5%] |
| >= 0.020 rad | 105 | 101/105 = 96.2% [90.6%, 98.5%] |
| >= 0.050 rad | 86 | 86/86 = 100% [95.7%, 100%] |

因此必须并列报告两种 coverage：计算 coverage 为 99.6%；在冻结 0.01 rad
分辨率之上的有效干预 coverage 为 `149/255 = 58.4%`。前者不能替代后者。

Epona common-random probe 扩为同一日志中的 40 个不重叠窗口后，冻结门保留
`318/320 = 99.4%` interval，`40/40` pair 可评分，yaw 方向为
`38/40 = 95.0% [83.5%, 98.6%]`。但 Epona 当前每个 pair 使用同样大小的
干预，action delta 的数值跨度只有约 `1.5e-8 rad`；这只是 float32 积分噪声，
不能计算排序。scorer 已改为把这种 nominally constant reference 的 Spearman
标为 unavailable，而不是报告伪相关。

随后完成了严格匹配对照。255 个 DriveWAM source 中，174 个在同一 scene 内保留
Epona 原生所需的 10 帧历史；该子集包含 75 个 lateral-turn、39 个 braking、
37 个 acceleration、19 个 straight-cruise 和 4 个 stop。Epona 不再使用自造的
固定左右扰动，而是逐 source 接收 DriveWAM 左右分支实际使用的同一组 8 点 SE(2)
动作轨迹，并在每对内共享随机数。

| 冻结 S1.3，同一 174 source | DriveWAM | Epona |
|---|---:|---:|
| pair coverage | 174/174 = 100% | 170/174 = 97.7% |
| 共同 material pair 方向准确率 | 90/105 = 85.7% | 97/105 = 92.4% |
| 共同 scored pair Spearman | 0.805 | 0.739 |

Epona 减 DriveWAM 的配对方向准确率差为 `+0.067 [0.000, 0.133]`，Spearman 差为
`-0.066 [-0.183, 0.040]`，均未排除零。结论因此分成两层：S1.3 已通过跨模型
可迁移性检查，但没有证明能区分这两个 WAM 的几何质量。`100%` 与 `97.7%` 的
coverage 差只表示 Epona 多弃权 4 对，不能替代质量分离。由于质量差异检验是在
看过单模型结果后补充，本轮报告为 post-hoc 证据，不用于 promotion。

自然生成的两个 WAM 未必具有可检测的质量差，因此“必须把任意两个模型排出高低”
不是有效性本身。更直接的阳性对照是在保持 source、动作、历史帧、标定、RAFT 和
阈值不变时，只把左右未来视频向逐像素中点收缩。该对照在运行前已于 commit
`f3b65c6` 预注册，强度为 `0 / 0.5 / 1.0`；强度 1.0 时两支未来帧逐字节相同。
这是仅用于验证的 pair-dependent 破坏，不进入正式评分协议。
原配置中的模型分离 promotion gate 仍记为失败，不因本轮阳性对照而事后改判；
本轮建立的是另一个、问题定义更直接的 action-response 有效性证据。

| 受控左右视频差异 | DriveWAM coverage | DriveWAM 响应中位数 | Epona coverage | Epona 响应中位数 |
|---:|---:|---:|---:|---:|
| 100% | 100.0% | 0.004854 | 97.7% | 0.005917 |
| 50% | 100.0% | 0.003357 | 98.3% | 0.002926 |
| 0% | 100.0% | 0 | 98.3% | 0 |

两模型均在 coverage 不低于 90% 的同时满足响应严格递减；完整消除左右视频差异后，
响应归零、material direction pair 归零、Spearman 按契约变为 unavailable。非视频
输入指纹在三档间完全一致。因此 S1.3 已通过两项独立有效性检查：跨 WAM 可迁移，
且能在输入仍可测时识别已知的反事实视觉信号丢失；它不是靠增加弃权制造差异。

据此，S1.3 可作为 **action-response 测量器** 使用。边界仍不变：该阳性对照不证明
它能按自然生成质量给 DriveWAM 与 Epona 排名，也不测 logged-GT 几何保真度。
冻结配置不改写，以保持预注册记录中的 SHA；验证状态单独记录在
`configs/flow_structure_yaw_v1_3_validation.json`。

S1.3 已被选为当前唯一 Step 1，并通过 action-response 测量器的跨模型与阳性对照
验证；自然 WAM 质量排序仍未建立，也不作为有效性的必要条件。冻结执行配置为
`configs/flow_structure_yaw_v1_3.json`；原 pilot 配置仅作来源留档，真实域阈值记录为
`configs/raft_refinement_uncertainty_real_navsim_v1.json`。

PhysicalAI DriveWAM 权重不能充当第二模型对照：兼容性检查显示其 checkpoint 不含
NAVSIM 的 `ego_history_mlp` 与 `ego_command_embed`，加载时会随机初始化这些条件层，
同时遗留未使用的 curvature 层。继续推理会把随机条件层误当成 WAM 差异，因此该
对照已在单样本生成前停止，不进入任何指标。

### 内参修复后的能量复核

在同一 `1920x1080 -> 448x256` 标定下，真实帧的 logged-GT flow energy 均值为
`0.940`，生成帧为 `2.064`；按 source 聚类 bootstrap，生成减真实的均值差为
`+1.115 [1.032, 1.213]`，`85.3%` 的配对区间生成帧更差。相反，生成帧的
action-head energy 均值为 `1.570`，与真实帧 `1.496` 的差异置信区间为
`[-0.048, 0.174]`，跨越零。

因此当前可支持的表述是：**生成视频包含可由自身 action 条件解释的刚体变化，
但其运动几何显著偏离 logged GT**。这不是“生成视频没有刚体运动”，也不能推出
它已经正确实现了动作；action-response 的方向/排序与 logged-GT 准确性必须分开报告。
在 lateral-turn 区间，生成相对真实的 logged-GT 能量差均值为 `+1.297`，说明
该偏离在协议重点场景尤其明显。

逐 interval 审计还显示，整条 `explained` 不能代表四个时刻都各自可识别。时间
契约修正后的 G 结果中，两支共同 explained 的覆盖为 `4/4: 63.1%`、
`>=3/4: 81.2%`、`>=2/4: 89.0%`、`>=1/4: 94.1%`。在 `>=2/4` 上仅汇总共同
interval 的 yaw 增量，方向准确率为 `108/127 = 85.0% [77.8%, 90.2%]`，
Spearman 为 `0.949`。缺失时刻不插补 terminal yaw；`>=1/4` 虽越过 90% 覆盖，
单个时刻不足以支撑四秒轨迹声明，因此部分时域仍只作独立诊断。

私有实验产物：`benchmark_v3_runs/flow_structure_v1_20260910/`，完整统计为
`validation_report.json`。该 action 对齐只检验 WAM 是否响应其条件动作，不等价于
生成视频符合 logged GT，也不等价于能按 PDM 做 policy ranking。

修正内参后的能量与自由拟合产物分别位于
`benchmark_v3_runs/energy_intrinsics_corrected_20260910/` 和
`benchmark_v3_runs/rigid_fit_intrinsics_corrected_20260910/`；SEA-RAFT 对照位于
`benchmark_v3_runs/flow_structure_sea_raft_20260910/`。

对应入口为 `scripts/evaluate_flow_structure.py`（逐分支测量）和
`scripts/score_flow_structure_alignment.py`（左右 pair 排序）；冻结参数在
`configs/flow_structure.json`。

## Step 1 / CFAC 与 FAU（修复前历史回溯值）

本节数值来自发布版 decoder 的历史产物。后续审计发现发布版在无有效投影时会返回恒定目标值，且旧下游 gate 只读取输入侧 observability；因此下表不是修复后的有效 benchmark 分数，也不能用于模型比较。修复后的评测必须使用 `post_projection_abstention` coverage，并重新计算均值与置信区间。

官方路径从 WAM 生成图像重新运行探针，未读取作者提交的运动剖面；GT 只在私有评测端 join。

| 指标 | 结果 |
|---|---:|
| CFAC primary shape composite | 0.7638 |
| CFAC 有效样本 | 823/1000 |
| CFAC bootstrap 95% CI | 0.7490--0.7785 |
| FAU-F（想象图像 vs 私有 GT） | 0.5449 |
| FAU-A（native action vs 私有 GT） | 0.4904 |
| FAU 几何均值 | 0.5169 |
| FAU/CFAC 有效样本 | 823/1000 |

历史 coverage `823/1000` 是 pre-fix gate 的统计，不是投影支持后的 coverage。

停车样本不进入运动均值；4 条样本因私有 GT 没有与公共 `[1,2,3,4] s` 完全一致的时间轴而标记 `unavailable`，没有插值或近邻替代。

逐样本结果由私有评测端保存为 `drivewam_cfac_fau.json`，不随公开仓库发布。

## Step 2 / CCFC（修复前历史回溯值）

对每个样本使用相同 history、seed 和 nuisance，仅改变导航命令为 left/right，各运行一次；两支均同时保留 WAM 生成 future images 与 native action。两支图像都通过冻结 Level-1 探针后再评分。

| 指标 | 结果 |
|---|---:|
| 成对组数 | 1000 |
| 结构有效组 | 1000/1000 |
| metric CCFC（诊断） | 0.1190，453 对 |
| scale-free CCFC | 0.1594，453 对 |
| arc-relative CCFC（主报告） | 0.2178，453 对 |
| claim scope | `command_conditioned_action_image_consistency` |

历史 `453` 对同样未应用输出投影支持弃权，不能作为修复后 CCFC 的有效分母。

这不是 semantic clear/risk 干预，因此不宣称语义危险因果；它是统一协议允许的 command-conditioned CCFC。

逐样本记录与评分由私有评测端保存为 `ccfc_full_records.jsonl`、
`ccfc_full_report.json`，不随公开仓库发布。

## Step 3 / FCS

native action 进入独立 NAVSIM PDM kinematic-bicycle closed-loop rollout。该步骤测量可执行动作的实现效果，不读取 WAM 生成图像，也不把 waypoint 当作 realized state。

| 指标 | 结果 |
|---|---:|
| 输入/成功 | 978 / 503 |
| FCS task success rate | 0.5143 |
| state reference | `navsim_pdm_kinematic_bicycle_closed_loop` |
| traffic policy | `static_cached_objects_compat` |

## 可复现实验命令

核心评分命令如下；尖括号表示运行者自己的数据路径：

```bash
PYTHONPATH=<repo_root>/src \
python scripts/evaluate_cfac_fau.py \
  --level1-input <joined_input.jsonl> \
  --level1-scores <measurement_scores.jsonl> \
  --private-manifest <private_benchmark.jsonl> \
  --model-id drivewam_navsim_checkpoint_20260824 \
  --output <cfac_fau.json>
```

所有评分状态采用 fail-closed：缺少图像、native action、私有 GT 或时间轴时报告 `unavailable`，不填零分。
