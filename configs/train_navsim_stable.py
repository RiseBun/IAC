from configs.train_consistency_mini import cfg as base

cfg = dict(base)
cfg["experiment_name"] = "iac_navsim_stable"
cfg["batch_size"] = 2
cfg["num_workers"] = 0
cfg["persistent_workers"] = False
cfg["prefetch_factor"] = 2

# 先关掉 group-ranking sampler，降低内存和 batch 波动；跑稳后再打开。
cfg["lambda_group_ranking"] = 0.0
cfg["ranking"] = dict(cfg.get("ranking", {}))
cfg["ranking"]["enabled"] = False
cfg["ranking"]["group_batches"] = False
