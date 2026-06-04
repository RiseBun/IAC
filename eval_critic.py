#!/usr/bin/env python3
"""IAC Consistency Critic 模型评估脚本

用法:
    # 评估 Consistency Critic 模型
    python eval_critic.py --checkpoint work_dirs/iac_full/checkpoints/best.pth

    # 限制评估样本数
    python eval_critic.py --checkpoint work_dirs/iac_full/checkpoints/best.pth --max-samples 100
    
    # Ranking 评估（需要索引中包含 ranking_groups）
    python eval_critic.py --checkpoint work_dirs/iac_full/checkpoints/best.pth --eval-ranking
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from train import ConsistencyDataset, ConsistencyCriticModel, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 Critic 模型")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint 文件路径 (.pth)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="配置文件路径（默认从 checkpoint 中读取）",
    )
    parser.add_argument(
        "--split",
        choices=["val", "train"],
        default="val",
        help="评估数据集划分 (默认: val)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="评估 batch size (默认: 32)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="最多评估样本数，0 表示全部 (默认: 0)",
    )
    parser.add_argument(
        "--eval-ranking",
        action="store_true",
        help="是否评估 ranking 能力（NDCG, MRR, Top-k）",
    )
    parser.add_argument(
        "--baseline-mode",
        choices=["full", "no_image", "ego_only", "no_traj", "traj_only"],
        default=None,
        help="覆盖 checkpoint/config 中的 P0 baseline mode",
    )
    parser.add_argument(
        "--max-ranking-groups",
        type=int,
        default=0,
        help="Maximum group_id groups for ranking evaluation; 0 means all.",
    )
    return parser.parse_args()


def _compute_ece(probs: torch.Tensor, labels: torch.Tensor, num_bins: int = 10) -> float:
    """Expected Calibration Error."""
    if labels.numel() == 0:
        return 0.0
    ece = torch.tensor(0.0)
    for i in range(num_bins):
        lo = i / num_bins
        hi = (i + 1) / num_bins
        if i == num_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if mask.any():
            conf = probs[mask].mean()
            acc = labels[mask].mean()
            ece += mask.float().mean() * torch.abs(conf - acc)
    return float(ece.item())


def _safe_average_precision(labels: torch.Tensor, probs: torch.Tensor) -> float | None:
    if labels.numel() == 0 or labels.unique().numel() < 2:
        return None
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(labels.numpy(), probs.numpy()))
    except ImportError:
        order = torch.argsort(probs, descending=True)
        sorted_labels = labels[order]
        positives = sorted_labels.sum().item()
        if positives <= 0:
            return None
        tp = torch.cumsum(sorted_labels, dim=0)
        ranks = torch.arange(1, sorted_labels.numel() + 1, dtype=torch.float32)
        precision_at_k = tp / ranks
        return float((precision_at_k * sorted_labels).sum().item() / positives)


def _compute_head_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, Any]:
    """计算单个 head 的详细指标"""
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    pos_mask = labels == 1.0
    neg_mask = labels == 0.0
    num_pos = pos_mask.sum().item()
    num_neg = neg_mask.sum().item()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None
        and (precision + recall) > 0
        else None
    )
    accuracy = (preds == labels).float().mean().item()

    # TNR / FPR
    tnr = tn / (tn + fp) if (tn + fp) > 0 else None
    fpr = fp / (fp + tn) if (fp + tn) > 0 else None

    # AUC 计算
    auc: float | None = None
    pr_auc: float | None = None
    if num_pos > 0 and num_neg > 0:
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(
                labels.numpy(), probs.numpy(),
            )
        except ImportError:
            # 简易 AUC: 正样本概率 > 负样本概率的比例
            pos_p = probs[pos_mask]
            neg_p = probs[neg_mask]
            comparisons = (
                pos_p.unsqueeze(1) > neg_p.unsqueeze(0)
            ).float().mean().item()
            auc = comparisons
        pr_auc = _safe_average_precision(labels, probs)

    pos_probs = probs[pos_mask].numpy() if num_pos > 0 else np.array([])
    neg_probs = probs[neg_mask].numpy() if num_neg > 0 else np.array([])

    return {
        "num_positive": int(num_pos),
        "num_negative": int(num_neg),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tnr": tnr,
        "fpr": fpr,
        "auc": auc,
        "pr_auc": pr_auc,
        "ece": _compute_ece(probs, labels),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "pos_prob_mean": float(pos_probs.mean()) if len(pos_probs) > 0 else 0.0,
        "neg_prob_mean": float(neg_probs.mean()) if len(neg_probs) > 0 else 0.0,
    }


def evaluate_consistency(
    model: nn.Module,
    dataset: "ConsistencyDataset",
    device: torch.device,
    batch_size: int,
    max_samples: int,
) -> Dict[str, Any]:
    """评估 Consistency Critic 模型，返回双头指标和 per-source-type 分组统计"""
    from torch.utils.data import DataLoader
    from collections import defaultdict

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    model.eval()

    all_c_logits: list = []
    all_v_logits: list = []
    all_c_labels: list = []
    all_v_labels: list = []
    all_source_types: list = []
    sample_meta: list = []
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            h_imgs = batch["history_images"].to(device, non_blocking=True)
            f_imgs = batch["future_images"].to(device, non_blocking=True)
            ego = batch["ego_state"].to(device, non_blocking=True)
            traj = batch["candidate_traj"].to(device, non_blocking=True)
            c_labels = batch["consistency_label"]
            v_labels = batch["validity_label"]

            out = model(h_imgs, f_imgs, ego, traj)
            all_c_logits.append(out["consistency_logit"].cpu())
            all_v_logits.append(out["validity_logit"].cpu())
            all_c_labels.append(c_labels)
            all_v_labels.append(v_labels)

            # 收集 source_type
            start = batch_idx * batch_size
            end = min(start + len(c_labels), len(dataset))
            for i in range(start, end):
                st = dataset.samples[i].get("source_type", "unknown")
                all_source_types.append(st)
                sample_meta.append(dataset.samples[i])

            total_samples += len(c_labels)
            if (batch_idx + 1) % 20 == 0:
                print(
                    f"[Eval] step={batch_idx + 1}/{len(loader)} "
                    f"samples={total_samples}",
                    flush=True,
                )
            if max_samples and total_samples >= max_samples:
                break

    c_logits = torch.cat(all_c_logits)[:total_samples]
    v_logits = torch.cat(all_v_logits)[:total_samples]
    c_labels = torch.cat(all_c_labels)[:total_samples]
    v_labels = torch.cat(all_v_labels)[:total_samples]
    source_types = all_source_types[:total_samples]
    sample_meta = sample_meta[:total_samples]

    # 整体指标
    consistency_metrics = _compute_head_metrics(c_logits, c_labels)
    validity_metrics = _compute_head_metrics(v_logits, v_labels)

    # per-source-type 分组指标
    source_groups: Dict[str, Dict[str, list]] = defaultdict(
        lambda: {"c_logits": [], "v_logits": [], "c_labels": [], "v_labels": []},
    )
    for i, st in enumerate(source_types):
        source_groups[st]["c_logits"].append(c_logits[i])
        source_groups[st]["v_logits"].append(v_logits[i])
        source_groups[st]["c_labels"].append(c_labels[i])
        source_groups[st]["v_labels"].append(v_labels[i])

    per_source: Dict[str, Dict] = {}
    for st, data in sorted(source_groups.items()):
        st_c_logits = torch.stack(data["c_logits"])
        st_c_labels = torch.stack(data["c_labels"])
        st_v_logits = torch.stack(data["v_logits"])
        st_v_labels = torch.stack(data["v_labels"])
        per_source[st] = {
            "count": len(data["c_logits"]),
            "consistency": _compute_head_metrics(st_c_logits, st_c_labels),
            "validity": _compute_head_metrics(st_v_logits, st_v_labels),
        }

    negative_recall_by_type = {
        st: data["consistency"]["tnr"]
        for st, data in per_source.items()
        if data["consistency"]["num_negative"] > 0
    }

    c_probs = torch.sigmoid(c_logits)
    graded_groups: Dict[str, Dict[str, list]] = defaultdict(lambda: {"probs": [], "magnitudes": []})
    for i, meta in enumerate(sample_meta):
        if not str(meta.get("source_type", "")).startswith("perturb_"):
            continue
        ptype = meta.get("perturb_type", meta.get("source_type", "perturb").replace("perturb_", ""))
        level = meta.get("perturb_level", "unknown")
        key = f"{ptype}:{level}"
        graded_groups[key]["probs"].append(float(c_probs[i].item()))
        if "perturb_magnitude" in meta:
            graded_groups[key]["magnitudes"].append(float(meta["perturb_magnitude"]))

    graded_curve = {}
    for key, data in sorted(graded_groups.items()):
        probs = data["probs"]
        mags = data["magnitudes"]
        graded_curve[key] = {
            "count": len(probs),
            "mean_consistency_prob": float(np.mean(probs)) if probs else 0.0,
            "mean_perturb_magnitude": float(np.mean(mags)) if mags else None,
        }

    return {
        "total_samples": total_samples,
        "consistency": consistency_metrics,
        "validity": validity_metrics,
        "per_source_type": per_source,
        "negative_recall_by_type": negative_recall_by_type,
        "graded_perturbation_curve": graded_curve,
    }


def _print_head_metrics(name: str, m: Dict[str, Any], indent: str = "  ") -> None:
    """打印单个 head 的评估指标"""
    print(f"{indent}[{name}]")
    print(f"{indent}  正/负样本数: {m['num_positive']} / {m['num_negative']}")
    print(f"{indent}  Accuracy:  {m['accuracy']:.4f}")
    if m['num_positive'] > 0:
        p = m['precision']
        r = m['recall']
        f1 = m['f1_score']
        print(f"{indent}  Precision: {p:.4f}" if p is not None else f"{indent}  Precision: N/A")
        print(f"{indent}  Recall:    {r:.4f}" if r is not None else f"{indent}  Recall:    N/A")
        print(f"{indent}  F1 Score:  {f1:.4f}" if f1 is not None else f"{indent}  F1 Score:  N/A")
    else:
        print(f"{indent}  (无正样本，Precision/Recall/F1 不适用)")
    if m['num_negative'] > 0:
        tnr = m.get('tnr')
        fpr = m.get('fpr')
        print(f"{indent}  TNR:       {tnr:.4f}" if tnr is not None else f"{indent}  TNR:       N/A")
        print(f"{indent}  FPR:       {fpr:.4f}" if fpr is not None else f"{indent}  FPR:       N/A")
    else:
        print(f"{indent}  (无负样本，TNR/FPR 不适用)")
    if m.get('auc') is not None:
        print(f"{indent}  AUC:       {m['auc']:.4f}")
    else:
        print(f"{indent}  AUC:       N/A (需要同时有正负样本)")
    if m.get('pr_auc') is not None:
        print(f"{indent}  PR-AUC:    {m['pr_auc']:.4f}")
    else:
        print(f"{indent}  PR-AUC:    N/A (需要同时有正负样本)")
    print(f"{indent}  ECE:       {m.get('ece', 0.0):.4f}")
    print(f"{indent}  TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, TN={m['tn']}")
    print(f"{indent}  正样本概率均值: {m['pos_prob_mean']:.4f}")
    print(f"{indent}  负样本概率均值: {m['neg_prob_mean']:.4f}")


def _format_source_line(m: Dict[str, Any]) -> str:
    """根据子集的正负样本情况，智能选择展示的指标"""
    parts = [f"acc={m['accuracy']:.4f}"]
    if m['num_positive'] > 0 and m['f1_score'] is not None:
        parts.append(f"f1={m['f1_score']:.4f}")
    if m['num_negative'] > 0 and m.get('tnr') is not None:
        parts.append(f"tnr={m['tnr']:.4f}")
    if m.get('auc') is not None:
        parts.append(f"auc={m['auc']:.4f}")
    return " ".join(parts)


def compute_ranking_metrics(
    model: nn.Module,
    dataset: "ConsistencyDataset",
    device: torch.device,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """评估 Consistency Critic 的 Ranking 能力
    
    对于同一 history 的多个候选轨迹，评估模型是否能正确排序
    Metrics: NDCG@k, MRR, Top-1 Hit Rate
    """
    from torch.utils.data import DataLoader
    
    # 按 scene 分组
    scene_groups = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        scene_name = sample.get("scene_name", "unknown")
        timestamp = sample.get("timestamp_us", idx)
        scene_groups[scene_name].append({
            "index": idx,
            "timestamp": timestamp,
            "consistency_label": sample.get("consistency_label", 0),
            "validity_label": sample.get("validity_label", 0),
        })
    
    # 过滤出有多个候选的 scenes
    multi_candidate_scenes = {
        scene: samples for scene, samples in scene_groups.items()
        if len(samples) >= 2
    }
    
    if not multi_candidate_scenes:
        print("[WARNING] 没有找到多候选场景，跳过 ranking 评估")
        return {}
    
    print(f"\n[Ranking Evaluation] 找到 {len(multi_candidate_scenes)} 个多候选场景")
    
    model.eval()
    all_ndcg_3 = []
    all_ndcg_5 = []
    all_mrr = []
    all_top1_hit = []
    
    with torch.no_grad():
        for scene_idx, (scene_name, candidates) in enumerate(multi_candidate_scenes.items(), start=1):
            # 收集该 scene 的所有样本
            scores = []
            relevances = []  # GT relevance (consistency_label)
            
            for cand in candidates:
                idx = cand["index"]
                sample = dataset[idx]
                
                h_imgs = sample["history_images"].unsqueeze(0).to(device)
                f_imgs = sample["future_images"].unsqueeze(0).to(device)
                ego = sample["ego_state"].unsqueeze(0).to(device)
                traj = sample["candidate_traj"].unsqueeze(0).to(device)
                
                out = model(h_imgs, f_imgs, ego, traj)
                score = torch.sigmoid(out["consistency_logit"]).item()
                
                scores.append(score)
                relevances.append(cand["consistency_label"])
            
            # 计算 NDCG@k
            def compute_ndcg(scores_list, relevance_list, k):
                if len(scores_list) < 2:
                    return 0.0
                
                # 按分数排序
                sorted_pairs = sorted(zip(scores_list, relevance_list), reverse=True)
                sorted_relevances = [rel for _, rel in sorted_pairs[:k]]
                
                # DCG
                dcg = sum(
                    rel / np.log2(i + 2) for i, rel in enumerate(sorted_relevances)
                )
                
                # Ideal DCG
                ideal_relevances = sorted(relevance_list, reverse=True)[:k]
                idcg = sum(
                    rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances)
                )
                
                return dcg / idcg if idcg > 0 else 0.0
            
            # 计算 MRR
            def compute_mrr(scores_list, relevance_list):
                if len(scores_list) < 2:
                    return 0.0
                
                # 按分数排序
                sorted_pairs = sorted(zip(scores_list, relevance_list), reverse=True)
                
                # 找到第一个正样本的位置
                for i, (_, rel) in enumerate(sorted_pairs):
                    if rel == 1:
                        return 1.0 / (i + 1)
                return 0.0
            
            # 计算 Top-1 Hit Rate
            def compute_top1_hit(scores_list, relevance_list):
                if len(scores_list) < 2:
                    return 0.0
                
                # 找到分数最高的样本
                best_idx = np.argmax(scores_list)
                return 1.0 if relevance_list[best_idx] == 1 else 0.0
            
            # 累积指标
            all_ndcg_3.append(compute_ndcg(scores, relevances, k=3))
            all_ndcg_5.append(compute_ndcg(scores, relevances, k=5))
            all_mrr.append(compute_mrr(scores, relevances))
            all_top1_hit.append(compute_top1_hit(scores, relevances))
            if scene_idx % 10 == 0 or scene_idx == len(multi_candidate_scenes):
                print(
                    f"[Ranking] scene={scene_idx}/{len(multi_candidate_scenes)} "
                    f"candidates={len(candidates)}",
                    flush=True,
                )
    
    return {
        "ndcg@3": float(np.mean(all_ndcg_3)) if all_ndcg_3 else 0.0,
        "ndcg@5": float(np.mean(all_ndcg_5)) if all_ndcg_5 else 0.0,
        "mrr": float(np.mean(all_mrr)) if all_mrr else 0.0,
        "top1_hit_rate": float(np.mean(all_top1_hit)) if all_top1_hit else 0.0,
        "num_scenes": len(multi_candidate_scenes),
    }

def compute_ranking_metrics(
    model: nn.Module,
    dataset: "ConsistencyDataset",
    device: torch.device,
    batch_size: int = 32,
    max_groups: int = 0,
    group_key: str = "group_id",
) -> Dict[str, Any]:
    """Evaluate ranking within group_id candidate sets."""

    ranking_groups = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        group_id = (
            sample.get(group_key)
            or sample.get("anchor_id")
            or f"{sample.get('scene_name', 'unknown')}::{sample.get('timestamp_us', idx)}"
        )
        ranking_groups[str(group_id)].append(
            {
                "index": idx,
                "consistency_label": sample.get("consistency_label", 0),
            }
        )

    groups = [
        (gid, items)
        for gid, items in ranking_groups.items()
        if len(items) >= 2 and any(item["consistency_label"] == 1 for item in items)
    ]
    if max_groups and max_groups > 0:
        groups = groups[:max_groups]
    if not groups:
        print("[WARNING] No multi-candidate ranking groups found; skipping ranking.")
        return {}

    print(f"\n[Ranking Evaluation] groups={len(groups)} group_key={group_key}")

    def ndcg(scores, labels, k):
        order = np.argsort(scores)[::-1][:k]
        gains = np.asarray(labels, dtype=np.float64)[order]
        discounts = 1.0 / np.log2(np.arange(len(gains)) + 2)
        dcg = float(np.sum(gains * discounts))
        ideal = np.sort(np.asarray(labels, dtype=np.float64))[::-1][:k]
        idcg = float(np.sum(ideal * discounts[: len(ideal)]))
        return dcg / idcg if idcg > 0 else 0.0

    top1_hits, mrrs, ndcg3, ndcg5 = [], [], [], []
    model.eval()
    with torch.no_grad():
        for group_idx, (group_id, candidates) in enumerate(groups, start=1):
            scores = []
            labels = [float(item["consistency_label"]) for item in candidates]

            for start in range(0, len(candidates), batch_size):
                chunk = candidates[start:start + batch_size]
                samples = [dataset[item["index"]] for item in chunk]
                h_imgs = torch.stack([s["history_images"] for s in samples]).to(device)
                f_imgs = torch.stack([s["future_images"] for s in samples]).to(device)
                ego = torch.stack([s["ego_state"] for s in samples]).to(device)
                traj = torch.stack([s["candidate_traj"] for s in samples]).to(device)
                out = model(h_imgs, f_imgs, ego, traj)
                scores.extend(
                    torch.sigmoid(out["consistency_logit"]).detach().cpu().tolist()
                )

            order = np.argsort(scores)[::-1]
            sorted_labels = np.asarray(labels, dtype=np.float64)[order]
            top1_hits.append(float(sorted_labels[0] > 0))
            first_pos = np.where(sorted_labels > 0)[0]
            mrrs.append(1.0 / float(first_pos[0] + 1) if len(first_pos) else 0.0)
            ndcg3.append(ndcg(scores, labels, 3))
            ndcg5.append(ndcg(scores, labels, 5))

            if group_idx % 100 == 0 or group_idx == len(groups):
                print(
                    f"[Ranking] group={group_idx}/{len(groups)} "
                    f"candidates={len(candidates)} group_id={group_id}",
                    flush=True,
                )

    return {
        "num_groups": len(groups),
        "num_scenes": len(groups),
        "ndcg@3": float(np.mean(ndcg3)) if ndcg3 else 0.0,
        "ndcg@5": float(np.mean(ndcg5)) if ndcg5 else 0.0,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
        "top1_hit_rate": float(np.mean(top1_hits)) if top1_hits else 0.0,
    }


def main() -> None:
    args = parse_args()

    # 加载 checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}")

    print(f"加载 checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 加载配置
    if args.config:
        cfg = load_config(args.config)
    elif "config" in checkpoint:
        cfg = checkpoint["config"]
    else:
        raise ValueError("Checkpoint 中无 config，请用 --config 指定配置文件")
    if args.baseline_mode is not None:
        cfg["baseline_mode"] = args.baseline_mode

    epoch = checkpoint.get("epoch", "?")
    best_val_loss = checkpoint.get("best_val_loss", "?")
    model_type = cfg.get("model_type")
    if model_type != "consistency":
        raise ValueError(
            "新版 eval_critic.py 只支持 IAC checkpoint "
            "(config.model_type 必须为 'consistency')。旧 critic checkpoint 请重新训练新版 IAC。"
        )
    print(f"Checkpoint 信息: epoch={epoch}, best_val_loss={best_val_loss}")
    print("模型类型: consistency")
    print(f"Baseline mode: {cfg.get('baseline_mode', 'full')}")

    # 构建模型并加载权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConsistencyCriticModel(cfg).to(device)
    
    model_state = checkpoint["model"]
    model.load_state_dict(model_state, strict=True)
    print("  权重严格匹配")
    
    print(f"模型加载完成，设备: {device}")

    # 构建数据集
    index_key = "val_index" if args.split == "val" else "train_index"
    index_path = cfg[index_key]
    print(f"数据集: {args.split} ({index_path})")

    dataset = ConsistencyDataset(
        index_path=index_path, cfg=cfg, training=False,
    )
    print(f"样本总数: {len(dataset)}")

    # 评估
    print("\n开始评估...")

    metrics = evaluate_consistency(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
    print("\n" + "=" * 60)
    print("IAC Consistency Critic 评估结果")
    print("=" * 60)
    print(f"  总样本数: {metrics['total_samples']}")
    _print_head_metrics("Consistency Head", metrics["consistency"])
    _print_head_metrics("Validity Head", metrics["validity"])

    if metrics.get("per_source_type"):
        print("\n  [Per Source Type]")
        for st, st_data in metrics["per_source_type"].items():
            print(f"    --- {st} (n={st_data['count']}) ---")
            c = st_data["consistency"]
            v = st_data["validity"]
            print(f"      consistency: {_format_source_line(c)}")
            print(f"      validity:    {_format_source_line(v)}")
    if metrics.get("negative_recall_by_type"):
        print("\n  [Negative Recall / TNR by Type]")
        for st, value in metrics["negative_recall_by_type"].items():
            print(f"    {st}: {value:.4f}" if value is not None else f"    {st}: N/A")
    if metrics.get("graded_perturbation_curve"):
        print("\n  [Graded Perturbation Curve]")
        for key, data in metrics["graded_perturbation_curve"].items():
            print(
                f"    {key}: n={data['count']} "
                f"mean_prob={data['mean_consistency_prob']:.4f} "
                f"mean_mag={data['mean_perturb_magnitude']}"
            )
    print("=" * 60)

    # 保存结果到 JSON
    result_path = ckpt_path.parent.parent / f"eval_{args.split}_results.json"
    
    # 如果启用 ranking 评估
    if args.eval_ranking:
        print("\n" + "=" * 60)
        print("开始 Ranking 评估...")
        print("=" * 60)
        ranking_metrics = compute_ranking_metrics(
            model=model,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
            max_groups=args.max_ranking_groups,
        )
        
        if ranking_metrics:
            print("\n[Ranking Metrics]")
            print(f"  场景数: {ranking_metrics['num_scenes']}")
            print(f"  NDCG@3:  {ranking_metrics['ndcg@3']:.4f}")
            print(f"  NDCG@5:  {ranking_metrics['ndcg@5']:.4f}")
            print(f"  MRR:     {ranking_metrics['mrr']:.4f}")
            print(f"  Top-1 Hit Rate: {ranking_metrics['top1_hit_rate']:.4f}")
            print("=" * 60)
            
            # 合并到结果中
            metrics["ranking"] = ranking_metrics
    
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {result_path}")

    summary_path = ckpt_path.parent.parent / f"eval_{args.split}_summary.json"
    summary = {
        "total_samples": metrics["total_samples"],
        "baseline_mode": cfg.get("baseline_mode", "full"),
        "consistency": {
            k: metrics["consistency"].get(k)
            for k in ("accuracy", "auc", "pr_auc", "ece", "f1_score", "tnr", "fpr")
        },
        "validity": {
            k: metrics["validity"].get(k)
            for k in ("accuracy", "auc", "pr_auc", "ece", "f1_score", "tnr", "fpr")
        },
        "negative_recall_by_type": metrics.get("negative_recall_by_type", {}),
        "graded_perturbation_curve": metrics.get("graded_perturbation_curve", {}),
        "ranking": metrics.get("ranking", {}),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"摘要已保存: {summary_path}")


if __name__ == "__main__":
    main()
