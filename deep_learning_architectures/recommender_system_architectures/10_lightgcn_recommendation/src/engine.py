from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.utils import ndcg_at_k, recall_at_k, save_checkpoint, save_json


def bpr_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    user_initial_embeddings: torch.Tensor,
    positive_initial_embeddings: torch.Tensor,
    negative_initial_embeddings: torch.Tensor,
    bpr_reg_weight: float,
) -> torch.Tensor:
    ranking_loss = -F.logsigmoid(positive_scores - negative_scores).mean()
    batch_size = positive_scores.size(0)
    reg_loss = (
        user_initial_embeddings.norm(2).pow(2)
        + positive_initial_embeddings.norm(2).pow(2)
        + negative_initial_embeddings.norm(2).pow(2)
    ) / batch_size
    return ranking_loss + bpr_reg_weight * reg_loss


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    normalized_adj: torch.Tensor,
    device: torch.device,
    bpr_reg_weight: float,
) -> dict[str, float]:
    model.train()
    total_loss_sum = 0.0
    total_samples = 0

    for user_ids, positive_item_ids, negative_item_ids in tqdm(train_loader, leave=False):
        user_ids = user_ids.to(device)
        positive_item_ids = positive_item_ids.to(device)
        negative_item_ids = negative_item_ids.to(device)

        optimizer.zero_grad()
        positive_scores, negative_scores = model(
            user_ids=user_ids,
            positive_item_ids=positive_item_ids,
            negative_item_ids=negative_item_ids,
            normalized_adj=normalized_adj,
        )
        user_init, pos_init, neg_init = model.get_initial_embeddings(
            user_ids,
            positive_item_ids,
            negative_item_ids,
        )
        loss = bpr_loss(
            positive_scores=positive_scores,
            negative_scores=negative_scores,
            user_initial_embeddings=user_init,
            positive_initial_embeddings=pos_init,
            negative_initial_embeddings=neg_init,
            bpr_reg_weight=bpr_reg_weight,
        )
        loss.backward()
        optimizer.step()

        batch_size = user_ids.size(0)
        total_loss_sum += loss.item() * batch_size
        total_samples += batch_size

    return {"loss": total_loss_sum / total_samples}


def group_edges_by_user(edges: pd.DataFrame) -> dict[int, set[int]]:
    grouped: dict[int, set[int]] = {}
    for user_id, user_edges in edges.groupby("user_id", sort=False):
        grouped[int(user_id)] = set(user_edges["item_id"].astype(int).tolist())
    return grouped


def evaluate(
    model: nn.Module,
    edges: pd.DataFrame,
    normalized_adj: torch.Tensor,
    train_user_positive_items: dict[int, set[int]],
    device: torch.device,
    top_k: int,
    eval_max_users: int,
) -> dict[str, float]:
    model.eval()
    user_targets = group_edges_by_user(edges)
    selected_user_ids = sorted(user_targets)[:eval_max_users]
    if not selected_user_ids:
        return {f"recall_at_{top_k}": 0.0, f"ndcg_at_{top_k}": 0.0, "num_users": 0.0}

    recalls = []
    ndcgs = []
    with torch.no_grad():
        user_tensor = torch.tensor(selected_user_ids, dtype=torch.long, device=device)
        all_scores = model.full_sort_scores(user_tensor, normalized_adj)

        for row_index, user_id in enumerate(selected_user_ids):
            scores = all_scores[row_index].clone()
            seen_items = train_user_positive_items.get(user_id, set())
            if seen_items:
                scores[list(seen_items)] = -float("inf")
            recommended_items = torch.topk(scores, k=min(top_k, scores.numel())).indices
            recommended_list = recommended_items.cpu().tolist()
            relevant_items = user_targets[user_id]
            recalls.append(recall_at_k(recommended_list, relevant_items, top_k))
            ndcgs.append(ndcg_at_k(recommended_list, relevant_items, top_k))

    return {
        f"recall_at_{top_k}": float(sum(recalls) / len(recalls)),
        f"ndcg_at_{top_k}": float(sum(ndcgs) / len(ndcgs)),
        "num_users": float(len(selected_user_ids)),
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    valid_edges: pd.DataFrame,
    normalized_adj: torch.Tensor,
    train_user_positive_items: dict[int, set[int]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    bpr_reg_weight: float,
    n_epochs: int,
    top_k: int,
    eval_max_users: int,
    checkpoint_path: Path,
    history_path: Path,
) -> dict[str, list[float]]:
    history: dict[str, list[float]] = {
        "train_bpr_losses": [],
        f"valid_recall_at_{top_k}": [],
        f"valid_ndcg_at_{top_k}": [],
    }
    best_valid_recall = -1.0

    for epoch in range(n_epochs):
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            normalized_adj=normalized_adj,
            device=device,
            bpr_reg_weight=bpr_reg_weight,
        )
        valid_metrics = evaluate(
            model=model,
            edges=valid_edges,
            normalized_adj=normalized_adj,
            train_user_positive_items=train_user_positive_items,
            device=device,
            top_k=top_k,
            eval_max_users=eval_max_users,
        )

        valid_recall = valid_metrics[f"recall_at_{top_k}"]
        valid_ndcg = valid_metrics[f"ndcg_at_{top_k}"]
        history["train_bpr_losses"].append(train_metrics["loss"])
        history[f"valid_recall_at_{top_k}"].append(valid_recall)
        history[f"valid_ndcg_at_{top_k}"].append(valid_ndcg)
        save_json(history, history_path)

        if valid_recall > best_valid_recall:
            best_valid_recall = valid_recall
            save_checkpoint(model, checkpoint_path)
            checkpoint_status = "saved best checkpoint"
        else:
            checkpoint_status = "checkpoint unchanged"

        print(
            f"Epoch {epoch + 1}/{n_epochs}, "
            f"train BPR loss: {train_metrics['loss']:.4f}, "
            f"valid Recall@{top_k}: {valid_recall:.4f}, "
            f"valid NDCG@{top_k}: {valid_ndcg:.4f}, "
            f"evaluated users: {int(valid_metrics['num_users'])}, "
            f"{checkpoint_status}"
        )

    return history
