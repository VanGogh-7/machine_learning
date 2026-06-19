import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig


def main() -> None:
    config = TrainConfig()
    output_dir = PROJECT_ROOT / config.output_dir
    checkpoint_path = output_dir / config.model_path

    if not config.processed_feature_meta_path.is_file():
        print("Processed SASRec feature metadata not found.")
        print("Run:")
        print("python scripts/prepare_data.py")
        return
    if not config.processed_test_path.is_file():
        print("Processed SASRec test CSV not found.")
        print("Run:")
        print("python scripts/prepare_data.py")
        return
    if not checkpoint_path.is_file():
        print("Saved SASRec checkpoint not found.")
        print("Run:")
        print("python scripts/train.py")
        return

    import pandas as pd
    import torch

    from src.data import parse_sequence
    from src.model import SASRecModel
    from src.utils import get_device, load_checkpoint, load_json

    metadata = load_json(config.processed_feature_meta_path)
    device = get_device()
    model = SASRecModel(
        num_items=metadata["num_items"],
        max_sequence_length=metadata["max_sequence_length"],
        embedding_dim=config.embedding_dim,
        num_attention_heads=config.num_attention_heads,
        num_transformer_blocks=config.num_transformer_blocks,
        feedforward_dim=config.feedforward_dim,
        dropout=config.dropout,
    ).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    examples = pd.read_csv(config.processed_test_path).head(5)
    candidate_item_ids = torch.arange(
        1,
        metadata["num_items"],
        dtype=torch.long,
        device=device,
    )
    item_id_to_title = metadata.get("item_id_to_title", {})

    with torch.no_grad():
        for row in examples.itertuples(index=False):
            sequence = parse_sequence(
                getattr(row, "input_item_ids"),
                expected_length=metadata["max_sequence_length"],
            ).unsqueeze(0)
            input_item_ids = sequence.to(device)
            sequence_vector = model.encode_sequence(input_item_ids)

            all_scores = []
            for start in range(0, candidate_item_ids.numel(), config.batch_size):
                batch_candidates = candidate_item_ids[
                    start : start + config.batch_size
                ]
                candidate_embeddings = model.item_embedding(batch_candidates)
                batch_scores = (sequence_vector @ candidate_embeddings.T).squeeze(0)
                all_scores.append(batch_scores)

            scores = torch.cat(all_scores)
            effective_top_k = min(config.top_k, scores.numel())
            top_scores, top_positions = scores.topk(effective_top_k)
            top_item_ids = candidate_item_ids[top_positions]

            print(f"input item sequence: {getattr(row, 'input_item_ids')}")
            print(f"top_{effective_top_k} recommended encoded item ids:")
            for item_id, score in zip(top_item_ids.cpu().tolist(), top_scores.cpu().tolist()):
                title = item_id_to_title.get(str(int(item_id)))
                title_text = "" if title is None else f", title: {title}"
                print(f"  item_id: {int(item_id)}, score: {score:.4f}{title_text}")


if __name__ == "__main__":
    main()
