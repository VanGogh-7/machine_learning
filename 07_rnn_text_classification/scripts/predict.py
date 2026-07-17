import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_datasets
from src.model import RNNTextClassifier
from src.utils import get_device, load_checkpoint


LABEL_NAMES = {0: "negative", 1: "positive"}
EXAMPLE_TEXTS = [
    "this movie is very good",
    "the story is boring and bad",
    "I really enjoyed this film",
    "this was a terrible experience",
]


def main() -> None:
    config = TrainConfig()
    device = get_device()
    print(f"Using device: {device}")

    _, _, _, vocab = build_datasets(
        dataset_path=config.dataset_path,
        valid_ratio=config.valid_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
        vocab_min_freq=config.vocab_min_freq,
        max_length=config.max_length,
    )
    model = RNNTextClassifier(
        vocab_size=len(vocab),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        n_classes=config.n_classes,
        pad_idx=vocab.pad_idx,
    ).to(device)
    checkpoint_path = PROJECT_ROOT / config.model_path
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    for text in EXAMPLE_TEXTS:
        input_ids = torch.tensor(
            [vocab.encode(text, config.max_length)],
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            probabilities = torch.softmax(model(input_ids), dim=1)
        predicted_label = probabilities.argmax(dim=1).item()
        probability = probabilities[0, predicted_label].item()
        print(f"Text: {text}")
        print(f"Predicted label: {predicted_label} ({LABEL_NAMES[predicted_label]})")
        print(f"Probability: {probability:.4f}")


if __name__ == "__main__":
    main()
