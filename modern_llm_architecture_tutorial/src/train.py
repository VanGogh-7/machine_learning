from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .config import LLMConfig, load_config
from .dataset import create_dataloaders
from .model import ModernDecoderLM
from .utils import checkpoint_path, count_parameters, load_checkpoint, perplexity, save_checkpoint, set_seed


def build_scheduler(optimizer, config: LLMConfig, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return max(1, step + 1) / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        min_ratio = config.min_learning_rate / config.learning_rate
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def evaluate(model, loader, device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x, targets=y)
            losses.append(float(out["loss"].item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def train(config_path: str) -> Path:
    config = load_config(config_path)
    set_seed(config.seed)
    tokenizer, train_loader, val_loader = create_dataloaders(config)
    config.vocab_size = tokenizer.vocab_size
    config.validate()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModernDecoderLM(config).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = max(1, config.epochs * len(train_loader))
    scheduler = build_scheduler(optimizer, config, total_steps)
    use_amp = config.use_amp and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ckpt_path = checkpoint_path(config.checkpoint_dir, config.checkpoint_name)
    start_epoch = 0
    if config.resume and ckpt_path.exists():
        start_epoch = load_checkpoint(ckpt_path, model, optimizer, scheduler, map_location=str(device))
        print(f"Resumed from {ckpt_path} at epoch {start_epoch}.")

    print(f"Device: {device}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Training batches per epoch: {len(train_loader)}")

    global_step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, config.epochs):
        model.train()
        running = 0.0
        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(x, targets=y)
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += float(loss.item())
            global_step += 1
            if step == 1 or step % 50 == 0:
                avg = running / step
                lr = scheduler.get_last_lr()[0]
                print(
                    f"epoch {epoch + 1}/{config.epochs} step {step}/{len(train_loader)} "
                    f"loss {avg:.4f} ppl {perplexity(avg):.2f} lr {lr:.2e}"
                )

        val_loss = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch + 1} validation loss {val_loss:.4f} "
            f"perplexity {perplexity(val_loss):.2f}"
        )
        save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch + 1, config.to_dict())
        print(f"Saved checkpoint to {ckpt_path}")

    return ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()

