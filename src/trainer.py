import os
from typing import Optional
import torch
import torch.nn as nn
from tqdm import tqdm


def train(
    data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    vocab_size: int,
    pad_idx: int,
    num_epoch: int = 20,
    device: str = "cuda",
    checkpoint_dir: str = "temp/models",
    label_smoothing: float = 0.1,
    max_grad_norm: float = 5.0,
    resume_checkpoint: Optional[str] = None,
) -> torch.nn.Module:
    """
    Training loop for image captioning.

    Improvements over baseline:
        - PAD tokens excluded from loss via ignore_index
        - Label smoothing to reduce overconfidence
        - Gradient clipping to stabilise LSTM/Transformer decoder
        - Cosine annealing LR schedule
        - Mixed precision (AMP) on CUDA
        - Per-epoch checkpointing; retains best checkpoint by loss
        - Resumable from a saved checkpoint

    Args:
        data_loader       : yields (images, captions, img_ids)
        model             : captioning model; forward(image, caption) -> logits
        optimizer         : pre-constructed optimizer
        vocab_size        : output vocabulary size
        pad_idx           : <PAD> token index — excluded from loss
        num_epoch         : total training epochs
        device            : 'cuda' or 'cpu'
        checkpoint_dir    : directory for .pt checkpoints
        label_smoothing   : smoothing factor for CrossEntropyLoss
        max_grad_norm     : gradient clipping ceiling
        resume_checkpoint : path to .pt checkpoint to resume from
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_idx,
        label_smoothing=label_smoothing,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epoch, eta_min=1e-6
    )

    use_amp = device == "cuda" and torch.cuda.is_available()
    scaler = torch.amp.grad_scaler.GradScaler(device, enabled=use_amp)

    start_epoch = 0
    best_loss = float("inf")

    if resume_checkpoint and os.path.isfile(resume_checkpoint):
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))

    model.to(device)
    model.train()

    for epoch in range(start_epoch, num_epoch):
        running_loss = 0.0
        loop = tqdm(data_loader, leave=True)

        for imgs, labels in loop:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast_mode.autocast(device_type=device, enabled=use_amp):
                out = model(imgs, labels)
                loss = criterion(
                    out.reshape(-1, vocab_size),
                    labels.reshape(-1),
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            loop.set_description(f"Epoch [{epoch + 1}/{num_epoch}]")
            loop.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()

        avg_loss = running_loss / len(data_loader)
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        ckpt_path = os.path.join(checkpoint_dir, f"ckpt_epoch{epoch + 1:03d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "avg_loss": avg_loss,
                "best_loss": best_loss,
            },
            ckpt_path,
        )

        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)

    return model
