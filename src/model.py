"""
model.py
--------
Neural network architecture and training loop for Personalized-BERT.

Architecture
~~~~~~~~~~~~
Input features are concatenated into a single vector:
  • BERT text embeddings        (768-d)
  • Emotion probabilities       (6-d)   [optional]
  • Big-Five personality traits (5-d)   [optional]
  • Usage-pattern features      (4-d)   [optional]

The concatenated vector is passed through one fully-connected hidden layer
followed by a softmax output layer (CrossEntropyLoss during training).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

# Model definition

class PersonalizedBERT(nn.Module):
    """
    Single hidden-layer classifier used for all feature combinations.

    Parameters
    ----------
    input_size  : dimension of the concatenated feature vector
    hidden_size : number of neurons in the hidden layer (default 768)
    num_classes : number of emoji classes
    """

    def __init__(self, input_size: int = 768, hidden_size: int = 768, num_classes: int = 20):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        return self.out(x)  # logits – softmax applied externally at inference

# Checkpoint helpers

def save_checkpoint(path: str, epoch: int, model: nn.Module, optimizer: optim.Optimizer):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer) -> int:
    """Load checkpoint and return the next epoch to start from."""
    if not os.path.exists(path):
        return 0
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start = ckpt["epoch"] + 1
    print(f"Resuming from epoch {start}")
    return start


# Training loop

def train_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    checkpoint_path: str = "checkpoint.pth",
    batch_size: int = 128,
    num_epochs: int = 100,
    num_classes: int = 20,
    val_split: float = 0.1,
    plot: bool = False,
) -> tuple:
    """
    Train the PersonalizedBERT classifier and return per-epoch metric history.

    Returns
    -------
    final_metrics : dict  – metrics at the last epoch
    max_metrics   : dict  – best metrics seen during training
    max_epochs    : dict  – epoch index at which each best metric was achieved
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Tensors --
    X_tr = torch.tensor(X_train.astype(np.float32))
    X_te = torch.tensor(X_test.astype(np.float32))
    y_tr = torch.tensor(y_train.astype(np.int64))
    y_te = torch.tensor(y_test.astype(np.int64))

    n_val = int(val_split * len(X_tr))
    n_train = len(X_tr) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        TensorDataset(X_tr, y_tr), [n_train, n_val]
    )
    test_ds = TensorDataset(X_te, y_te)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = PersonalizedBERT(
        input_size=X_train.shape[1], hidden_size=768, num_classes=num_classes
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    _resume_path = checkpoint_path + ".resume.pth"
    start_epoch = load_checkpoint(_resume_path, model, optimizer)

    # Metric history
    tr_losses, tr_accs, tr_precs, tr_recs, tr_f1s = [], [], [], [], []
    va_losses, va_accs, va_precs, va_recs, va_f1s = [], [], [], [], []

    for epoch in range(start_epoch, num_epochs):
        # ---- Training ----
        model.train()
        ep_loss, all_preds, all_labels = 0.0, [], []
        for bX, by in train_loader:
            bX, by = bX.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bX)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * bX.size(0)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(by.cpu().numpy())

        ep_loss /= len(train_ds)
        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        tr_losses.append(ep_loss); tr_accs.append(acc)
        tr_precs.append(prec); tr_recs.append(rec); tr_f1s.append(f1)

        # ---- Validation ----
        model.eval()
        va_loss, va_preds, va_labels = 0.0, [], []
        with torch.no_grad():
            for bX, by in val_loader:
                bX, by = bX.to(device), by.to(device)
                out = model(bX)
                va_loss += criterion(out, by).item() * bX.size(0)
                va_preds.extend(out.argmax(1).cpu().numpy())
                va_labels.extend(by.cpu().numpy())
        va_loss /= len(val_ds)
        va_acc = accuracy_score(va_labels, va_preds)
        va_prec, va_rec, va_f1, _ = precision_recall_fscore_support(
            va_labels, va_preds, average="weighted", zero_division=0
        )
        va_losses.append(va_loss); va_accs.append(va_acc)
        va_precs.append(va_prec); va_recs.append(va_rec); va_f1s.append(va_f1)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Loss {ep_loss:.4f} | Acc {acc:.4f} | F1 {f1:.4f} | "
            f"Val Loss {va_loss:.4f} | Val Acc {va_acc:.4f}"
        )
        save_checkpoint(_resume_path, epoch, model, optimizer)

    # Save final weights (plain state dict for evaluate())
    torch.save(model.state_dict(), checkpoint_path)
    # Clean up the resume checkpoint
    if os.path.exists(_resume_path):
        os.remove(_resume_path)
    print(f"Model saved to {checkpoint_path}")

    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score",
                    "Val Accuracy", "Val Precision", "Val Recall", "Val F1 Score"]
    history = [tr_accs, tr_precs, tr_recs, tr_f1s, va_accs, va_precs, va_recs, va_f1s]

    final_metrics = {n: h[-1] for n, h in zip(metric_names, history)}
    max_metrics = {n: max(h) for n, h in zip(metric_names, history)}
    max_epochs = {n: start_epoch + h.index(max(h)) for n, h in zip(metric_names, history)}

    if plot:
        _plot_training_curves(
            list(range(start_epoch, num_epochs)),
            tr_losses, va_losses, tr_accs, va_accs,
        )

    return final_metrics, max_metrics, max_epochs


def _plot_training_curves(epochs, tr_loss, va_loss, tr_acc, va_acc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, tr_loss, label="Train Loss")
    ax1.plot(epochs, va_loss, label="Val Loss", linestyle="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss")
    ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, tr_acc, label="Train Acc")
    ax2.plot(epochs, va_acc, label="Val Acc", linestyle="--")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title("Accuracy")
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.show()


# Inference helpers

def get_topk_predictions(
    X_test: np.ndarray,
    y_test: np.ndarray,
    checkpoint_path: str,
    label_encoder,
    k: int,
    num_classes: int,
) -> tuple:
    """
    Load saved weights and return (actual_emojis, top1_emojis, topk_emojis).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PersonalizedBERT(input_size=X_test.shape[1], num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    X = torch.tensor(X_test.astype(np.float32)).to(device)
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    top1 = np.argmax(probs, axis=1)
    topk = np.argsort(probs, axis=1)[:, -k:][:, ::-1]

    actual_emojis = label_encoder.inverse_transform(y_test.astype(int)).tolist()
    top1_emojis = label_encoder.inverse_transform(top1).tolist()
    topk_emojis = [label_encoder.inverse_transform(row).tolist() for row in topk]

    return actual_emojis, top1_emojis, topk_emojis
