"""
evaluation.py
-------------
Standard classification metrics (Accuracy, Precision, Recall, F1, Top-K).
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Top-K metrics

def metrics_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> dict:
    """
    Compute Accuracy@K, Precision@K, Recall@K, F1@K.

    Parameters
    ----------
    y_true : 1-D integer array of true class indices
    y_prob : 2-D float array (num_samples, num_classes) of predicted probabilities
    k      : number of top predictions to consider

    Returns
    -------
    dict with keys:
        Accuracy@1, Accuracy@K,
        Micro Precision@K, Micro Recall@K, Micro F1@K,
        Macro Precision@K, Macro Recall@K, Macro F1@K,
        Weighted Precision@K, Weighted Recall@K, Weighted F1@K
    """
    top1 = np.argmax(y_prob, axis=1)
    topk = np.argsort(y_prob, axis=1)[:, -k:]  # ascending → last k are highest

    acc_top1 = float(np.mean(top1 == y_true))
    acc_topk = float(np.mean([y_true[i] in topk[i] for i in range(len(y_true))]))

    prec_per = [len({y_true[i]} & set(topk[i])) / k for i in range(len(y_true))]
    rec_per  = [float(y_true[i] in topk[i]) for i in range(len(y_true))]
    prec_micro = float(np.mean(prec_per))
    rec_micro  = float(np.mean(rec_per))
    denom = prec_micro + rec_micro
    f1_micro = 2 * prec_micro * rec_micro / denom if denom > 0 else 0.0

    # Macro / Weighted – use top-1 argmax of the top-k subset as the "hard" prediction
    topk_argmax = np.array([topk[i][np.argmax(y_prob[i, topk[i]])] for i in range(len(y_true))])
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, topk_argmax, average="macro", zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, topk_argmax, average="weighted", zero_division=0
    )

    return {
        "Accuracy@1": acc_top1,
        "Accuracy@K": acc_topk,
        "Micro Precision@K": prec_micro,
        "Micro Recall@K": rec_micro,
        "Micro F1@K": f1_micro,
        "Macro Precision@K": float(prec_macro),
        "Macro Recall@K": float(rec_macro),
        "Macro F1@K": float(f1_macro),
        "Weighted Precision@K": float(prec_w),
        "Weighted Recall@K": float(rec_w),
        "Weighted F1@K": float(f1_w),
    }


# Full evaluation on a test set

def evaluate(
    X_test: np.ndarray,
    y_test: np.ndarray,
    checkpoint_path: str,
    num_classes: int = 20,
) -> dict:
    """
    Load a saved model and compute all standard metrics on *X_test*.

    Returns a structured dict with keys:
        Accuracy, Micro-Averaged, Macro-Averaged, Weighted-Averaged,
        Top-3 Metrics, Top-5 Metrics
    """
    import torch
    from src.model import PersonalizedBERT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PersonalizedBERT(input_size=X_test.shape[1], num_classes=num_classes).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    X_t = torch.tensor(X_test.astype(np.float32)).to(device)
    with torch.no_grad():
        logits = model(X_t)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()

    y = y_test.astype(int)
    acc = accuracy_score(y, preds)
    p_mi, r_mi, f_mi, _ = precision_recall_fscore_support(y, preds, average="micro",     zero_division=0)
    p_ma, r_ma, f_ma, _ = precision_recall_fscore_support(y, preds, average="macro",     zero_division=0)
    p_w,  r_w,  f_w,  _ = precision_recall_fscore_support(y, preds, average="weighted",  zero_division=0)

    top3 = metrics_at_k(y, probs, k=3)
    top5 = metrics_at_k(y, probs, k=5)

    return {
        "Accuracy": float(acc),
        "Micro-Averaged":    {"Precision": float(p_mi), "Recall": float(r_mi), "F1 Score": float(f_mi)},
        "Macro-Averaged":    {"Precision": float(p_ma), "Recall": float(r_ma), "F1 Score": float(f_ma)},
        "Weighted-Averaged": {"Precision": float(p_w),  "Recall": float(r_w),  "F1 Score": float(f_w)},
        "Top-3 Metrics": {
            "Top-3 Accuracy":   top3["Accuracy@K"],
            "Micro Precision":  top3["Micro Precision@K"],
            "Micro Recall":     top3["Micro Recall@K"],
            "Micro F1":         top3["Micro F1@K"],
            "Macro Precision":  top3["Macro Precision@K"],
            "Macro F1":         top3["Macro F1@K"],
            "Weighted Precision": top3["Weighted Precision@K"],
            "Weighted F1":      top3["Weighted F1@K"],
        },
        "Top-5 Metrics": {
            "Top-5 Accuracy":   top5["Accuracy@K"],
            "Micro Precision":  top5["Micro Precision@K"],
            "Micro Recall":     top5["Micro Recall@K"],
            "Micro F1":         top5["Micro F1@K"],
            "Macro Precision":  top5["Macro Precision@K"],
            "Macro F1":         top5["Macro F1@K"],
            "Weighted Precision": top5["Weighted Precision@K"],
            "Weighted F1":      top5["Weighted F1@K"],
        },
    }


# Results aggregation

def summarise_repetitions(all_results: list) -> dict:
    """
    Given a list of metric dicts (one per bootstrap repetition), compute
    mean ± std for every scalar leaf value.

    Returns a nested dict mirroring the input structure, with each leaf
    replaced by {"mean": ..., "std": ...}.
    """
    from collections.abc import Mapping

    summary: dict = {}
    for key in all_results[0]:
        values = [r[key] for r in all_results]
        if isinstance(values[0], Mapping):
            summary[key] = {}
            for sub_key in values[0]:
                sub_vals = [v[sub_key] for v in values]
                summary[key][sub_key] = {
                    "mean": float(np.mean(sub_vals)),
                    "std":  float(np.std(sub_vals)),
                }
        else:
            summary[key] = {
                "mean": float(np.mean(values)),
                "std":  float(np.std(values)),
            }
    return summary
