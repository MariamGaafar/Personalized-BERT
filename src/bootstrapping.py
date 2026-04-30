"""
bootstrapping.py
----------------
Bootstrap evaluation loop (10 resampling iterations, paper Section 3).

For each repetition the dataset is randomly re-split (no fixed seed) to
estimate the robustness and statistical significance of performance gains.
Memory is released after every repetition.
"""

import gc
import json
from collections import defaultdict
from itertools import chain

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.data_utils import build_aligned_dataset, label_encode
from src.evaluation import evaluate, summarise_repetitions
from src.model import train_model, get_topk_predictions
from src.semantic_eval import get_similarity_scores, evaluate_semantic, build_emoji_clusters


# Helper: convert numpy scalars for JSON serialisation

def _json_safe(obj):
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


# Single-model bootstrap runner

def run_bootstrap(
    embeddings: list,
    raw_data: dict,
    emoji_vocab: list,
    similarity_triples: list,
    model_name: str = "BERT",
    checkpoint_dir: str = "checkpoints",
    emoji_threshold: int = 20,
    num_epochs: int = 10,
    repetitions: int = 10,
    batch_size: int = 128,
    val_split: float = 0.1,
    semantic_threshold: float = 0.7,
) -> tuple:
    """
    Run *repetitions* independent train/test splits, train a model on each,
    and collect performance metrics.

    Parameters
    ----------
    embeddings         : list of {author_id: [[emb], ...]}  (pre-computed)
    raw_data           : dict {author_id: [tweet_str, ...]}
    emoji_vocab        : ordered list of emojis defining the label set
    similarity_triples : EmoSim508 triples [[e1, e2, score], ...]
    model_name         : used for checkpoint file naming
    checkpoint_dir     : directory to store .pth checkpoints
    emoji_threshold    : number of top emojis to use (dataset size)
    num_epochs         : training epochs per repetition
    repetitions        : number of bootstrap iterations
    batch_size         : mini-batch size
    val_split          : fraction of training data held out for validation
    semantic_threshold : EmoSim508 similarity threshold for cluster construction

    Returns
    -------
    all_results     : list of metric dicts (one per repetition)
    all_similarities: list of similarity dicts (one per repetition)
    all_semantic    : list of semantic-eval dicts (one per repetition)
    df_predictions  : DataFrame with columns [actual, predicted, top3, top5]
    """
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)

    cluster_map = build_emoji_clusters(similarity_triples, threshold=semantic_threshold)

    all_results: list = []
    all_similarities: list = []
    all_semantic: list = []
    actual_all, pred_all, top3_all, top5_all = [], [], [], []

    for rep in range(repetitions):
        print(f"\n{'='*60}")
        print(f"  Repetition {rep + 1} / {repetitions}  |  model: {model_name}  |  N={emoji_threshold}")
        print(f"{'='*60}")

        ckpt = os.path.join(
            checkpoint_dir,
            f"{model_name}_top{emoji_threshold}_epochs{num_epochs}_rep{rep+1}.pth",
        )

        # -- Build aligned dataset --
        _, X, y, _ = build_aligned_dataset(embeddings, raw_data, emoji_vocab)

        # -- Fit label encoder on full vocab union --
        all_labels = sorted(set(emoji_vocab) | set(y))
        le = LabelEncoder()
        le.fit(all_labels)
        y_enc = label_encode(y, le)

        # Remove samples with unknown labels (-1)
        mask = y_enc >= 0
        X, y_enc = X[mask], y_enc[mask]

        if len(X) < 10:
            print(f"  Skipping rep {rep + 1}: only {len(X)} samples after filtering — "
                  "not enough to split. Use more data or a smaller emoji_threshold.")
            continue

        num_classes = len(le.classes_)

        # -- Random train/test split (no fixed seed → bootstrap effect) --
        # Only stratify when every class has ≥2 samples
        class_counts = np.bincount(y_enc, minlength=num_classes)
        use_stratify = bool(np.all(class_counts[np.unique(y_enc)] >= 2))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, stratify=y_enc if use_stratify else None
        )

        # -- Train --
        train_model(
            X_train, X_test, y_train, y_test,
            checkpoint_path=ckpt,
            batch_size=batch_size,
            num_epochs=num_epochs,
            num_classes=num_classes,
            val_split=val_split,
        )

        # -- Standard evaluation --
        results = evaluate(X_test, y_test, ckpt, num_classes=num_classes)
        all_results.append(results)

        # -- Similarity-based evaluation --
        sims = get_similarity_scores(
            X_test, y_test, ckpt, le, similarity_triples, num_classes=num_classes
        )
        all_similarities.append(sims)

        # -- Semantic cluster evaluation --
        sem = evaluate_semantic(
            X_test, y_test, ckpt, le, cluster_map, num_classes=num_classes
        )
        all_semantic.append(sem)

        # -- Collect predictions for output DataFrame --
        actual, top1, top3 = get_topk_predictions(X_test, y_test, ckpt, le, k=3, num_classes=num_classes)
        _, _, top5 = get_topk_predictions(X_test, y_test, ckpt, le, k=5, num_classes=num_classes)
        actual_all.extend(actual)
        pred_all.extend(top1)
        top3_all.extend(top3)
        top5_all.extend(top5)

        # -- Free memory --
        del X, y_enc, X_train, X_test, y_train, y_test
        gc.collect()

    # -- Summary --
    print("\n" + "="*60)
    print("  Bootstrap Summary")
    print("="*60)
    if not all_results:
        print("  No repetitions completed (all skipped due to insufficient data).")
    else:
        summary = summarise_repetitions(all_results)
        for metric, val in summary.items():
            if isinstance(val, dict) and "mean" not in val:
                print(f"  {metric}:")
                for sub, sv in val.items():
                    print(f"    {sub}: {sv['mean']:.4f} ± {sv['std']:.4f}")
            else:
                print(f"  {metric}: {val['mean']:.4f} ± {val['std']:.4f}")

    df = pd.DataFrame({
        "actual":    actual_all,
        "predicted": pred_all,
        "top3":      top3_all,
        "top5":      top5_all,
    })

    return all_results, all_similarities, all_semantic, df


# Results persistence

def save_results(
    all_results: list,
    all_similarities: list,
    all_semantic: list,
    df: pd.DataFrame,
    prefix: str = "results",
) -> None:
    """Save metric JSONs and prediction Excel file."""
    with open(f"{prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=_json_safe)

    with open(f"{prefix}_similarity.json", "w", encoding="utf-8") as f:
        json.dump(all_similarities, f, ensure_ascii=False, indent=2, default=_json_safe)

    with open(f"{prefix}_semantic.json", "w", encoding="utf-8") as f:
        json.dump(all_semantic, f, ensure_ascii=False, indent=2, default=_json_safe)

    df.to_excel(f"{prefix}_predictions.xlsx", index=False)
    print(f"Results saved with prefix '{prefix}'.")
