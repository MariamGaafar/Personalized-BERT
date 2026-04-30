"""
run_experiments.py
------------------
Main entry-point to reproduce all experiments from:

    Gaafar et al., "Personalized-BERT: Personalized Emoji Prediction
    Framework Using Emotion, and User Preferences, and Personality"
    Procedia Computer Science 275 (2026) 762-770.

Usage
-----
    python run_experiments.py --config config.json

See README.md for full instructions and expected data layout.
"""

import argparse
import json
import os
import sys

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.data_utils import load_json, load_jsonl, jsonl_list_to_dict, label_encode
from src.features import (
    compute_leave_one_out_personality,
    process_emoji_stats,
    combine_features,
)
from src.bootstrapping import run_bootstrap, save_results


# Default configuration

DEFAULT_CONFIG = {
    # ---- Paths ----
    "raw_text_path":          "data/pan17_documents.json",
    "bert_embeddings_path":   "data/Pan17_bert_embeddings.jsonl",
    "personality_path":       "data/pan17_personality_predictions_sentence_BERT.jsonl",
    "emotion_path":           "data/pan17_emotion_predictions.jsonl",
    "emoji_stats_path":       "data/emoji_stats_holdout.json",
    "emoji_counts_path":      "data/emoji_counts.xlsx",
    "similarity_path":        "data/emoji_similarity.json",

    # ---- Experiment settings ----
    "emoji_thresholds":       [20, 50, 64, 100, 150, 200, 250, 300],
    "repetitions":            10,
    "num_epochs":             10,
    "batch_size":             128,
    "val_split":              0.1,
    "semantic_threshold":     0.7,
    "checkpoint_dir":         "checkpoints",
    "results_dir":            "results",

    # ---- Feature ablations to run ----
    # Each entry: {"name": str, "use_personality": bool, "use_emotion": bool, "use_usage": bool}
    "experiments": [
        {"name": "BERT",                             "use_personality": False, "use_emotion": False, "use_usage": False},
        {"name": "BERT_Emotion",                     "use_personality": False, "use_emotion": True,  "use_usage": False},
        {"name": "BERT_Personality",                 "use_personality": True,  "use_emotion": False, "use_usage": False},
        {"name": "BERT_Usage",                       "use_personality": False, "use_emotion": False, "use_usage": True},
        {"name": "Personalized_BERT",                "use_personality": True,  "use_emotion": True,  "use_usage": True},
    ],
}

# Data loading


def load_all_data(cfg: dict) -> dict:
    print("Loading raw text …")
    raw_text = load_json(cfg["raw_text_path"])

    print("Loading BERT embeddings …")
    bert_embeddings = load_jsonl(cfg["bert_embeddings_path"])

    print("Loading personality predictions …")
    personality_raw = load_jsonl(cfg["personality_path"])
    mean_personality = compute_leave_one_out_personality(
        personality_raw, save_path="data/leave_one_out_personality.json"
    )

    print("Loading emotion predictions …")
    emotions_list = load_jsonl(cfg["emotion_path"])
    emotions_dict = jsonl_list_to_dict(emotions_list)

    print("Loading emoji usage stats …")
    emoji_stats_raw = load_json(cfg["emoji_stats_path"])

    print("Loading emoji similarity triples …")
    similarity_triples = load_json(cfg["similarity_path"])

    print("Loading emoji vocabulary …")
    emoji_df = pd.read_excel(cfg["emoji_counts_path"])
    emoji_vocab_full = list(emoji_df["Emoji"][: cfg.get("emoji_thresholds", [300])[-1]])

    # Fit a global label encoder on the maximum vocabulary
    le_global = LabelEncoder()
    le_global.fit(emoji_vocab_full)

    emoji_stats_processed = process_emoji_stats(emoji_stats_raw, le_global)

    return {
        "raw_text":          raw_text,
        "bert_embeddings":   bert_embeddings,
        "mean_personality":  mean_personality,
        "emotions_dict":     emotions_dict,
        "emoji_stats":       emoji_stats_processed,
        "similarity_triples": similarity_triples,
        "emoji_vocab_full":  emoji_vocab_full,
        "le_global":         le_global,
    }


# Feature vector builder

def build_embeddings_for_experiment(data: dict, exp: dict) -> list:
    """
    Starting from BERT embeddings, optionally append personality, emotion,
    and usage-pattern features according to the experiment configuration.
    """
    embeddings = data["bert_embeddings"]

    if exp["use_personality"]:
        print(f"  + personality features")
        embeddings = combine_features(embeddings, data["mean_personality"])

    if exp["use_emotion"]:
        print(f"  + emotion features")
        embeddings = combine_features(embeddings, data["emotions_dict"])

    if exp["use_usage"]:
        print(f"  + usage-pattern features")
        embeddings = combine_features(embeddings, data["emoji_stats"])

    return embeddings


# Main

def main(cfg: dict) -> None:
    os.makedirs(cfg["results_dir"],   exist_ok=True)
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    data = load_all_data(cfg)

    for threshold in cfg["emoji_thresholds"]:
        emoji_vocab = data["emoji_vocab_full"][:threshold]
        print(f"\n{'#'*70}")
        print(f"  Emoji threshold: top {threshold}")
        print(f"{'#'*70}")

        for exp in cfg["experiments"]:
            name = exp["name"]
            print(f"\n  Experiment: {name}")
            embeddings = build_embeddings_for_experiment(data, exp)

            results, similarities, semantic, df = run_bootstrap(
                embeddings=embeddings,
                raw_data=data["raw_text"],
                emoji_vocab=emoji_vocab,
                similarity_triples=data["similarity_triples"],
                model_name=name,
                checkpoint_dir=cfg["checkpoint_dir"],
                emoji_threshold=threshold,
                num_epochs=cfg["num_epochs"],
                repetitions=cfg["repetitions"],
                batch_size=cfg["batch_size"],
                val_split=cfg["val_split"],
                semantic_threshold=cfg["semantic_threshold"],
            )

            prefix = os.path.join(cfg["results_dir"], f"{name}_top{threshold}")
            save_results(results, similarities, semantic, df, prefix=prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Personalized-BERT experiments")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON config file (uses built-in defaults if omitted)",
    )
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            user_cfg = json.load(f)
        cfg = {**DEFAULT_CONFIG, **user_cfg}
    else:
        cfg = DEFAULT_CONFIG

    main(cfg)
